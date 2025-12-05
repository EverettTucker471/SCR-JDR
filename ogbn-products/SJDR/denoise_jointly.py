from torch_geometric.data import Data
import torch_geometric
import copy
from tqdm import tqdm
import torch
from types import SimpleNamespace

# Sparse import for top-k eigen computation
from torch_sparse import spmm


def randomized_sparse_svd(X, k, iter=2):
    """
    Computes the svd of the feature matrix using the block power method
    Ensures sparse computations

    Args:
        X (ndarray): The feature matrix to return
        k (int): The number of singular values to compute
        n (int): The number of iterations to use for computation
    """

    N, F = X.shape

    omega = torch.randn(F, k, device=X.device)
    Y = X @ omega
    Q, _ = torch.linalg.qr(Y)

    for i in range(iter):
        Z = X.T @ Q
        Q, _ = torch.linalg.qr(X @ (torch.linalg.qr(Z)[0]))

    B = Q.T @ X
    Ub, S, Vt = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub

    return U, S, Vt.T


def sparse_mat_vec(edge_index, edge_weight, x, num_nodes):
    """
    Performs sparse matrix / vector multiplication where edge_index is a sparse representation
    and x is dense
    """

    # Use torch.sparse.mm probably
    return spmm(edge_index, edge_weight, num_nodes, num_nodes, x)


def topk_eigen_sparse(edge_index, edge_weight, N, k, iter=8):
    """
    Computes the k most significant (largest) eigenvalues from the edge indices

    Args:
        edge_index (ndarray) The indices (i, j) that denote an edge between Ui and Uj
        N (int): The number of nodes in the graph
        k (int): The number of eigenvectors to select
    """

    # starting with a randomized matrix
    Q = torch.randn(N, k, device=edge_index.device)
    Q, _ = torch.linalg.qr(Q)  # Orthonormalize

    for i in range(iter):
        # Multiplying and normalizing
        Z = sparse_mat_vec(edge_index, edge_weight, Q, N)
        Q, _ = torch.linalg.qr(Z)

    B = Q.T @ sparse_mat_vec(edge_index, edge_weight, Q, N)

    evals, S = torch.linalg.eigh(B)
    evals = evals.flip(0)
    S = S.flip(1)
    V = Q @ S

    return evals, V


def per_edge_update(edge_index, VA_old, VA_new, la):
    """
    Computes the delta contributions for each edge
    """

    src, dst = edge_index

    src_old = VA_old[src]
    dst_old = VA_old[dst]
    src_new = VA_new[src]
    dst_new = VA_new[dst]

    diff = (src_new * dst_new) - (src_old * dst_old)

    return diff @ la.to(diff.device)


def prune_edges(edge_index, edge_weight, score, N, keep=0.9, min_degree=1):
    """
    Prunes noisy edges based on their score (E,)
    Keeps a number of nodes globally, but also a certain number to/from each node
    """

    E = edge_weight.size(0)  # Number of edges
    
    # Sampling to estimate quantile, because otherwise the tensor is too large
    sample = score[torch.randperm(E)[:2_000_000]]
    thresh = torch.quantile(sample, q=1.0 - keep)
    mask = score >= thresh
    # ensure min_degree: naive safeguard (optional, could be expensive)
    # compute degrees of surviving edges
    new_edge_index = edge_index[:, mask]
    new_edge_weight = edge_weight[mask]
    # ensure not isolating nodes - if some node degree becomes 0, keep its highest-scoring incident edge
    deg = torch.zeros((N,), dtype=torch.long, device=edge_index.device)
    deg = deg.index_add(0, new_edge_index[0], torch.ones(new_edge_index.size(1), device=edge_index.device, dtype=torch.long))
    isolated = torch.nonzero(deg == 0).squeeze()
    if isolated.numel() > 0:
        # for each isolated node, find its best original incident edge and re-add it
        src_all, dst_all = edge_index
        for node in isolated.tolist():
            # find incident edge indices
            mask_incident = (src_all == node)
            if mask_incident.any():
                cidx = torch.nonzero(mask_incident, as_tuple=False).flatten()
                
                best_local = torch.argmax(score[cidx])
                best_idx = cidx[best_local]

                # append best edge
                new_edge_index = torch.cat([new_edge_index, edge_index[:, best_idx].unsqueeze(1)], dim=1)
                new_edge_weight = torch.cat([new_edge_weight, edge_weight[best_idx].unsqueeze(0)])
    return new_edge_index, new_edge_weight


def find_largest(matrix: torch.tensor, k: int) -> torch.tensor:
    """
    Find the k-th largest value in a matrix
    Args:
        matrix: input matrix
        k: number of values to return
    Returns:
    """
    flat_matrix = matrix.flatten()
    values, _ = torch.topk(flat_matrix, k, largest=True, sorted=True)
    return values[-1]


def get_top_k_matrix(matrix: torch.tensor, k: int = 128) -> torch.tensor:
    """
    Get the top k value matrix, all other values are set to 0
    Args:
        matrix: input matrix (N,N)
        k: number of values k to keep

    Returns:
    """
    num_nodes = matrix.shape[0]
    row_idx = torch.arange(num_nodes)
    matrix[matrix.argsort(axis=0)[:num_nodes - k], row_idx] = 0.0
    return matrix


def get_top_k_features(matrix: torch.Tensor, k: int = 128) -> torch.Tensor:
    """
    Get the top k value matrix (rectangular), all other values are set to 0
    Args:
        matrix: input matrix shape (N, F)
        k: number of values k to keep

    Returns:
    """
    _, top_k_indices = matrix.topk(k, dim=1)
    mask = torch.zeros_like(matrix)
    mask.scatter_(1, top_k_indices, 1.)
    matrix *= mask
    return matrix


def get_clipped_matrix(matrix: torch.tensor, eps: float = 0.01) -> torch.tensor:
    """
    Clip the matrix values to 0 if they are below a certain threshold
    Args:
        matrix: input matrix, possibly rectangular
        eps: the threshold value

    Returns:
    """
    matrix[matrix < eps] = 0.0
    return matrix


def get_clipped_features(matrix: torch.tensor, eps: float = 0.01) -> torch.tensor:
    """
        Clip the matrix values to 0 if they are below a certain threshold
        Args:
            matrix: input matrix, possibly rectangular
            eps: the threshold value

        Returns:
    """
    matrix[matrix < eps] = 0.0
    return matrix


def compute_alignment(X: torch.tensor, A: torch.tensor, args, ord=2):
    """
    Compute the alignment between the node features X and the adjacency matrix A.
    We use the min between L_A and L_X to compute the alignment.
    Args:
        X: feature matrix (N, F)
        A: adjacency matrix (N, N)
        args: arguments from argparse
        ord: e.g. 2 or 'fro' (default 2)

    Returns: alignment value (float)
    """
    VX, s, U = torch.linalg.svd(X)
    la, VA = torch.linalg.eigh(A)
    if args.abs_ordering:
        sort_idx = torch.argsort(torch.abs(la))
        VA = VA[:, sort_idx]
    else:
        pass
    rewired_index = min(args.rewired_index_X, args.rewired_index_A)
    VX = VX[:, :rewired_index]
    if args.denoise_offset == 1:
        VA = VA[:, -rewired_index-1:-1]
    else:
        VA = VA[:, -rewired_index:]
    alignment = torch.linalg.norm(torch.matmul(VX.T, VA), ord=ord)
    return alignment.item()


def denoise_jointly(data, args, device):
    """
    Denoise the graph data jointly using the given arguments
    Args:
        data: PyG data object
        args: arguments from argparse
        device: device to run the operations on (cuda or cpu)

    Returns: denoised PyG data object
    """
    offset = args.denoise_offset
    X = data.x.to(device)
    A = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index)[0].to(device)
    X_denoised = data.x.to(device)
    A_denoised = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index)[0].to(device)
    non_binary = args.denoise_non_binary

    # Main loop for denoising
    for iteration in tqdm(range(args.denoise_iterations), desc="Denoising"):
        # Decomposition
        VX, s, U = torch.linalg.svd(X_denoised)
        la, VA = torch.linalg.eigh(A_denoised)
        # Sort the eigenvalues and eigenvectors if needed
        if args.abs_ordering:
            if iteration == 0:
                sort_idx = torch.argsort(torch.abs(la))
                resort_idx = torch.argsort(la[sort_idx])
            VA = VA[:, sort_idx]
            la_abs_sort = la[sort_idx]
        else:
            la_abs_sort = la
        N = A_denoised.shape[0]

        # Denoise the node features X first
        VX_new = copy.deepcopy(VX).to(device)
        for i in range(args.rewired_index_X):
            vx = copy.deepcopy(VX[:, i]).to(device)
            va = VA[:, N - args.rewired_index_X-offset: N-offset]
            overlap = torch.matmul(vx, va)
            maxoverlap_index = torch.argmax(torch.abs(overlap))-offset
            maxoverlap = overlap[maxoverlap_index]
            maxoverlap_index = N - args.rewired_index_X + maxoverlap_index
            VX_new[:, i] = (vx * (1 - args.rewired_ratio_X) + VA[:, maxoverlap_index]
                            * args.rewired_ratio_X * torch.sign(maxoverlap))

        SI = torch.zeros(VX.shape[0], U.shape[0]).to(device)
        SI[range(min(U.shape[0], VX.shape[0])), range(min(U.shape[0], VX.shape[0]))] = s
        if args.use_right_eigvec:
            new_X = VX_new @ SI @ U
        else:
            new_X = VX_new @ SI

        # Sparsify the node features X by thresholding or top-k if needed
        if args.use_node_attr:
            if iteration == args.denoise_iterations - 1:
                if args.denoise_X_k > 0:
                    flip_X = get_top_k_features(new_X, k=args.denoise_X_k)
                else:
                    flip_X = get_clipped_features(new_X, eps=args.denoise_X_eps)
            else:
                flip_X = new_X

        # Otherwise, flip a certain amount of the node features X if needed or just keep the new features
        else:
            if iteration == args.denoise_iterations - 1:
                if non_binary:
                    flip_X = (1 - args.rewired_ratio_X_non_binary) * X + args.rewired_ratio_X_non_binary * new_X
                elif X[X == 1].sum() == X[X > 0].sum():
                    non_binary = False
                    flip_X = copy.deepcopy(X).to(device)
                    if args.flip_number_X_1 > 0:
                        mask1 = (X == 1).to(device)
                        topk = find_largest((X - new_X) * mask1, args.flip_number_X_1)
                        flip_X[mask1 & ((X - new_X) >= topk)] = 0
                    if args.flip_number_X_0 > 0:
                        mask0 = (X == 0).to(device)
                        topk = find_largest((new_X - X) * mask0, args.flip_number_X_0)
                        flip_X[mask0 & ((new_X - X) >= topk)] = 1
                else:
                    non_binary = True
                    print(f"Using non-binary denoising for features with rate {args.rewired_ratio_X_non_binary}")
                    flip_X = (1-args.rewired_ratio_X_non_binary) * X + args.rewired_ratio_X_non_binary * new_X
            else:
                flip_X = new_X

        # Use exp(X@X.T) as the feature matrix for denoising A
        if args.kernel_X:
            from sklearn import metrics
            X_kernel = metrics.pairwise.pairwise_kernels(X_denoised.cpu(), Y=None, metric='rbf')
            X_kernel = torch.tensor(X_kernel).to(device)
            _, VX = torch.linalg.eigh(X_kernel)
        else:
            VX, s, U = torch.linalg.svd(X_denoised)

        # Denoise the adjacency matrix A
        VA_new = copy.deepcopy(VA).to(device)
        for i in range(N - args.rewired_index_A-offset, N-offset):
            va = copy.deepcopy(VA[:, i]).to(device)
            vx = VX[:, :args.rewired_index_A]
            overlap = torch.matmul(va, vx)
            maxoverlap_index = torch.argmax(torch.abs(overlap))
            maxoverlap = overlap[maxoverlap_index]
            VA_new[:, i] = (va * (1 - args.rewired_ratio_A) + VX[:, maxoverlap_index]
                            * args.rewired_ratio_A * torch.sign(maxoverlap))
        # Order the eigenvalues if needed
        if args.abs_ordering:
            la_abs_sort = la_abs_sort[resort_idx]
            VA_new = VA_new[:, resort_idx]
        else:
            pass
        new_A = VA_new @ torch.diag(la_abs_sort) @ VA_new.T

        # Sparsify by threshold or top-k the adjacency matrix A if needed and build a weighted A
        if args.use_edge_attr:
            if iteration == args.denoise_iterations - 1:
                if args.denoise_A_k > 0:
                    flip_A = get_top_k_matrix(new_A, k=args.denoise_A_k)
                elif args.denoise_A_eps > 0:
                    flip_A = get_clipped_matrix(new_A, eps=args.denoise_A_eps)
            else:
                flip_A = new_A

        # Otherwise filp a certain amount of A to stay binary
        else:
            if iteration == args.denoise_iterations - 1:
                flip_A = copy.deepcopy(A).to(device)
                if args.flip_number_A_1 > 0:
                    mask1 = (A == 1).to(device)
                    topk = find_largest((A - new_A) * mask1, args.flip_number_A_1)
                    flip_A[mask1 & ((A - new_A) >= topk)] = 0

                if args.flip_number_A_0 > 0:
                    mask0 = (A == 0).to(device)
                    topk = find_largest((new_A - A) * mask0, args.flip_number_A_0)
                    flip_A[mask0 & ((new_A - A) >= topk)] = 1
            else:
                flip_A = new_A

        # Update the node features and adjacency matrix if flags are True
        if args.denoise_A:
            A_denoised = copy.deepcopy(flip_A)
        if args.denoise_x:
            X_denoised = copy.deepcopy(flip_X)

    # Otherwise just keep the original data
    if not args.denoise_A:
        A_denoised = copy.deepcopy(A)
    if not args.denoise_x:
        X_denoised = copy.deepcopy(X)

    # Create a new data object with the denoised features and adjacency matrix
    if args.use_edge_attr:
        new_data = Data(x=X_denoised, edge_index=torch_geometric.utils.dense_to_sparse(A_denoised)[0],
                        edge_attr=torch_geometric.utils.dense_to_sparse(A_denoised)[1], y=data.y).to(device)
    else:
        new_data = Data(x=X_denoised, edge_index=torch_geometric.utils.dense_to_sparse(A_denoised)[0],
                        y=data.y).to(device)

    # Log or print the results
    if args.wandb_log:
        import wandb
        if args.use_edge_attr:
            A_denoised = torch_geometric.utils.to_dense_adj(edge_index=new_data.edge_index).squeeze()
        if args.use_node_attr:
            wandb.run.summary["x_value/original"] = abs(X).sum().item()
            wandb.run.summary["x_value/denoised"] = abs(X_denoised).sum().item()
            wandb.run.summary["x_value/change"] = abs(X - X_denoised).sum().item()
        wandb.run.summary["edges/original"] = (A == 1).sum().item()
        wandb.run.summary["edges/denoised"] = (A_denoised == 1).sum().item()
        wandb.run.summary["edges/add"] = ((A-A_denoised) == -1).sum().item()
        wandb.run.summary["edges/remove"] = ((A-A_denoised) == 1).sum().item()
        wandb.run.summary["x_entries/original"] = (X == 1).sum().item()
        wandb.run.summary["x_entries/denoised"] = (X_denoised == 1).sum().item()
        wandb.run.summary["x_entries/add"] = ((X-X_denoised) == -1).sum().item()
        wandb.run.summary["x_entries/remove"] = ((X-X_denoised) == 1).sum().item()
        if non_binary:
            wandb.run.summary["x_value/original"] = abs(X).sum().item()
            wandb.run.summary["x_value/denoised"] = abs(X_denoised).sum().item()
            wandb.run.summary["x_value/change"] = abs(X - X_denoised).sum().item()
        wandb.run.summary["align"] = compute_alignment(X, A, args)
        wandb.run.summary["align_denoised"] = compute_alignment(X_denoised, A_denoised, args)

    else:
        if args.use_edge_attr:
            A_denoised = torch_geometric.utils.to_dense_adj(edge_index=new_data.edge_index).squeeze()
        if args.use_node_attr:
            print("Value of x_entries in original X: ", abs(X).sum().item())
            print("Value of x_entries in denoised X: ", abs(X_denoised).sum().item())
            print("Value of x_entries change: ", abs(X - X_denoised).sum().item())
        print("Number of edges in original A: ", (A == 1).sum().item())
        print("Number of edges denoised A: ", (A_denoised == 1).sum().item())
        print("Number of edges_added: ", ((A - A_denoised) == -1).sum().item())
        print("Number of edges_removed: ", ((A - A_denoised) == 1).sum().item())
        print("Number of x_entries in original X: ", (X == 1).sum().item())
        print("Number of x_entries in denoised X: ", (X_denoised == 1).sum().item())
        print("Number of x_entries_added: ", ((X - X_denoised) == -1).sum().item())
        print("Number of x_entries_removed: ", ((X - X_denoised) == 1).sum().item())
        if non_binary:
            print("Value of x_entries in original X: ", abs(X).sum().item())
            print("Value of x_entries in denoised X: ", abs(X_denoised).sum().item())
            print("Value of x_entries change: ", abs(X - X_denoised).sum().item())
        print("Alignment of X and A: ", compute_alignment(X, A, args))
        print("Alignment of X_denoised and A_denoised: ", compute_alignment(X_denoised, A_denoised, args))
    return new_data


def denoise_jointly_large(data, args, device):
    """
    Denoises a large graph jointly using the given arguments and truncated SVD/eigsh
    Args:
        data: PyG data object
        args: arguments from argparse
        device: device to run the operations on (cuda or cpu)

    Returns: denoised PyG data object
    """
    import scipy.sparse.linalg as sp_linalg
    offset = args.denoise_offset
    X = data.x
    A = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index)[0]
    non_binary = args.denoise_non_binary

    # Main loop for denoising
    for iteration in tqdm(range(args.denoise_iterations), desc="Denoising"):
        # Decomposition
        # VX, s, U = sp_linalg.svds(X.numpy(), k=args.rewired_index_X)
        VX, s, U = randomized_sparse_svd(X.numpy(), k=args.rewired_index_x)  # Adding sparse svd approximation
        VX = torch.tensor(VX.copy())
        s = torch.tensor(s.copy())
        U = torch.tensor(U.copy())
        sort_idx_X = torch.argsort(s, descending=True)
        VX = VX[:, sort_idx_X]
        s = s[sort_idx_X]
        U = U[sort_idx_X, :]
        if args.abs_ordering:
            la, VA = sp_linalg.eigsh(A.numpy(), k=(args.rewired_index_A+offset))
            la = torch.tensor(la.copy())
            VA = torch.tensor(VA.copy())
        else:
            la, VA = sp_linalg.eigsh(A.numpy(), k=2*(args.rewired_index_A+offset))
            la = torch.tensor(la.copy())
            VA = torch.tensor(VA.copy())
            sort_idx_A = torch.argsort(la, descending=True)
            la = la[sort_idx_A]
            VA = VA[:, sort_idx_A]
            la = la[:args.rewired_index_A]
            VA = VA[:, :len(la)]
        N = A.shape[0]

        # Denoise the node features X first
        VX_new = copy.deepcopy(VX).to(device)
        for i in range(args.rewired_index_X):
            vx = copy.deepcopy(VX[:, i]).to(device)
            va = VA[:, offset:args.rewired_index_A+offset].to(device)
            overlap = torch.matmul(vx, va)
            maxoverlap_index = torch.argmax(torch.abs(overlap))-offset
            maxoverlap = overlap[maxoverlap_index]
            VX_new[:, i] = (vx * (1 - args.rewired_ratio_X) + VA[:, maxoverlap_index].to(device)
                            * args.rewired_ratio_X * torch.sign(maxoverlap))

        SI = torch.zeros(args.rewired_index_X, args.rewired_index_X)
        SI[range(min(U.shape[0], VX.shape[0])), range(min(U.shape[0], VX.shape[0]))] = s
        new_X = X - VX @ SI @ U + VX_new.cpu() @ SI @ U

        # Sparsify the node features X by thresholding or top-k if needed
        if args.use_node_attr:
            if iteration == args.denoise_iterations - 1:
                if args.denoise_X_k > 0:
                    flip_X = get_top_k_features(new_X, k=args.denoise_X_k)
                else:
                    flip_X = get_clipped_features(new_X, eps=args.denoise_X_eps)
            else:
                flip_X = new_X

        # Otherwise, flip a certain amount of the node features X if needed or just keep the new features
        else:
            if iteration == args.denoise_iterations - 1:
                if non_binary:
                    flip_X = (1 - args.rewired_ratio_X_non_binary) * X + args.rewired_ratio_X_non_binary * new_X
                elif X[X == 1].sum() == X[X > 0].sum():
                    non_binary = False
                    flip_X = copy.deepcopy(X).to(device)
                    if args.flip_number_X_1 > 0:
                        mask1 = (X == 1).to(device)
                        topk = find_largest((X - new_X) * mask1, args.flip_number_X_1)
                        flip_X[mask1 & ((X - new_X) >= topk)] = 0
                    if args.flip_number_X_0 > 0:
                        mask0 = (X == 0).to(device)
                        topk = find_largest((new_X - X) * mask0, args.flip_number_X_0)
                        flip_X[mask0 & ((new_X - X) >= topk)] = 1
                else:
                    non_binary = True
                    print(f"Using non-binary denoising for features with rate {args.rewired_ratio_X_non_binary}")
                    flip_X = (1-args.rewired_ratio_X_non_binary) * X + args.rewired_ratio_X_non_binary * new_X
            else:
                flip_X = new_X

        if args.denoise_x:
            X = copy.deepcopy(flip_X)
            del flip_X, new_X
        # Denoise the adjacency matrix A
        VA_new = copy.deepcopy(VA).to(device)
        for i in range(offset, args.rewired_index_A+offset):
            va = copy.deepcopy(VA[:, i]).to(device)
            vx = VX[:, :args.rewired_index_A].to(device)
            overlap = torch.matmul(va, vx)
            maxoverlap_index = torch.argmax(torch.abs(overlap))
            maxoverlap = overlap[maxoverlap_index]
            VA_new[:, i] = (va * (1 - args.rewired_ratio_A) + VX[:, maxoverlap_index].to(device)
                            * args.rewired_ratio_A * torch.sign(maxoverlap))
        new_A = A + VA_new.cpu() @ torch.diag(la) @ VA_new.cpu().T - VA @ torch.diag(la) @ VA.T

        # Sparsify by threshold or top-k the adjacency matrix A if needed and build a weighted A
        if args.use_edge_attr:
            if iteration == args.denoise_iterations - 1:
                if args.denoise_A_k > 0:
                    flip_A = get_top_k_matrix(new_A, k=args.denoise_A_k)
                elif args.denoise_A_eps > 0:
                    flip_A = get_clipped_matrix(new_A, eps=args.denoise_A_eps)
            else:
                flip_A = new_A

        # Otherwise filp a certain amount of A to stay binary
        else:
            if iteration == args.denoise_iterations - 1:
                flip_A = copy.deepcopy(A).to(device)
                if args.flip_number_A_1 > 0:
                    mask1 = (A == 1).to(device)
                    topk = find_largest((A - new_A) * mask1, args.flip_number_A_1)
                    flip_A[mask1 & ((A - new_A) >= topk)] = 0

                if args.flip_number_A_0 > 0:
                    mask0 = (A == 0).to(device)
                    topk = find_largest((new_A - A) * mask0, args.flip_number_A_0)
                    flip_A[mask0 & ((new_A - A) >= topk)] = 1
            else:
                flip_A = new_A

        # Update the node features and adjacency matrix if flags are True
        if args.denoise_A:
            A = copy.deepcopy(flip_A)
        del VX, s, U, la, VA, VX_new, VA_new, new_A, flip_A, va, vx

    # Otherwise just keep the original data
    if not args.denoise_A:
        A = copy.deepcopy(A)
    if not args.denoise_x:
        X = copy.deepcopy(X)

    # Create a new data object with the denoised features and adjacency matrix
    if args.use_edge_attr:
        new_data = Data(x=X, edge_index=torch_geometric.utils.dense_to_sparse(A)[0],
                        edge_attr=torch_geometric.utils.dense_to_sparse(A)[1], y=data.y, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask).to(device)
    else:
        new_data = Data(x=X, edge_index=torch_geometric.utils.dense_to_sparse(A)[0],
                        y=data.y, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask).to(device)

    # Log or print the results
    if args.wandb_log:
        import wandb
        wandb.run.summary["edges/denoised"] = (A != 0).sum().item()
    else:
        print("Number of edges denoised A: ", (A != 0).sum().item())
    return new_data


def denoise_jointly_sparse(dataset, args, device):
    """
    Denoises a very large graph jointly using sparse methods with the given arguments
    Utilizes the block power method for svd computation, and matrix-free Lanczos for eigenvalue computation
    Args:
        dataset: PyG data
        args: arguments from argparse
        device: device to run the operations on (cuda or cpu)

    Returns: denoised PyG data object
    """

    # Defining data
    data = dataset[0]
    X = data.x.to(device)
    N = data.num_nodes
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_attr.to(device) if (hasattr(data, "edge_attr") and data.edge_attr is not None) else torch.ones(edge_index.size(1), device=device)
    
    # Build a local arguments object with defaults so we reference a single local object
    local_args = SimpleNamespace(
        rewired_index_X=getattr(args, "rewired_index_X", 12),
        rewired_index_A=getattr(args, "rewired_index_A", 12),
        denoise_offset=getattr(args, "denoise_offset", 0),
        denoise_power_iter=getattr(args, "denoise_power_iter", 8),
        prune_keep_fraction=getattr(args, "prune_keep_fraction", 0.97),
        alpha_feat=getattr(args, "alpha_feat", 0.4),
        beta_struct=getattr(args, "beta_struct", 0.6),
        abs_ordering=getattr(args, "abs_ordering", False),
        rewired_ratio_A=getattr(args, "rewired_ratio_A", 0.03),
        rewired_ratio_X=getattr(args, "rewired_ratio_X", 0.03),
        use_node_attr=getattr(args, "use_node_attr", True),
        use_edge_attr=getattr(args, "use_edge_attr", True),
        denoise_iterations=getattr(args, "denoise_iterations", 1),
        denoise_X_k=getattr(args, "denoise_X_k", 0),
        denoise_X_eps=getattr(args, "denoise_X_eps", 1e-6),
        denoise_X=getattr(args, "denoise_X", True),
        rewired_ratio_X_non_binary=getattr(args, "rewired_ratio_X_non_binary", None),
        wandb_log=getattr(args, "wandb_log", True),
        edge_update_batch=getattr(args, "edge_update_batch", 1_000_000),
    )

    # Main loop for denoising
    for iteration in tqdm(range(local_args.denoise_iterations), desc="Denoising"):
        # Decomposition
        Ux, Sx, Vx = randomized_sparse_svd(X, k=local_args.rewired_index_X)  # Adding sparse svd approximation
        la, VA = topk_eigen_sparse(edge_index, edge_weight, N, k=(local_args.rewired_index_A + local_args.denoise_offset), iter=local_args.denoise_power_iter)

        if not local_args.abs_ordering:
            abs_la, idxs = torch.sort(torch.abs(la), descending=True)
            idxs = idxs[:(local_args.rewired_index_A + local_args.denoise_offset)]
            la = la[idxs]
            VA = VA[:, idxs]

        VA_for_align = VA[:, local_args.denoise_offset: local_args.denoise_offset + local_args.rewired_index_A]
        la_for_align = la[local_args.denoise_offset: local_args.denoise_offset + local_args.rewired_index_A] if la.numel() >= local_args.denoise_offset + local_args.rewired_index_A else la

        VA_for_align = VA_for_align.to(device)
        Ux = Ux.to(device)

        # preserve device and avoid creating a new tensor via torch.tensor (which may copy to CPU)
        Ux_new = Ux.clone().detach().to(device)

        overlap = (Ux.T @ VA_for_align)

        for i in range(min(local_args.rewired_index_X, overlap.shape[0])):
            row = overlap[i]
            maxj = torch.argmax(torch.abs(row))
            sign = torch.sign(row[maxj]).to(device)

            Ux_new[:, i] = (1.0 - local_args.rewired_ratio_X) * Ux[:, i] + local_args.rewired_ratio_X * sign * VA_for_align[:, maxj]

        # Reconstructing new features

        S_diag = Sx.to(device)
        # Chunking to reduce peak memory usage
        new_X = torch.empty_like(X)

        # Check for shape
        assert S_diag.numel() == Ux.shape[1]

        chunk_rows = 2000
        for i in range(0, N, chunk_rows):
            i2 = min(i + chunk_rows, N)
            chunk_diff = (Ux_new[i:i2] - Ux[i:i2]) * S_diag
            delta_chunk = chunk_diff @ Vx.T.to(device)
            new_X[i:i2] = X[i:i2] + delta_chunk

        if local_args.use_node_attr:
            if iteration == local_args.denoise_iterations - 1:
                if local_args.denoise_X_k > 0:
                    # keep only top-k features per node (simple top-k)
                    k = local_args.denoise_X_k
                    vals, idxs = torch.topk(new_X.abs(), k=k, dim=1)
                    mask = torch.zeros_like(new_X, dtype=torch.bool)
                    mask.scatter_(1, idxs, True)
                    flip_X = new_X.clone()
                    flip_X[~mask] = 0.0
                else:
                    # clip small values
                    eps = local_args.denoise_X_eps
                    flip_X = new_X.clone()
                    flip_X[flip_X.abs() < eps] = 0.0
            else:
                flip_X = new_X
        else:
            # simple non-binary mixing
            if iteration == local_args.denoise_iterations - 1 and local_args.rewired_ratio_X_non_binary is not None:
                rrate = local_args.rewired_ratio_X_non_binary
                flip_X = (1 - rrate) * X + rrate * new_X
            else:
                flip_X = new_X

        # update X if requested
        if local_args.denoise_X:
            X = flip_X.clone()

        VA_new = VA.clone().to(device)    # (N, rA+offset)
        VA_small = VA_for_align  # (N, rA)
        # compute overlap: (rA, rX) = VA_small.T @ Ux
        ov2 = (VA_small.T @ Ux)   # (rA, rX)
        for j in range(min(VA_small.shape[1], ov2.shape[0])):
            col = ov2[j]   # (rX,)
            maxi = torch.argmax(torch.abs(col))
            sign = torch.sign(col[maxi]).to(device)
            VA_new[:, local_args.denoise_offset + j] = (1.0 - local_args.rewired_ratio_A) * VA[:, local_args.denoise_offset + j].to(device) + \
                                    local_args.rewired_ratio_A * sign * Ux[:, maxi]

        # ---- 6) Compute per-edge spectral update (sparse) ----
        # Use the same eigenvector columns for per-edge projection that were used
        # to build `la_for_align`/`VA_for_align` earlier. This avoids mismatched
        # projections when an offset is used.
        rA = la_for_align.numel()
        offset = local_args.denoise_offset
        start_col = offset
        end_col = offset + rA

        # bring VA into device space and slice to the exact columns used by la_for_align
        VA_old = VA.to(device)
        VA_new_full = VA_new.to(device)
        VA_old_small = VA_old[:, start_col:end_col]
        VA_new_small = VA_new_full[:, start_col:end_col]

        # Ensure lam matches the number of columns
        lam_dev = la_for_align.to(VA_old_small.device).to(VA_old_small.dtype)
        if VA_old_small.shape[1] != lam_dev.numel():
            raise RuntimeError(f"VA columns ({VA_old_small.shape[1]}) != lam length ({lam_dev.numel()})")

        # compute per-edge delta (E,) in chunks to avoid materializing huge (E,k) tensors
        batch_size = local_args.edge_update_batch
        E_total = edge_index.size(1)
        edge_delta = torch.empty((E_total,), dtype=edge_weight.dtype, device=edge_weight.device)
        src_all, dst_all = edge_index

        for start in range(0, E_total, batch_size):
            end = min(start + batch_size, E_total)
            idx = slice(start, end)
            src_idx = src_all[idx]
            dst_idx = dst_all[idx]

            src_old = VA_old_small[src_idx]
            dst_old = VA_old_small[dst_idx]
            src_new = VA_new_small[src_idx]
            dst_new = VA_new_small[dst_idx]

            diff = (src_new * dst_new) - (src_old * dst_old)  # (batch, rA)
            # compute chunk-wise delta and store
            edge_delta[idx] = (diff @ lam_dev).to(edge_weight.device)

        edge_weight = edge_weight + edge_delta

        # ---- 7) Score edges for pruning (simple combination of feature & structural similarity) ----
        # feature similarity
        src, dst = edge_index
        # Changed from using Ux to Ux_new for the updated computation
        feat_sim = ((Ux_new[src] * Ux_new[dst]).sum(dim=1))   # (E,)
        struct_sim = ((VA_new_small[src] * VA_new_small[dst]).sum(dim=1))  # (E,)
        score = local_args.alpha_feat * feat_sim + local_args.beta_struct * struct_sim
        # prune edges by global keep fraction, with minimal re-add behavior
        edge_index, edge_weight = prune_edges(edge_index, edge_weight, score, N, keep=local_args.prune_keep_fraction, min_degree=1)

        # coalesce duplicates (if any) using torch_sparse.coalesce if available
        try:
            from torch_sparse import coalesce
            edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
        except Exception:
            # simple fallback: nothing (assume no duplicates)
            pass

        # end iteration — optionally log stats
        if local_args.wandb_log:
            import wandb
            wandb.init(project="Denoising Products", config=local_args)
            wandb.log({"denoise/edges": edge_weight.nonzero().size(0)})
        else:
            if iteration % max(1, (local_args.denoise_iterations // 5)) == 0:
                print(f"Iteration {iteration}: edges {edge_weight.size(0)}")

    # If user expects binary adjacency, threshold weights back to 0/1
    if not local_args.use_edge_attr:
        # threshold at zero
        mask_keep = edge_weight > 0
        edge_index = edge_index[:, mask_keep]
        edge_weight = None
    else:
        # keep weights and attach as edge_attr
        edge_weight = edge_weight

    # final feature X: keep on cpu or device depending on original data
    X_out = X.cpu()

    # Defining train, validation, and test masks for ogbn-products
    split =  dataset.get_idx_split()
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    # Assigning masks based on PyG splits
    train_mask[split["train"]] = True
    val_mask[split["valid"]] = True
    test_mask[split["test"]] = True

    if edge_weight is None:
        new_data = Data(x=X_out, edge_index=edge_index.cpu(), y=data.y,
                        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    else:
        new_data = Data(x=X_out, edge_index=edge_index.cpu(), edge_attr=edge_weight.cpu(),
                        y=data.y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return new_data