import torch
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from types import SimpleNamespace

from denoise_jointly import denoise_jointly_sparse


def load_pyg_graph():
    dataset = NodePropPredDataset(name="ogbn-products", root=".")
    graph, y = dataset[0]
    x = torch.tensor(graph["node_feat"]).float()
    row, col = graph["edge_index"]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    y = torch.tensor(y.squeeze(), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    return data


def compute_stats(data):
    ei = data.edge_index
    deg = torch.bincount(ei[0], minlength=data.num_nodes)
    mean_deg = deg.float().mean().item()
    frac_iso = (deg == 0).float().mean().item()

    src, dst = ei
    hom = (data.y[src] == data.y[dst]).float().mean().item()

    return mean_deg, frac_iso, hom


if __name__ == "__main__":
    device = "cpu"

    data = load_pyg_graph().to(device)

    # BEFORE
    md_b, iso_b, hom_b = compute_stats(data)
    print("Mean Degree Before:", md_b)
    print("Fraction Isolated Before:", iso_b)
    print("Global Homophily Before:", hom_b)

    # PARAMETERS similar to teammate's run
    args = SimpleNamespace(
        rewired_index_X=4,
        rewired_index_A=4,
        denoise_offset=0,
        denoise_power_iter=3,
        prune_keep_fraction=0.95,  # means remove 6% of edges
        alpha_feat=0.6,
        beta_struct=0.4,
        abs_ordering=False,
        rewired_ratio_X=0.05,
        use_node_attr=True,
        denoise_iterations=1,
        denoise_X=True,
        denoise_X_k=0,
        denoise_X_eps=1e-4,
        wandb_log=False,
    )

    new_data = denoise_jointly_sparse(data, args, device)

    # AFTER
    md_a, iso_a, hom_a = compute_stats(new_data)
    print("Mean Degree After:", md_a)
    print("Fraction Isolated After:", iso_a)
    print("Global Homophily After:", hom_a)

    # Save adjacency
    torch.save(new_data.edge_index.cpu(), "denoised_edge_index.pt")
    torch.save(new_data.x.cpu(), "denoised_x.pt")

    print("Saved denoised_edge_index.pt and denoised_x.pt")
