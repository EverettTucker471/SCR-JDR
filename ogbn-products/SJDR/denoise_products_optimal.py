import os
import json
import torch
import time
from types import SimpleNamespace
from ogb.nodeproppred import PygNodePropPredDataset
from denoise_jointly import denoise_jointly_sparse


def main():
    # load dataset (same root as original script)
    root = "../data/products"
    dataset = PygNodePropPredDataset(name="ogbn-products", root=root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters from run #7 (0-indexed) in the original `denoise_products.py`
    run7_args = {
        'rewired_index_X': 12,
        'rewired_index_A': 12,
        'prune_keep_fraction': 0.995,
        'alpha_feat': 0.8,
        'beta_struct': 0.2,
        'denoise_iterations': 1,
        'denoise_X_eps': 1e-6,
        'wandb_log': True,
    }

    args_ns = SimpleNamespace(**run7_args)

    print("Starting denoising with run #7 hyperparameters:", run7_args)
    denoised_data = denoise_jointly_sparse(dataset, args_ns, device)

    # prepare output directory
    out_dir = os.path.join(os.getcwd(), 'denoised_runs_run7')
    os.makedirs(out_dir, exist_ok=True)

    # Save the denoised data object
    timestamp = int(time.time())  # To nearest second is good enough for me
    # Saving data with a unique timestamp for uniqueness
    model_path = os.path.join(out_dir, f'denoised_data_optimal{timestamp}.pt')
    torch.save(denoised_data, model_path)
    print(f"Saved denoised data to {model_path}")

    # Compute and save a small metrics summary
    N = denoised_data.num_nodes
    deg_after = torch.bincount(denoised_data.edge_index.view(-1), minlength=N)
    src_after, dst_after = denoised_data.edge_index
    same_after = (denoised_data.y[src_after].view(-1) == denoised_data.y[dst_after].view(-1)).float()
    global_hom_after = float(same_after.mean().item())

    # load original for before metrics
    data_orig = dataset[0]
    deg_before = torch.bincount(data_orig.edge_index.view(-1), minlength=N)
    src, dst = data_orig.edge_index
    same = (data_orig.y[src].view(-1) == data_orig.y[dst].view(-1)).float()
    global_hom_before = float(same.mean().item())

    metrics = {
        'mean_degree_before': float(deg_before.float().mean().item()),
        'mean_degree_after': float(deg_after.float().mean().item()),
        'fraction_isolated_before': float((deg_before == 0).float().mean().item()),
        'fraction_isolated_after': float((deg_after == 0).float().mean().item()),
        'global_hom_before': global_hom_before,
        'global_hom_after': global_hom_after,
        'args': run7_args,
    }

    with open(os.path.join(out_dir, 'metrics_run7.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Saved metrics to {os.path.join(out_dir, 'metrics_run7.json')}")


if __name__ == '__main__':
    main()
