import torch
import torch_geometric
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch_scatter import scatter_add
from ogb.nodeproppred import PygNodePropPredDataset
from denoise_jointly import denoise_jointly_sparse
from types import SimpleNamespace
import os
import json

# Importing dataset
root = "../data/products"
dataset = PygNodePropPredDataset(name="ogbn-products", root=root)
data = dataset[0]

# Defining the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare experiment configurations (list of dicts). If empty, a single default run will be executed.
# Example usage: uncomment and edit arg_dicts to run multiple configs overnight.
# arg_dicts = [
#     {'rewired_index_X': 8, 'rewired_index_A': 8, 'prune_keep_fraction': 0.98, 'alpha_feat': 0.8, 'beta_struct': 0.2},
#     {'rewired_index_X': 16, 'rewired_index_A': 16, 'prune_keep_fraction': 0.95, 'alpha_feat': 0.7, 'beta_struct': 0.3},
# ]
arg_dicts = [
    {'rewired_index_X': 8, 'rewired_index_A': 8, 'prune_keep_fraction': 0.99, 'alpha_feat': 0.8, 'beta_struct': 0.2, 'denoise_X_eps': 1e-6},
]

# Default args used when arg_dicts is empty
default_args = {}

# output directory for denoised results
out_dir = os.path.join(os.getcwd(), 'denoised_runs')
os.makedirs(out_dir, exist_ok=True)


# Precompute the "before" metrics once
N = data.num_nodes
deg_before = torch.bincount(data.edge_index.view(-1), minlength=N)
src, dst = data.edge_index
same = (data.y[src].view(-1) == data.y[dst].view(-1)).float()
global_hom_before = same.mean().item()
y = data.y.view(-1)
num_classes = int(y.max().item() + 1)
per_class_h_before = torch.zeros(num_classes)
for c in range(num_classes):
    # mask edges incident to class c (use node labels, not node indices)
    mask = (y[src] == c) | (y[dst] == c)
    if mask.sum() == 0:
        per_class_h_before[c] = float('nan')
    else:
        per_class_h_before[c] = same[mask].float().mean().item()
pairs = y[src] * num_classes + y[dst]
hist_before = torch.bincount(pairs, minlength=num_classes * num_classes).view(num_classes, num_classes)


def run_and_evaluate(arg_dict, run_idx=0):
    # build args namespace
    args_ns = SimpleNamespace(**arg_dict) if arg_dict is not None else SimpleNamespace(**default_args)
    print(f"Running denoising run {run_idx} with args: {arg_dict}")
    denoised_data = denoise_jointly_sparse(dataset, args_ns, device)

    # post-run metrics
    deg_after = torch.bincount(denoised_data.edge_index.view(-1), minlength=N)
    src_after, dst_after = denoised_data.edge_index
    same_after = (denoised_data.y[src_after].view(-1) == denoised_data.y[dst_after].view(-1)).float()
    global_hom_after = same_after.mean().item()

    per_class_h_after = torch.zeros(num_classes)
    for c in range(num_classes):
        # mask edges incident to class c (use node labels)
        mask = (y[src_after] == c) | (y[dst_after] == c)
        if mask.sum() == 0:
            per_class_h_after[c] = float('nan')
        else:
            per_class_h_after[c] = same_after[mask].float().mean().item()

    pairs_after = y[src_after] * num_classes + y[dst_after]
    hist_after = torch.bincount(pairs_after, minlength=num_classes * num_classes).view(num_classes, num_classes)

    # Save denoised data and metrics
    run_name = f"run_{run_idx}"
    out_path = os.path.join(out_dir, run_name)
    os.makedirs(out_path, exist_ok=True)
    torch.save(denoised_data, os.path.join(out_path, 'denoised_data.pt'))

    metrics = {
        'mean_degree_before': float(deg_before.float().mean().item()),
        'mean_degree_after': float(deg_after.float().mean().item()),
        'fraction_isolated_before': float((deg_before == 0).float().mean().item()),
        'fraction_isolated_after': float((deg_after == 0).float().mean().item()),
        'global_hom_before': float(global_hom_before),
        'global_hom_after': float(global_hom_after)
    }
    with open(os.path.join(out_path, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    torch.save(per_class_h_after, os.path.join(out_path, 'per_class_h_after.pt'))
    torch.save(hist_after, os.path.join(out_path, 'hist_after.pt'))

    print(f"Run {run_idx} complete. Metrics saved to {out_path}")
    return {
        'metrics': metrics,
        'per_class_h_after': per_class_h_after,
        'hist_after': hist_after,
    }


# Execute runs
if len(arg_dicts) == 0:
    run_and_evaluate(default_args, run_idx=0)
else:
    for i, ad in enumerate(arg_dicts):
        try:
            run_and_evaluate(ad, run_idx=i)
        except Exception as e:
            print(f"Run {i} failed with exception: {e}")

# Build confusion_diff for the last run to keep plotting code consistent
try:
    # load last hist_after if present
    last_hist = torch.load(os.path.join(out_dir, os.listdir(out_dir)[-1], 'hist_after.pt'))
    hist_after = last_hist
    confusion_diff = hist_after - hist_before
    # compute deg_after and global_hom_after for printing (from last run's metrics file)
    with open(os.path.join(out_dir, os.listdir(out_dir)[-1], 'metrics.json')) as fh:
        last_metrics = json.load(fh)
    print(f"Mean Degree Before: {last_metrics['mean_degree_before']}")
    print(f"Mean Degree After: {last_metrics['mean_degree_after']}")
    print(f"Fraction Isolated Before: {last_metrics['fraction_isolated_before']}")
    print(f"Fraction Isolated After: {last_metrics['fraction_isolated_after']}")
    print(f"Global Homophily Before: {last_metrics['global_hom_before']}")
    print(f"Global Homophily After: {last_metrics['global_hom_after']}")
    # also try to load per-class homophily for plotting
    try:
        per_class_h_after = torch.load(os.path.join(out_dir, os.listdir(out_dir)[-1], 'per_class_h_after.pt'))
    except Exception:
        per_class_h_after = per_class_h_before
except Exception:
    # Fallback to previous behavior if saving/loading failed
    hist_after = hist_before
    confusion_diff = hist_before - hist_before
    print(f"Mean Degree Before: {deg_before.float().mean().item()}")
    print(f"Global Homophily Before: {global_hom_before}")
    per_class_h_after = per_class_h_before

# Plotting Edge-label confusion
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
sns.heatmap(hist_before.cpu().numpy(), cmap='viridis')
plt.title('Before Denoising')
plt.xlabel('Dst Label')
plt.ylabel('Src Label')

plt.subplot(1,3,2)
sns.heatmap(hist_after.cpu().numpy(), cmap='viridis')
plt.title('After Denoising')
plt.xlabel('Dst Label')
plt.ylabel('Src Label')

plt.subplot(1,3,3)
sns.heatmap(confusion_diff.cpu().numpy(), cmap='coolwarm', center=0)
plt.title('Difference (After - Before)')
plt.xlabel('Dst Label')
plt.ylabel('Src Label')

plt.tight_layout()
plt.show()

# Plotting per class homophily
classes = np.arange(num_classes)
plt.figure(figsize=(12,5))
plt.bar(classes-0.2, per_class_h_before.cpu().numpy(), width=0.4, label='Before Denoising')
plt.bar(classes+0.2, per_class_h_after.cpu().numpy(), width=0.4, label='After Denoising')
plt.xlabel('Class')
plt.ylabel('Fraction of same-class edges')
plt.title('Per-class Homophily Before and After Denoising')
plt.legend()
plt.show()

