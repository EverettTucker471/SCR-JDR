import numpy as np
import torch
import torch.nn as nn
from ogb.nodeproppred import NodePropPredDataset, Evaluator
import torch.nn.functional as F
import gc
from cogdl.data import Graph
from cogdl.utils import to_undirected, spmm_cpu


def prepare_label_emb(args, graph, labels, n_classes, train_node_nums, label_teacher_emb=None):
    if label_teacher_emb == None:
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[:train_node_nums] = F.one_hot(labels[:train_node_nums].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.FloatTensor(y)
    else:
        print("use teacher label")
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[train_node_nums:] = label_teacher_emb[train_node_nums:]
        y[:train_node_nums] = F.one_hot(labels[:train_node_nums].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.FloatTensor(y)
    graph.row_norm()
    for hop in range(args.label_num_hops):
        y = spmm_cpu(graph, y)
    return y


def get_ogb_evaluator(dataset):
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]


def load_dataset(name, device, args, return_nid=False):
    if name not in ["ogbn-products", "ogbn-papers100M"]:
        raise RuntimeError("Dataset {} is not supported".format(name))

    dataset = NodePropPredDataset(name=name, root=args.root)
    splitted_idx = dataset.get_idx_split()

    # Always load correct labels from OGB (not from denoised graph)
    graph_raw, y_raw = dataset[0]
    y = torch.LongTensor(y_raw.squeeze()).contiguous()

    # If a denoised graph file is provided, use its adjacency + features
    if hasattr(args, "denoised_graph_path") and args.denoised_graph_path is not None:
        print(f"Using denoised graph from {args.denoised_graph_path}")

        data = torch.load(args.denoised_graph_path, weights_only=False)

        # adjacency + features come from denoiser
        x = data.x.float().contiguous()
        edge_index = data.edge_index.long().contiguous()

        # build graph using denoised structure + correct labels
        graph = Graph(x=x, edge_index=edge_index, y=y)

    else:
        # ORIGINAL BEHAVIOR when no denoised graph is used
        x = torch.tensor(graph_raw["node_feat"]).float().contiguous() if graph_raw["node_feat"] is not None else None

        row, col = graph_raw["edge_index"][0], graph_raw["edge_index"][1]
        row = torch.from_numpy(row)
        col = torch.from_numpy(col)
        edge_index = torch.stack([row, col], dim=0)

        graph = Graph(x=x, edge_index=edge_index, y=y)

    # attach splits to graph
    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]

    # stats display remains unchanged
    print(f"# Nodes: {graph.num_nodes}\n"
          f"# Edges: {graph.num_edges}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {graph.num_classes}\n")

    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    evaluator = get_ogb_evaluator(name)

    if not return_nid:
        return graph, train_node_nums, valid_node_nums, test_node_nums, evaluator
    # return_nid=True → return node ID tensors instead of counts
    train_nid = torch.LongTensor(train_nid)
    val_nid = torch.LongTensor(val_nid)
    test_nid = torch.LongTensor(test_nid)

    return graph, train_nid, val_nid, test_nid, evaluator


def prepare_data(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """

    data = load_dataset(args.dataset, device, args)

    graph, train_node_nums, valid_node_nums, test_node_nums, evaluator = data
    if args.dataset == 'ogbn-papers100M':
        graph.edge_index = to_undirected(graph.edge_index)

    # move to device
    feats=[]
    for i in range(args.num_hops+1):
        if args.giant:
            print(f"load feat_{i}_giant.pt")
            feats.append(torch.load(f"./{args.dataset}/feat/{args.dataset}_feat_{i}_giant.pt"))
        else:
            print(f"load feat_{i}.pt")
            feats.append(torch.load(f"./{args.dataset}/feat/{args.dataset}_feat_{i}.pt"))
    in_feats=feats[0].shape[1]

    if args.dataset == 'ogbn-products':
        return graph, feats, graph.y, in_feats, graph.num_classes, \
               train_node_nums, valid_node_nums, test_node_nums, evaluator

