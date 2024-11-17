import argparse

import numpy as np
import torch
from spikegcl.evaluate import test
from spikegcl.dataset import get_dataset
from spikegcl.model import SpikeGCL, SpikeGCL_DASGNN
from spikegcl.utils import tab_printer
from torch_geometric import seed_everything
from torch_geometric.logging import log
from sklearn.cluster import KMeans
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import os.path as osp

def tag2index(dataset):
    tag_set = []
    for g in dataset:
        all_nodes = torch.cat([g.edge_index[0], g.edge_index[1]])
        node_tags = torch.bincount(all_nodes, minlength=g.num_nodes)/2
        node_tags = list(set(list(np.array(node_tags))))
        tag_set += node_tags
    tagset = list(set(tag_set))
    tag2index_dict = {int(tagset[i]):i for i in range(len(tagset))}
    return tag2index_dict

def apply_deg_features(dataset, dataset_name, deg_features=0):
    if deg_features == 1:  # Set node features to the node's degree
        tag2index_dict = tag2index(dataset)
        processed_dataset = []

        for i in range(len(dataset)):
            g = dataset[i]
            all_nodes = torch.cat([g.edge_index[0], g.edge_index[1]])
            node_tags = list(np.array(torch.bincount(all_nodes, minlength=g.num_nodes) / 2))
            features = torch.zeros(g.num_nodes, len(tag2index_dict))
            features[[range(g.num_nodes)], [tag2index_dict[tag] for tag in node_tags]] = 1
            g.x = features
            processed_dataset.append(g)

        dataset = processed_dataset
    
    ### MODIFIED CAUTION
    # TODO: Modifying some of the should modified
    else:
        processed_dataset = []
        for i in range(len(dataset)):
            g = dataset[i]
            features = torch.ones((g.num_nodes, 1))
            g.x = features
            processed_dataset.append(g)

        dataset = processed_dataset

    return dataset

def dataset_selection(root, dataset_name):
    global split_idx
    if dataset_name.lower() == "reddit":
        dataset = Reddit(osp.join(root, 'Reddit'))
        data = dataset[0]
    elif dataset_name.lower() == "flickr":
        dataset = Flickr(osp.join(root, 'Flickr'))
        data = dataset[0]
    elif dataset_name.lower() == "yelp":
        dataset = Yelp(osp.join(root, 'Yelp'))
        data = dataset[0]
    elif dataset_name.lower() == "cora":
        dataset = Planetoid(osp.join(root, 'Cora'), name = "Cora")
        data = dataset[0]
    elif dataset_name.lower() == 'citeseer':
        dataset = Planetoid(osp.join(root, 'Citeseer'), name = "CiteSeer")
        data = dataset[0]
    elif dataset_name.lower() == 'pubmed':
        dataset = Planetoid(osp.join(root, 'PubMed'), name = 'PubMed')
        data = dataset[0]
    elif dataset_name.lower() == 'enzymes':
        dataset = TUDataset(osp.join(root, 'ENZYMES'), name = 'ENZYMES', use_node_attr=True)
        # dataset = apply_deg_features(dataset, dataset_name, 0)
        data = dataset[0]
    elif dataset_name.lower() == 'enzymes_one':
        dataset = TUDataset(osp.join(root, 'ENZYMES'), name = 'ENZYMES', use_node_attr=True)
        dataset = apply_deg_features(dataset, dataset_name, 0)
        data = dataset[0]
    elif dataset_name.lower() == 'mutag':
        dataset = TUDataset(osp.join(root, 'MUTAG'), name = 'MUTAG', use_node_attr=True)
        data = dataset[0]
    elif dataset_name.lower() == 'mutag_one':
        dataset = TUDataset(osp.join(root, 'MUTAG'), name = 'MUTAG', use_node_attr=True)
        dataset = apply_deg_features(dataset, dataset_name, 0)
        data = dataset[0]
    elif dataset_name.lower() == 'proteins':
        dataset = TUDataset(osp.join(root, 'PROTEINS'), name = 'PROTEINS', use_node_attr=True)
        data = dataset[0]
    elif dataset_name.lower() == 'proteins_one':
        dataset = TUDataset(osp.join(root, 'PROTEINS'), name = 'PROTEINS', use_node_attr=True)
        dataset = apply_deg_features(dataset, dataset_name, 0)
        data = dataset[0]
    elif dataset_name == 'COLLAB':
        dataset = TUDataset(osp.join(root, 'COLLAB'), name = 'COLLAB', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'COLLAB', 0)
        data = dataset[0]
    elif dataset_name == 'IMDB-BINARY':
        dataset = TUDataset(osp.join(root, 'IMDB-BINARY'), name = 'IMDB-BINARY', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'IMDB-BINARY', 0)
        data = dataset[0]
    elif dataset_name == 'IMDB-MULTI':
        dataset = TUDataset(osp.join(root, 'IMDB-MULTI'), name = 'IMDB-MULTI', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'IMDB-MULTI', 0)
        data = dataset[0]
    elif dataset_name  == 'REDDIT-BINARY':
        dataset = TUDataset(osp.join(root, 'REDDIT-BINARY'), name = 'REDDIT-BINARY', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'REDDIT-BINARY', 0)
        data = dataset[0]
    elif dataset_name == 'REDDIT-MULTI-5K':
        dataset = TUDataset(osp.join(root, 'REDDIT-MULTI-5K'), name = 'REDDIT-MULTI-5K', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'REDDIT-MULTI-5K', 0)
        data = dataset[0]
    elif dataset_name  == 'PTC_FM':
        dataset = TUDataset(osp.join(root, 'PTC_FM'), name = 'PTC_FM', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'PTC_FM', 0)
        data = dataset[0]
    elif dataset_name == 'NCI1':
        dataset = TUDataset(osp.join(root, 'NCI1'), name = 'NCI1', use_node_attr=True)
        # dataset = apply_deg_features(dataset, 'NCI1', 0)
        data = dataset[0]
    elif dataset_name == 'NCI1_ONE':
        dataset = TUDataset(osp.join(root, 'NCI1'), name = 'NCI1', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'NCI1', 0)
        data = dataset[0]  
    elif dataset_name.lower() == 'imdb-binary-deg':
        dataset = TUDataset(osp.join(root, 'IMDB-BINARY'), name = 'IMDB-BINARY', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'IMDB-BINARY', 1)
        data = dataset[0]
    elif dataset_name.lower() == 'dd':
        dataset = TUDataset(osp.join(root, 'DD'), name = 'DD', use_node_attr=True)
        data = dataset[0]
    else:
        data = None
        dataset = None

    assert type(dataset) is not None , f"Please select dataset correctly"
    return data, dataset
def read_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="data/", help="Data folder"
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="Pubmed",
        help="Datasets (Photo, Computers, CS, Physics, Cora, Citeseer, Pubmed, ogbn-arxiv, ogbn-mag). (default: Pubmed)",
    )
    parser.add_argument(
        "--hids",
        type=int,
        default=64,
        help="Hidden units for each layer. (default: 64)",
    )
    parser.add_argument(
        "--outs",
        type=int,
        default=64,
        help="Out_channels for final embedding. (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for training. (default: 1e-3)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs. (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help="Random seed for model. (default: 2023)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Smooth factor for surrogate learning. (default: 2.0)",
    )
    parser.add_argument(
        "--surrogate",
        nargs="?",
        default="sigmoid",
        help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')",
    )
    parser.add_argument(
        "--neuron",
        nargs="?",
        default="PLIF",
        help="Spiking neuron used for training. (IF, LIF, PLIF). (default: PLIF)",
    )
    parser.add_argument(
        "--reset",
        nargs="?",
        default="subtract",
        help="Ways to reset spiking neuron. (zero, subtract). (default: subtract)",
    )
    parser.add_argument(
        "--act",
        nargs="?",
        default="elu",
        help="Activation function. (relu, elu, None). (default: elu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5e-3,
        help="Voltage threshold in spiking neuron. (default: 5e-3)",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=30,
        help="Time steps for spiking neural networks. (default: 30)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability. (default: 0.5)"
    )
    parser.add_argument(
        "--dropedge",
        type=float,
        default=0.2,
        help="Edge dropout probability. (default: 0.2)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Margin used in ranking loss. (default: 0.0)",
    )
    parser.add_argument('--bn', action='store_true',
                    help='Whether to use batch normalization. (default: False)')
    parser.add_argument('--no_shuffle', action='store_true',
                    help='Whether to perform feature shuffling augmentation. (default: False)')    
    parser.add_argument('--deg_bins', type=int, default=-1,
                    help='Number of bins for degree clustering. (default: -1)')
    try:
        args = parser.parse_args()
        tab_printer(args)
        return args
    except:
        parser.print_help()
        exit(0)


args = read_parser()
seed_everything(args.seed)

data = get_dataset(
    root=args.root,
    dataset=args.dataset,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.neuron == "DASGNN":
    degree_to_label = {}
    all_degrees = []            
    node_degrees = degree(data.edge_index[0], dtype=torch.long)
    all_degrees.append(node_degrees.numpy())  # Convert to numpy for KMeans

    # Concatenate all degree arrays into a single numpy array
    all_degrees = np.concatenate(all_degrees).reshape(-1, 1)  # Reshape for KMeans
    if args.deg_bins == -1:
        unique_degrees = np.unique(all_degrees)
        args.deg_bins = len(unique_degrees)  # Set number of clusters to unique degrees count
    kmeans = KMeans(n_clusters=args.deg_bins, random_state=args.seed).fit(all_degrees)
    cluster_labels = kmeans.labels_

    # Sort centroids and create a new mapping of labels
    centroids = kmeans.cluster_centers_.squeeze()
    sorted_indices = np.argsort(centroids)
    label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
    new_labels = np.array([label_map[label] for label in cluster_labels])

    # Group degrees by new labels
    degree_label_mapping = {}

    for ddegree, label in zip(all_degrees.squeeze(), new_labels):
        if label not in degree_label_mapping:
            degree_label_mapping[label] = []
        degree_label_mapping[label].append(ddegree)

    # Print the mapping of degrees to new labels
    for label, degrees in degree_label_mapping.items():
        print(f"Label {label}: Degrees {set(degrees)}")

    # Optionally, print more structured output
    print("\nStructured Mapping:")
    for label in sorted(degree_label_mapping):
        print(f"Label {label}: Degrees {set(degree_label_mapping[label])}")
    for label, degrees in degree_label_mapping.items():
        for degree in degrees:
            degree_to_label[degree] = label


    model = SpikeGCL_DASGNN(
        data.x.size(1),
        args.hids,
        args.outs,
        args.T,
        args.alpha,
        args.surrogate,
        args.threshold,
        args.neuron,
        args.reset,
        args.act,
        args.dropedge,
        args.dropout,
        bn=args.bn,
        shuffle=not args.no_shuffle,
        deg_labels=degree_to_label,
        bins=args.deg_bins,
    )
else:
    model = SpikeGCL_GraphClassification(
    data.x.size(1),
    args.hids,
    args.outs,
    args.T,
    args.alpha,
    args.surrogate,
    args.threshold,
    args.neuron,
    args.reset,
    args.act,
    args.dropedge,
    args.dropout,
    bn=args.bn,
    shuffle=not args.no_shuffle,
)


print(model)
model, data = model.to(device), data.to(device)
optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=args.lr)

def train():
    model.train()
    optimizer.zero_grad()
    loss_total = 0.0
    z1s, z2s = model(data.x, data.edge_index, data.edge_attr)

    # 모든 z1, z2 쌍에 대한 loss를 먼저 계산하여 합산
    for z1, z2 in zip(z1s, z2s):
        loss = model.loss(z1, z2, args.margin)
        loss_total += loss
    
    # 누적된 전체 loss에 대해 한 번만 backward 호출
    loss_total.backward()
    optimizer.step()
    return loss_total.item()


best_val_acc = final_test_acc = 0

for epoch in range(1, args.epochs + 1):
    loss = train()
    model.eval()
    with torch.no_grad():
        embeds = model.encode(data.x, data.edge_index, data.edge_attr)
        embeds = torch.cat(embeds, dim=-1)
        print("=" * 100)
        print(f"Firing rate: {embeds.mean().item():.2%}")
        print("=" * 100)
    val_accs, test_accs = test(embeds, data, data.num_classes)
    val_acc = np.mean(val_accs)
    test_acc = np.mean(test_accs)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        final_test_acc = test_acc

    log(Epoch=epoch, Loss=loss, val_acc=val_acc, test_acc=test_acc, best=final_test_acc)

log(Final_Acc=final_test_acc)
