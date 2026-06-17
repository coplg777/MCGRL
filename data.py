






















import numpy as np
from dataset_utils import get_dataset

def load_data(args):
    dataset = get_dataset(args, normalize=args.normalize)



    if hasattr(dataset, 'num_node_features'):
        args.num_features = dataset.num_node_features
    elif hasattr(dataset, 'data') and hasattr(dataset.data, 'num_features'):
        args.num_features = dataset.data.num_features
    else:
        raise AttributeError("Cannot find num_features on dataset")


    if hasattr(dataset, 'num_classes'):
        args.num_classes = dataset.num_classes
    else:

        args.num_classes = int(dataset[0].y.max().item()) + 1


    args.avg_num_nodes = np.ceil(
        np.mean([data.num_nodes for data in dataset])
    )


    print('# %s: [FEATURES]-%d [NUM_CLASSES]-%d [AVG_NODES]-%d' %
          (dataset, args.num_features, args.num_classes, args.avg_num_nodes))

    return dataset

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)
