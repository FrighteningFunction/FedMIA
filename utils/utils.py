import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sp

from sklearn.manifold import TSNE
from copy import deepcopy


from sklearn.metrics import precision_recall_curve, auc

import string

import json

import torch
import torch.nn.functional as FN

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def plotGraphWithColors( G: nx.Graph, P, P2 = None, index = 0 ):
    fig = plt.figure( figsize=(25,10), facecolor='white' )
    ax = fig.add_gridspec( 2, 3 )
    ax1 = fig.add_subplot(ax[0:2, 0:2])
    ax2 = fig.add_subplot(ax[0, -1])
    ax3 = fig.add_subplot(ax[1, -1])
    nx.draw_networkx( G, with_labels=True, node_color=P[:,index], font_color = 'r', pos=nx.spring_layout(G, seed = 42), ax=ax1 )

    # Create a color map using viridis
    cmap = plt.get_cmap('viridis')
    ax2.bar(np.arange(len(P[:,index])), P[:,index], color=cmap(P[:,index]))
    # Set the x-tick labels to show the index of each value
    ax2.set_xticks(np.arange(len(P[:,index])))

    cmap2 = plt.get_cmap('viridis')
    ax3.bar(np.arange(len(P2[:,index])), P2[:,index], color=cmap2(P2[:,index]))
    # Set the x-tick labels to show the index of each value
    ax3.set_xticks(np.arange(len(P2[:,index])))

    plt.show()

def plotGraphWithPies( G: nx.Graph, E, H, title = None, show_labels = True, radius = 0.04, save_to_file = None  ):
    pos = nx.spring_layout(G, seed = 42)

    fig, ax = plt.subplots( ncols=4, figsize=(25,10), facecolor='white' )
    
    if title != None:
        ax[0].set_title( title )

    ax[0].matshow( nx.adjacency_matrix(G).todense() )
    ax[1].matshow( 1.0 - np.exp( - E ))
    ax[2].matshow( H )
    
    viridis = plt.cm.get_cmap('viridis', H.shape[1])

    nx.draw_networkx_edges(G, pos=pos, ax=ax[3])

    for key, value in dict( zip( range(G.number_of_nodes()), list(G.nodes) ) ).items():
        vec = H[key,:]
        if sum(vec) == 0.0:
            continue
        
        w = plt.pie(
            vec,
            center= pos[value],
            colors=viridis(range(H.shape[1])),
            radius=np.sqrt(sum(vec))*radius,
        )

        if show_labels:
            plt.text( pos[value][0]-0.01, pos[value][1]-0.01, s=str(key), color='red' )

    ax[3].set_xlim(np.min([p[0] for x,p in pos.items()]),np.max([p[0] for x,p in pos.items()]))
    ax[3].set_ylim(np.min([p[1] for x,p in pos.items()]),np.max([p[1] for x,p in pos.items()]))

    if save_to_file != None:
        plt.savefig(save_to_file)
        plt.close();
    
def getNumberOfLengthNPaths(G, start_node, end_node, n):
    # Get all the simple paths between the start and end nodes
    paths = nx.all_simple_paths(G, start_node, end_node, cutoff=n+1)

    # Filter the paths to count only the ones with length n
    n_length_paths = [p for p in paths if len(p) == n+1]

    # Count the number of n-length paths
    count = len(n_length_paths)

    return count, n_length_paths

def cosine_similarity(a, b, eps=1e-8):
    """
    Computes cosine similarity of all columns in a matrix with all columns in an other matrix.
    """
    a_n, b_n = a.norm(dim=0)[None, :], b.norm(dim=0)[None, :]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm.transpose(0, 1), b_norm)
    return sim_mt

def cosine_self_similarity(a, eps=1e-8):
    """
    Computes cosine similarity of all columns in a matrix.
    """
    a_n = a.norm(dim=0)[None, :]
    a_norm = a / torch.clamp(a_n, min=eps)
    sim_mt = torch.mm(a_norm.transpose(0, 1), a_norm)
    return sim_mt



def select_random_indices(tensor, count = 1):
    unique_elements = torch.unique(tensor)
    random_indices = []

    for element in unique_elements:
        indices = torch.nonzero(tensor == element).flatten()
        random_index = torch.randint(0, indices.size(0), (count,))
        selected_index = indices[random_index]
        random_indices.append(selected_index)

    random_indices = torch.cat(random_indices)

    return random_indices

def groundtruth_communities_to_one_hot_array(indices_list, n):
    num_classes = len(indices_list)
    one_hot_array = np.zeros((n, num_classes), dtype=int)
    for i, indices in enumerate(indices_list):
        one_hot_array[indices, i] = 1
    return one_hot_array

def overlapping_nmi(X, Y):
    """Compute NMI between two overlapping community covers.

    Parameters
    ----------
    X : array-like, shape [N, m]
        Matrix with samples stored as columns.
    Y : array-like, shape [N, n]
        Matrix with samples stored as columns.

    Returns
    -------
    nmi : float
        Float in [0, 1] quantifying the agreement between the two partitions.
        Higher is better.

    References
    ----------
    McDaid, Aaron F., Derek Greene, and Neil Hurley.
    "Normalized mutual information to evaluate overlapping
    community finding algorithms."
    arXiv preprint arXiv:1110.2515 (2011).

    """
    if not ((X == 0) | (X == 1)).all():
        raise ValueError("X should be a binary matrix")
    if not ((Y == 0) | (Y == 1)).all():
        raise ValueError("Y should be a binary matrix")

    if X.shape[1] > X.shape[0] or Y.shape[1] > Y.shape[0]:
        print("It seems that you forgot to transpose the F matrix")
    X = X.T
    Y = Y.T
    def cmp(x, y):
        """Compare two binary vectors."""
        a = (1 - x).dot(1 - y)
        d = x.dot(y)
        c = (1 - y).dot(x)
        b = (1 - x).dot(y)
        return a, b, c, d

    def h(w, n):
        """Compute contribution of a single term to the entropy."""
        if w == 0:
            return 0
        else:
            return -w * np.log2(w / n)

    def H(x, y):
        """Compute conditional entropy between two vectors."""
        a, b, c, d = cmp(x, y)
        n = len(x)
        if h(a, n) + h(d, n) >= h(b, n) + h(c, n):
            return h(a, n) + h(b, n) + h(c, n) + h(d, n) - h(b + d, n) - h(a + c, n)
        else:
            return h(c + d, n) + h(a + b, n)
    def H_uncond(X):
        """Compute unconditional entropy of a single binary matrix."""
        return sum(h(x.sum(), len(x)) + h(len(x) - x.sum(), len(x)) for x in X)

    def H_cond(X, Y):
        """Compute conditional entropy between two binary matrices."""
        m, n = X.shape[0], Y.shape[0]
        scores = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                scores[i, j] = H(X[i], Y[j])
        return scores.min(axis=1).sum()

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimensions of X and Y don't match. (Samples must be stored as COLUMNS)")
    H_X = H_uncond(X)
    H_Y = H_uncond(Y)
    I_XY = 0.5 * (H_X + H_Y - H_cond(X, Y) - H_cond(Y, X))
    return I_XY / max(H_X, H_Y)

def inv_softplus(bias: float | torch.Tensor) -> float | torch.Tensor:
    """Inverse softplus function.

    Args:
        bias (float or tensor): the value to be softplus-inverted.
    """
    is_tensor = True
    if not isinstance(bias, torch.Tensor):
        is_tensor = False
        bias = torch.tensor(bias)
    out = bias.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out

def getAdjacencySparseTensors( G : nx.MultiDiGraph | nx.DiGraph, transpose : bool = True ):
    """
    Returns the sparse adjacency matrix (in case of DiGraph) / matrices (in case of MultiDiGraph) 
    of a networkx graph as a dictionary of sparse pytorch tensors
    """

    # Initialize a dictionary to store sparse tensors for each edge type
    adj_matrices = dict()
    
    if isinstance(G, nx.MultiDiGraph):
        # Get all the unique edge keys (representing edge types)
        edge_keys = set([key for u, v, key in G.edges(keys=True)])
        
        # Initialize the dictionary to store a None value for each key
        for key in edge_keys:
            adj_matrices[key] = None
        
        # Get a dictionary of ids -> positions
        id_to_position = {value: index for index, value in enumerate(G.nodes())}

        # Populate adjacency matrices based on edge types
        for u, v, key in G.edges(keys=True):
            i, j = id_to_position[u], id_to_position[v] 

            # Create the sparse tensor
            edge_indices = torch.tensor([[i, j]], dtype=torch.long).t()
            edge_values = torch.tensor([1], dtype=torch.float)
            size = torch.Size((len(G), len(G)))

            # Initialize the sparse tensor or add to it if already initialized
            if adj_matrices[key] is None:
                adj_matrices[key] = torch.sparse.FloatTensor(edge_indices, edge_values, size)
            else:
                adj_matrices[key] += torch.sparse.FloatTensor(edge_indices, edge_values, size)

    elif isinstance(G, nx.DiGraph):
        # Convert the NetworkX graph to a SciPy sparse matrix
        adj_matrix = nx.to_scipy_sparse_array(G).tocoo()
        size = G.number_of_nodes()
        # check size
        assert size == adj_matrix.shape[0]

        # Convert the SciPy sparse matrix to a PyTorch sparse tensor
        adj_matrices["0"] = torch.sparse.FloatTensor( torch.LongTensor([adj_matrix.row, adj_matrix.col]),
                                                      torch.FloatTensor(adj_matrix.data),
                                                      torch.Size(adj_matrix.shape) )

    else:
        raise AttributeError()
    
    if transpose:
        for key in adj_matrices:
            adj_matrices[key].t_()

    return adj_matrices

def getIdentitySparseTensor( size : int ):
    return torch.sparse.FloatTensor( torch.arange(0, size).long().unsqueeze(0).expand(2, size), 
                                     torch.tensor(1.0).expand(size), 
                                     torch.Size((size,size)) )

def hasParallelEdges( graph : nx.Graph, verbose : bool = False ):
    # Dictionary to store parallel edges (source, target) -> [edge keys]
    parallel_edges = {}

    # Iterate through the edges and collect parallel edges
    for u, v, key in graph.edges(keys=True):
        if (u, v) not in parallel_edges:
            parallel_edges[(u, v)] = []

        parallel_edges[(u, v)].append(key)

    # Filter out nodes with only a single edge (not parallel)
    parallel_edges = {edge: keys for edge, keys in parallel_edges.items() if len(keys) > 1}

    if verbose:
        # Print the parallel edges
        for edge, keys in parallel_edges.items():
            print(f"Parallel edges between nodes {edge}: {keys}")

    return len(parallel_edges) > 0

def getAllAncestors( graph : nx.Graph, term_id : str ):
    return [term_id] + list(nx.ancestors(graph, term_id))  

def compute_level( graph : nx.Graph, goid, root):
    if goid != root :
        return len(max(nx.all_simple_paths(graph, goid, root), key=lambda x: len(x)))-1
    else:
        return 0
    
def compute_directed_path_matrix( graph : nx.Graph ):
    """
    Computes a dense matrix indicating directed paths in a graph.

    Parameters:
    graph (nx.DiGraph): A directed graph (NetworkX).

    Returns:
    np.ndarray: Dense matrix representing directed paths.
    """
    # Get the nodes in the graph
    nodes = list(graph.nodes())
    id_to_position = {value: index for index, value in enumerate(graph.nodes())}

    # Initialize an empty matrix filled with zeros
    num_nodes = len(nodes)
    path_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)

    # Iterate through each node as the destination (i)
    for i in range(num_nodes):
        # Depth-first search from node i to find paths
        paths = nx.single_source_shortest_path_length(graph, nodes[i])
        for j in paths:
            # Mark the corresponding position as True if a path exists
            path_matrix[i, id_to_position[j]] = True

    return path_matrix.astype(np.float32)

def calculateMetrics( y_preds : dict, ys : dict, masks : dict ):

    accuracies = {}
    sensitivities = {}
    specificities = {}
    precisions = {}
    f1_scores = {}
    auprcs = {}

    for key in y_preds:
        probabilities = torch.sigmoid(y_preds[key]) if masks[key].dim() == 2 else y_preds[key]

        # Step 2: Apply mask
        probabilities_masked = probabilities[masks[key]] if masks[key].dim() == 2 else probabilities[masks[key],:]
        ground_truth_masked = ys[key][masks[key]] if masks[key].dim() == 2 else ys[key][masks[key],:]

        # Step 3: Apply thresholding (e.g., 0.5)
        binary_predictions = (probabilities_masked > 0.5).float()

        # Step 4: Compute confusion matrix
        true_positive = (binary_predictions * ground_truth_masked).sum()
        false_positive = (binary_predictions * (1 - ground_truth_masked)).sum()
        false_negative = ((1 - binary_predictions) * ground_truth_masked).sum()
        true_negative = ((1 - binary_predictions) * (1 - ground_truth_masked)).sum()

        # Step 5: Compute accuracy, sensitivity, specificity, precision, and F1 score
        accuracies[key] = ((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)).item()
        sensitivities[key] = (true_positive / (true_positive + false_negative)).item()
        specificities[key] = (true_negative / (true_negative + false_positive)).item()
        precisions[key] = (true_positive / (true_positive + false_positive)).item()
        f1_scores[key] = (2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative)).item()

        # Step 6: Compute Area Under the Precision-Recall Curve
        # Initialize a list to store AUC-PR values for each label
        auc_pr_values = []
        # Iterate through each column (label)
        for label_idx in range(probabilities.shape[1]):            
            # Get the evaluation mask for the current label
            label_mask = masks[key][:, label_idx] if masks[key].dim() == 2 else masks[key]
            # Check if there are any true items in the evaluation mask for this label
            if label_mask.any():
                # Get the predicted probabilities, and ground truth labels for the current label
                label_predictions = probabilities[:, label_idx]
                label_ground_truth = ys[key][:, label_idx]
                
                # Filter instances based on the evaluation mask
                label_predictions = label_predictions[label_mask]
                label_ground_truth = label_ground_truth[label_mask]

                # Compute the precision-recall curve
                precision, recall, _ = precision_recall_curve(label_ground_truth.cpu().numpy(), label_predictions.cpu().numpy())

                # Compute the AUC-PR for the current label
                label_auc_pr = auc(recall, precision)

                # Append the AUC-PR value to the list
                auc_pr_values.append(label_auc_pr)
           
        auprcs[key] = np.mean(auc_pr_values)

    return accuracies, sensitivities, specificities, precisions, f1_scores, auprcs


def saveResultsToJSON( internal_evaluation_scores, pred_communities, filename : str ):
    pred_communities.communities = [[int(value) for value in sublist] for sublist in pred_communities.communities]

    parsed_dict = json.loads(pred_communities.to_json())

    # Step 2: Convert NumPy int64 to Python int
    existing_dict_converted = {key: int(value) if isinstance(value, np.int64) else value for key, value in internal_evaluation_scores.items()}

    # Step 2: Merge dictionaries
    merged_dict = {**existing_dict_converted, **parsed_dict}

    # Optionally, convert merged dictionary back to JSON-formatted string
    merged_json_string = json.dumps(merged_dict)

    # Open the file in append mode
    with open(filename, "wt") as file:
        file.write(merged_json_string)

def saveResultsToJSON3( internal_evaluation_scores, communities, filename : str, node_index_dict ):
    
    partition = {
        "communities": [[node_index_dict[int(value)] for value in sublist] for sublist in communities]
    }

    # Step 2: Convert NumPy int64 to Python int
    existing_dict_converted = {key: int(value) if isinstance(value, np.int64) else value for key, value in internal_evaluation_scores.items()}

    # Step 2: Merge dictionaries
    merged_dict = {**existing_dict_converted, **partition}

    # Optionally, convert merged dictionary back to JSON-formatted string
    merged_json_string = json.dumps(merged_dict)

    # Open the file in append mode
    with open(filename, "wt") as file:
        file.write(merged_json_string)


class PartialFormatter(string.Formatter):
    def __init__(self, missing='~~', bad_fmt='!!'):
        self.missing, self.bad_fmt=missing, bad_fmt

    def get_field(self, field_name, args, kwargs):
        # Handle a key not found
        try:
            val=super(PartialFormatter, self).get_field(field_name, args, kwargs)
            # Python 3, 'super().get_field(field_name, args, kwargs)' works
        except (KeyError, AttributeError):
            val=None,field_name 
        return val 

    def format_field(self, value, spec):
        # handle an invalid format
        if value==None: return self.missing
        try:
            return super(PartialFormatter, self).format_field(value, spec)
        except ValueError:
            if self.bad_fmt is not None: return self.bad_fmt   
            else: raise

class ModelSaver:
    """In-memory saver for model parameters.

    Storing weights in memory is faster than saving to disk with torch.save.
    """
    def __init__(self, model):
        self.model = model
        self.state_dict = None

    def save(self):
        self.state_dict = deepcopy(self.model.state_dict())

    def restore(self):
        if self.state_dict != None:
            self.model.load_state_dict(self.state_dict)


class EarlyStopping:
    """Stop training when the validation metric stops improving.

    Parameters
    ----------
    validation_fn : function
        Calling this function returns the current value of the validation metric.
    patience : int
        Number of iterations without improvement before stopping.
    tolerance : float
        Minimal improvement in validation metric to not trigger patience.
    
    Attributes
    ----------
    _best_value : float
        Best value of the validation loss.
    _num_bad_epochs : int
        Number of epochs since last significant improvement in validation metric.
    _time_to_save : bool
        Is it time to save the model weights?
    _is_better : function
        Tells if new validation metric value is better than the best one so far.
        Signature self._is_better(new_value, best_value).

    """
    def __init__(self, validation_fn, patience=10, tolerance=0.0):
        super().__init__()
        self.validation_fn = validation_fn
        self.patience = patience
        self.tolerance = tolerance
        self.reset()

        self._is_better = lambda new, best: new < best - tolerance

    def reset(self):
        """Reset the internal state."""
        self._best_value = self.validation_fn()
        self._num_bad_epochs = 0
        self._time_to_save = False

    def next_step(self):
        """Should be called at every iteration."""
        last_value = self.validation_fn()
        if self._is_better(last_value, self._best_value):
            self._time_to_save = True
            self._best_value = last_value
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1

    def should_save(self):
        """Says if it's time to save model weights."""
        if self._time_to_save:
            self._time_to_save = False
            return True
        else:
            return False

    def should_stop(self):
        """Says if it's time to stop training."""
        return self._num_bad_epochs >= self.patience

