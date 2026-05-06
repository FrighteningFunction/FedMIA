# Notes:
# - Graph must be a DAG with edges directed from inputs (more specific) -> outputs (more general).
# - Leaf/input nodes have in-degree 0.
# - Nodes with out-degree 0 are "roots" of the ontology (most general).
# - Uses torch_sparse.spmm for fast sparse-dense multiplication.

import math
import pandas as pd

from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm, coalesce


class GraphBasedModelV2( nn.Module ):
    """
    Sparse DAG/BINN model driven by an ontology graph.

    Each ontology node is a neuron; ontology edges define allowed connections.
    The model constructs a feed-forward computation that respects DAG dependencies,
    including skip connections across multiple layers.

    Forward pass:
      - Input layer (layer 0): user-provided features for input nodes.
      - For each ontology layer ell=1..L:
          z^{(ell)} = W^{(ell)} * concat( a^{(0)},...,a^{(ell-1)} ) + b^{(ell)}
          a^{(ell)} = activation(z^{(ell)})
          dropout optionally applied on activations.
      - Output:
          explicit residual sum with fixed coefficients:
              y = sum_{ell in output_layers} coeff[ell] * Linear_ell( a^{(ell)} )

    This gives you:
      - scalable sparse computation (one spmm per layer)
      - clean “multi-resolution” readout with fixed residual coefficients
      - no batchnorm, no interpretability extras
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        output_size: int,
        *,
                                                              # Output configuration
        output_last_layers: Optional[ int ] = None,
        output_coeffs: Optional[ Dict[ int, float ] ] = None,
                                                              # Nonlinearity / regularization
        non_linearity: str = "relu",
        dropout_prob: float = 0.0,
                                                              # Device / dtype
        device: str = "cpu",
                                                              # Parameter init
        init_weight_scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        graph
            A directed acyclic MultiDiGraph. Edges must point from inputs -> outputs.
        output_size
            Dimension of the final prediction vector.
        output_last_layers
            If provided, build readouts from the last `output_last_layers` ontology layers.
            If None, default to only the final layer.
        output_coeffs
            Fixed coefficients for the residual output sum, keyed by ontology layer index.
            Example: {L: 1.0} (only last layer), or {L-1:0.5, L:0.5}.
            If None, coefficients are set uniformly across selected output layers.
        non_linearity
            One of {'relu','gelu','tanh','sigmoid','elu','leaky_relu','identity'}.
        dropout_prob
            Dropout probability applied to each hidden layer activation (no BN).
        device
            Torch device string, e.g. 'cpu' or 'cuda:0'.
        init_weight_scale
            Multiplier for sparse weight initialization magnitude.
        """
        super().__init__()

        assert graph.number_of_nodes() > 0, "Graph is empty."
        assert nx.is_directed_acyclic_graph( graph ), "Graph must be a DAG."

        self.device = torch.device( device )
        self.graph = graph
        self.output_size = int( output_size )

        # -----------------------------
        # Activation function selection
        # -----------------------------
        act = non_linearity.lower()
        self._activations = {
            "relu": F.relu,
            "gelu": F.gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "elu": F.elu,
            "leaky_relu": F.leaky_relu,
            "identity": lambda x: x,
        }
        if act not in self._activations:
            raise ValueError( f"Invalid activation '{non_linearity}'. Options: {list(self._activations)}" )
        self.activation = self._activations[ act ]

        self.dropout_prob = float( dropout_prob )
        self.use_dropout = self.dropout_prob > 0.0
        self.dropout = nn.Dropout( p = self.dropout_prob ) if self.use_dropout else None

        # -----------------------------
        # Identify input nodes (layer 0)
        # -----------------------------
        # Input nodes are those with in-degree 0.
        self.input_nodes: List[ str ] = [ n for n in graph.nodes if graph.in_degree( n ) == 0 ]
        if len( self.input_nodes ) == 0:
            raise ValueError( "No input nodes found (in-degree 0). Cannot define the model input." )

        # -----------------------------
        # Topological order (stable index)
        # -----------------------------
        self.sorted_nodes: List[ str ] = list( nx.topological_sort( graph ) )

        # Basic sanity check: topological ordering should not reshuffle input nodes relative to DAG constraints.
        topo_input = [ n for n in self.sorted_nodes if graph.in_degree( n ) == 0 ]
        assert topo_input == self.input_nodes, ( "Topological sorting changed the input node order. "
                                                 "If you need a stable custom input order, pass graph nodes already ordered." )

        # -----------------------------
        # Layer assignment
        # -----------------------------
        # Your BFS layering assigns each node to the maximum distance from inputs,
        # then moves ontology roots (out-degree 0, in-degree>0) to the last layer.
        self.node_layers: Dict[ str, int ] = self.layering_bfs()

        self.number_of_layers: int = max( self.node_layers.values() )
        if self.number_of_layers <= 0:
            raise RuntimeError( "Invalid layering produced no layers." )

        # Precompute nodes grouped by layer for fast indexing.
        self.nodes_by_layer: Dict[ int, List[ str ] ] = {
            ell: [ n for n in self.sorted_nodes if self.node_layers[ n ] == ell ]
            for ell in range( self.number_of_layers + 1 )
        }

        # Input dimensionality is the number of input nodes.
        self.input_size = len( self.nodes_by_layer[ 0 ] )

        # -----------------------------
        # Output layers selection
        # -----------------------------
        if output_last_layers is None:
            output_last_layers = 1 # default: only final layer readout

        output_last_layers = int( output_last_layers )
        if output_last_layers <= 0:
            raise ValueError( "output_last_layers must be >= 1." )

        # Example: L=5, output_last_layers=2 => layers {4,5}
        start_out_layer = max( self.number_of_layers - output_last_layers + 1, 0 )
        self.output_layers: List[ int ] = list( range( start_out_layer, self.number_of_layers + 1 ) )

        # Dense readout (per selected layer).
        # Each readout maps that layer's activations -> output_size.
        self.readouts = nn.ModuleDict()
        for ell in self.output_layers:
            dim_ell = len( self.nodes_by_layer[ ell ] )
            self.readouts[ f"layer_{ell}" ] = nn.Linear( dim_ell, self.output_size )

        # -----------------------------
        # Fixed residual coefficients
        # -----------------------------
        # These are NOT learnable: they define an explicit fixed residual sum to form output.
        # If user did not provide coefficients, default to uniform weights over output layers.
        if output_coeffs is None:
            w = 1.0 / max( len( self.output_layers ), 1 )
            output_coeffs = { ell: w for ell in self.output_layers }
        else:
            # If provided, we still ensure every output layer has a coefficient.
            # Missing layers get 0.
            output_coeffs = { int( k ): float( v ) for k, v in output_coeffs.items() }
            for ell in self.output_layers:
                output_coeffs.setdefault( ell, 0.0 )

        # Store coefficients as a buffer tensor aligned with self.output_layers.
        coeff_vec = torch.tensor( [ output_coeffs[ ell ] for ell in self.output_layers ], dtype = torch.float32 )
        self.register_buffer( "output_coeffs", coeff_vec ) # shape: (#out_layers,)

        # -----------------------------
        # Build ONE sparse weight matrix per layer
        # -----------------------------
        # For each layer ell, we connect ALL nodes from layers < ell into ell.
        #
        # We define a parent concatenation order:
        #   parent_nodes(ell) = nodes_by_layer[0] + nodes_by_layer[1] + ... + nodes_by_layer[ell-1]
        #
        # Then W^{(ell)} has shape (|ell|, sum_{p<ell} |p|).
        #
        # Forward uses:
        #   rhs = cat( a^{(0)}, ..., a^{(ell-1)} )  shape (sum_{p<ell}|p|, B)
        #   z   = spmm(indices_ell, values_ell, rows, cols, rhs)
        #
        self.biases = nn.ParameterDict()                           # bias per layer ell (ell>=1)
        self._parent_nodes_by_layer: Dict[ int, List[ str ] ] = {} # python-side cache of parent order
        self._parent_sizes_by_layer: Dict[ int, int ] = {}

        # One parameter vector per layer (values), plus buffers for indices and sizes.
        for ell in range( 1, self.number_of_layers + 1 ):
            child_nodes = self.nodes_by_layer[ ell ]
            child_dim = len( child_nodes )

            # Bias for this layer's nodes (broadcasted across batch).
            self.biases[ str( ell ) ] = nn.Parameter( torch.zeros( child_dim, dtype = torch.float32 ) )

            # Parent concatenation order (fixed and reused each forward).
            parent_nodes: List[ str ] = []
            for p in range( ell ):
                parent_nodes.extend( self.nodes_by_layer[ p ] )
            parent_dim = len( parent_nodes )

            self._parent_nodes_by_layer[ ell ] = parent_nodes
            self._parent_sizes_by_layer[ ell ] = parent_dim

            # Map nodes to row/col indices in the sparse matrix.
            child_to_row = { n: i for i, n in enumerate( child_nodes ) }
            parent_to_col = { n: i for i, n in enumerate( parent_nodes ) }

            # Collect unique (u,v) edges that land in this layer.
            #
            # IMPORTANT for MultiDiGraph:
            # - We collapse parallel edges: if there are multiple edges u->v, we treat it as one connection.
            # - This avoids duplicated sparse indices (which can be problematic unless coalesced).
            edge_set = set()
            for u, v in graph.edges():
                if self.node_layers.get( v, None ) != ell:
                    continue
                # only parents from earlier layers
                if self.node_layers.get( u, None ) is None or self.node_layers[ u ] >= ell:
                    continue
                edge_set.add( ( u, v ) )

            # If there are no incoming edges into this layer from earlier layers,
            # the layer cannot be computed (except by bias). This is usually a graph issue.
            # We still allow it (z will be zeros), but warn via a runtime assert-like check.
            if len( edge_set ) == 0:
                # Create a dummy empty sparse structure.
                indices = torch.empty( ( 2, 0 ), dtype = torch.long )
                values = nn.Parameter( torch.empty( ( 0, ), dtype = torch.float32 ) )
            else:
                # Convert edge list into COO indices (row=child, col=parent).
                rows = []
                cols = []
                for ( u, v ) in edge_set:
                    # v must be in this layer
                    r = child_to_row[ v ]
                    # u must be in parent set
                    if u not in parent_to_col:
                        # This can happen only if layering produced an inconsistency.
                        continue
                    c = parent_to_col[ u ]
                    rows.append( r )
                    cols.append( c )

                indices = torch.tensor( [ rows, cols ], dtype = torch.long )

                # Coalesce indices (sort + unique) to be safe.
                # coalesce returns (idx, val) with summed values for duplicates,
                # but we start with "val=1" placeholders and replace with parameters.
                if indices.numel() > 0:
                    idx_sorted, _ = coalesce( indices, torch.ones( indices.shape[ 1 ] ), m = child_dim, n = parent_dim )
                    indices = idx_sorted
                values = nn.Parameter( torch.zeros( ( indices.shape[ 1 ], ), dtype = torch.float32 ) )

            # Register buffers for sparse structure.
            self.register_buffer( f"_sparse_indices_{ell}", indices )
            self.register_buffer(
                f"_sparse_sizes_{ell}",
                torch.tensor( [ child_dim, parent_dim ], dtype = torch.long ),
            )
            # Register the parameter vector of sparse weights.
            self.register_parameter( f"_sparse_weights_{ell}", values )

        # -----------------------------
        # Initialize parameters
        # -----------------------------
        self.trainable_params = sum( p.numel() for p in self.parameters() if p.requires_grad )
        self.reset_parameters( init_weight_scale )

        # Move module to device (buffers + params)
        self.to( self.device )

    # ---------------------------------------------------------------------
    # 2.1 Layering: keep your BFS logic (as requested)
    # ---------------------------------------------------------------------
    def layering_bfs( self ) -> Dict[ str, int ]:
        """
        Kahn-style pass that assigns each node to the maximum "distance" from any input,
        then moves ontology roots (out-degree 0, in-degree>0) to the last layer.

        This is intentionally close to your previous version ("leave alone 2.1").
        """
        in_degree = { n: self.graph.in_degree( n ) for n in self.graph.nodes }
        queue = deque( [ n for n, deg in in_degree.items() if deg == 0 ] )

        node_layers: Dict[ str, int ] = {}

        # Initialize input nodes to layer 0
        for n in queue:
            node_layers[ n ] = 0

        while queue:
            u = queue.popleft()
            for v in self.graph.successors( u ):
                node_layers[ v ] = max( node_layers.get( v, 0 ), node_layers[ u ] + 1 )
                in_degree[ v ] -= 1
                if in_degree[ v ] == 0:
                    queue.append( v )

        # Move ontology roots (out-degree 0, but not input nodes) to the final layer
        max_layer = max( node_layers.values() ) if len( node_layers ) else 0
        for n in self.graph.nodes:
            if self.graph.out_degree( n ) == 0 and self.graph.in_degree( n ) > 0:
                node_layers[ n ] = max_layer

        return node_layers

    # ---------------------------------------------------------------------
    # Parameter initialization
    # ---------------------------------------------------------------------
    def reset_parameters( self, init_weight_scale: float = 1.0 ) -> None:
        """
        Initializes:
        - Dense readout layers with PyTorch defaults.
        - Biases to zero.
        - Sparse weights with fan-in aware uniform init.

        For sparse edges, we approximate fan-in per child node by counting
        *incoming edges from any earlier layer* (consistent with our layer-wise W^{(ell)}).
        """
        # Dense readouts: standard init
        for m in self.readouts.values():
            if isinstance( m, nn.Linear ):
                m.reset_parameters()

        # Biases: zeros
        for k in self.biases:
            nn.init.zeros_( self.biases[ k ] )

        # Sparse weights: fan-in adapted uniform init.
        #
        # For each layer ell:
        # - each sparse weight corresponds to one edge u->v where v is in layer ell
        # - we want Var ~ 1/fan_in(v) scaling (Kaiming-like).
        #
        # Implementation detail:
        # - We compute fan-in per child node for edges that go into this layer from earlier layers.
        #
        for ell in range( 1, self.number_of_layers + 1 ):
            indices = getattr( self, f"_sparse_indices_{ell}" )
            sizes = getattr( self, f"_sparse_sizes_{ell}" )
            child_dim, parent_dim = int( sizes[ 0 ].item() ), int( sizes[ 1 ].item() )

            w = getattr( self, f"_sparse_weights_{ell}" )
            if w.numel() == 0:
                continue

            # fan_in_per_child[r] = number of incoming sparse edges into child node r (within this W^{(ell)})
            fan_in_per_child = torch.zeros( ( child_dim, ), dtype = torch.float32, device = indices.device )
            fan_in_per_child.scatter_add_(
                dim = 0,
                index = indices[ 0 ],
                src = torch.ones_like( indices[ 0 ], dtype = torch.float32,  ),
            )
            fan_in_per_child = torch.clamp( fan_in_per_child, min = 1.0 )

            # Initialize each edge weight using bound = sqrt(6/fan_in(child)).
            # This is analogous to Xavier/He uniform but per-node adapted.
            with torch.no_grad():
                for e in range( indices.shape[ 1 ] ):
                    r = int( indices[ 0, e ].item() )
                    bound = math.sqrt( 6.0 / float( fan_in_per_child[ r ].item() ) )
                    bound *= float( init_weight_scale )
                    w[ e ].uniform_( -bound, bound )

    # ---------------------------------------------------------------------
    # Forward pass (no interpretability, no overrides, no BN)
    # ---------------------------------------------------------------------
    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, input_size) where input_size == number of input nodes (layer 0).

        Returns
        -------
        y : torch.Tensor
            Shape (B, output_size)
        """
        if x.dim() != 2:
            raise ValueError( f"Expected x to be 2D (B,input_size), got {tuple(x.shape)}." )
        B, Din = x.shape
        if Din != self.input_size:
            raise ValueError( f"Input size mismatch: got {Din}, expected {self.input_size}." )

        # We store activations per layer in "node-major" layout:
        #   a^{(ell)} has shape (n_nodes_in_layer_ell, B)
        layer_acts: List[ torch.Tensor ] = [ None ] * ( self.number_of_layers + 1 )

        # Layer 0: input features, transpose to node-major.
        layer_acts[ 0 ] = x.t().contiguous() # (n0, B)

        # Iterate layers 1..L
        for ell in range( 1, self.number_of_layers + 1 ):
            indices = getattr( self, f"_sparse_indices_{ell}" )
            sizes = getattr( self, f"_sparse_sizes_{ell}" )
            rows, cols = int( sizes[ 0 ].item() ), int( sizes[ 1 ].item() )

            w = getattr( self, f"_sparse_weights_{ell}" )

            # Concatenate all parent activations in the SAME order we used when building parent_to_col.
            # rhs: (sum_parent_nodes, B)
            rhs = torch.cat( [ layer_acts[ p ] for p in range( ell ) ], dim = 0 ).contiguous()

            # Sparse * dense:
            # z: (rows, B) = W^{(ell)} * rhs
            #
            # If indices empty, spmm would fail; handle that explicitly.
            if w.numel() == 0 or indices.numel() == 0:
                z = torch.zeros( ( rows, B ), device = rhs.device, dtype = rhs.dtype )
            else:
                z = spmm( indices, w, rows, cols, rhs )

            # Add bias (rows,1) broadcast across batch
            z = z + self.biases[ str( ell ) ].unsqueeze( 1 )

            # Nonlinearity
            a = self.activation( z )

            # Dropout (if enabled): operate in batch-major and transpose back for efficiency
            if self.use_dropout:
                a = self.dropout( a.t() ).t()

            layer_acts[ ell ] = a

        # ---------------------------------------------------------
        # Output: explicit fixed residual sum
        # ---------------------------------------------------------
        # y = sum_j coeff_j * Readout_{ell_j}( a^{(ell_j)} )
        #
        # Each readout expects batch-major: (B, n_nodes) -> (B, output_size)
        out_terms = []
        for j, ell in enumerate( self.output_layers ):
            coeff = float( self.output_coeffs[ j ].item() )
            if coeff == 0.0:
                continue

            a_ell = layer_acts[ ell ].t()                   # (B, n_nodes_in_layer)
            term = self.readouts[ f"layer_{ell}" ]( a_ell ) # (B, output_size)
            out_terms.append( term * coeff )                # (B, output_size)

        if len( out_terms ) == 0:
            # Degenerate case: all coeffs are zero. Return zeros (still differentiable w.r.t. nothing).
            y = torch.zeros( ( B, self.output_size ), device = self.device, dtype = x.dtype )
        else:
            y = torch.stack( out_terms, dim = 0 ).sum( dim = 0 ) # (L, B, output_size) -> (B, output_size)

        return y


class GraphBasedModel( nn.Module ):
    """
    A PyTorch model that dynamically constructs its layers and forward pass
    based on the structure of a directed acyclic graph (DAG) provided as input.

    **Description:**
    - Nodes in the graph correspond to neurons in the model.
    - Edges in the graph define the connections between neurons across layers.
    - The model ensures that dependencies between neurons are respected, even
      if some connections skip multiple layers.

    **Edge Orientation:**
    - Edges must point from inputs to outputs.
    - For graphs representing ontologies, edges should point from more specific terms (inputs)
      to less specific terms (outputs). The root of the ontology is the node with an out-degree of zero.

    **Inputs:**
    - A `networkx.MultiDiGraph` representing the DAG structure.
      - Nodes: Represent individual neurons.
      - Edges: Represent connections between neurons.


    **Outputs:**
    - A tensor of shape `(batch_size, output_size)`, where `output_size` is
      the number of output nodes (nodes in the last layer of the graph).

    **Key Features:**
    - Sparse weight matrices are used for computational efficiency.
    - Supports graphs with arbitrary layer connections, including skipped connections.
    - Ensures consistency between the graph structure and the layer computations.
    - Optional batch normalization and dropout on each layer.
    - SHAP-based interpretability method.

    """

    def __init__(
            self,
            graph: nx.MultiDiGraph,
            output_size: int,
            output_method: str = "nodewise", # ['nodewise','layerwise']
            output_last_layers: int = None,
            non_linearity: str = "relu",
            use_batchnorm: bool = False,
            dropout_prob: float = 0.2 ):
        """
        Parameters
        ----------
        graph : nx.MultiDiGraph
            A DAG describing the network architecture.
        output_size : int
            Dimension of the final output layer.
        output_method : str
            One of {'nodewise','layerwise'} controlling how the final output is constructed.
        output_last_layers : int or None
            If not None, the final output is computed from the last `output_last_layers` layers
        non_linearity : str
            Activation name. Options: ['relu','sigmoid','tanh','leaky_relu','gelu','elu','identity'].
        use_batchnorm : bool
            If True, apply BatchNorm1d to each layer.
        dropout_prob : float
            Probability for Dropout layers. If larger than 0, apply Dropout to each layer.
        """
        super().__init__()

        assert graph.number_of_nodes() > 0, "No nodes in the graph"
        assert nx.is_directed_acyclic_graph( graph ), "Not a DAG"
        #assert nx.is_weakly_connected(graph), "Not connected graph"
        assert output_method in [ "nodewise", "layerwise" ], "`output_method` should be one of 'nodewise','layerwise'"

        self.graph = graph
        self.output_size = output_size
        self.output_method = output_method
        self.layers = nn.ModuleDict()

        self.activation_functions = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "leaky_relu": F.leaky_relu,
            "gelu": F.gelu,
            "elu": F.elu,
            "identity": lambda x: x,    # No activation
        }
        self.non_linearity = non_linearity.lower()
        assert self.non_linearity in self.activation_functions, f"Invalid activation: {non_linearity}"
        self.activation = self.activation_functions[ self.non_linearity ]

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout_prob > 0
        self.dropout_prob = dropout_prob

        # Identify layers
        # Leaf nodes = input nodes (in-degree=0).
        self.input_nodes = [ n for n in graph.nodes if graph.in_degree( n ) == 0 ]

        # A topological ordering of the nodes.
        self.sorted_nodes = list( nx.topological_sort( graph ) )
        self.node_to_idx = { node: idx for idx, node in enumerate( self.sorted_nodes ) }

        # Basic consistency check
        assert self.input_nodes == [ n for n in self.sorted_nodes if graph.in_degree( n ) == 0 ], ( "Topological sorting changed the input node order!" )

        # Group nodes by layer (level = max distance from any input node)
        self.node_layers = self.layering_bfs()
        self.number_of_layers = max( self.node_layers.values() )

        # Insert the final linear transform(s)
        self.output_from_layers = max( self.number_of_layers - output_last_layers + 1, 0 ) if output_last_layers != None else 0
        for layer in range( self.output_from_layers, self.number_of_layers + 1 ):
            layer_nodes = [ n for n in self.sorted_nodes if self.node_layers[ n ] == layer ]
            self.layers[ f"layer_{layer}" ] = nn.Linear( len( layer_nodes ), self.output_size )

        # Sparse weights data structures
        self.sparse_values = {}
        # Per-layer bias
        self.biases = nn.ParameterDict()

        # Create per-layer batchnorm and dropout modules if requested
        self.batchnorms = nn.ModuleDict() if self.use_batchnorm else None
        self.dropouts = nn.ModuleDict() if self.use_dropout else None

        # Construct each layer from layer=1..number_of_layers
        for layer in range( 1, self.number_of_layers + 1 ):
            # Initialize dictionaries to store sparse connections
            self.sparse_values[ layer ] = {}

            next_layer_nodes = [ n for n in self.sorted_nodes if self.node_layers[ n ] == layer ]
            next_node_to_idx = { node: idx for idx, node in enumerate( next_layer_nodes ) }
            set_of_next_layer_nodes = set( next_layer_nodes )

            # Create a bias vector for the nodes in this layer
            bias_param = nn.Parameter( torch.zeros( len( next_layer_nodes ) ) )
            self.biases[ str( layer ) ] = bias_param

            # Optionally add BatchNorm1d for this layer's dimension
            if self.use_batchnorm:
                # BatchNorm1d expects shape (batch_size, features)
                # We'll need a minor transpose in forward(), see below
                self.batchnorms[ str( layer ) ] = nn.BatchNorm1d( num_features = len( next_layer_nodes ) )

            # Optionally add Dropout for this layer
            if self.use_dropout:
                self.dropouts[ str( layer ) ] = nn.Dropout( p = self.dropout_prob )

            # Find edges from all previous layers up to `layer-1`
            for prev_layer in range( layer ):

                # if prev_layer < layer-1: # TODO: might be a hiperparameter if we don't want to use edges between non-neighboring layers
                #     continue

                prev_nodes = [ n for n in self.sorted_nodes if self.node_layers[ n ] == prev_layer ]
                prev_node_to_idx = { node: idx for idx, node in enumerate( prev_nodes ) }
                set_of_prev_nodes = set( prev_nodes )

                edges = [ ( u, v ) for ( u, v ) in graph.edges() if ( u in set_of_prev_nodes and v in set_of_next_layer_nodes ) ]
                if not edges:
                    continue

                # Indices for the sparse matrix
                indices = torch.tensor( [ [ next_node_to_idx[ v ], prev_node_to_idx[ u ] ] for ( u, v ) in edges ], dtype = torch.long ).t()
                self.register_buffer( f"_sparse_indices_{layer}_{prev_layer}", indices )

                # The actual weight values
                values = nn.Parameter( torch.zeros( indices.shape[ 1 ] ) )
                self.sparse_values[ layer ][ prev_layer ] = values
                self.register_parameter( f"_sparse_weights_{layer}_{prev_layer}", values )

                # Tensor size (rows, cols)
                sizes = torch.tensor( [ len( next_layer_nodes ), len( prev_nodes ) ], dtype = torch.long )
                self.register_buffer( f"_sparse_sizes_{layer}_{prev_layer}", sizes )

        self.trainable_params = sum( p.numel() for p in self.parameters() if p.requires_grad )
        self.reset_parameters()

    def layering_bfs( self ):
        # Kahn’s algorithm for topological sorting + layer assignment
        in_degree = { n: self.graph.in_degree( n ) for n in self.graph.nodes }
        queue = deque( [ n for n, deg in in_degree.items() if deg == 0 ] )
        node_layers = {}
        # Initialize input nodes to layer 0
        for n in queue:
            node_layers[ n ] = 0

        # BFS-like pass
        while queue:
            u = queue.popleft()
            for v in self.graph.successors( u ):
                # v’s layer is at least (u’s layer + 1)
                node_layers[ v ] = max( node_layers.get( v, 0 ), node_layers[ u ] + 1 )
                in_degree[ v ] -= 1
                if in_degree[ v ] == 0:
                    queue.append( v )

        ### Hack: those nodes that have no outgoing edges (i.e., root nodes of the ontology)
        # go to the last layer.
        max_layer = max( node_layers.values() )
        for n in self.graph.nodes:
            if self.graph.out_degree( n ) == 0 and self.graph.in_degree( n ) > 0:
                node_layers[ n ] = max_layer

        return node_layers

    def reset_parameters( self ):
        """Reinitialize all model parameters, including sparse weights."""
        # 1) Reset standard dense layers
        for layer in self.layers.values():
            if isinstance( layer, nn.Linear ):
                layer.reset_parameters() # PyTorch's built-in reset

        # 2) Reset biases to zeros
        for layer_idx in self.biases:
            nn.init.zeros_( self.biases[ layer_idx ] )

        fan_in_of_nodes = { n: self.graph.in_degree( n ) for n in self.graph.nodes }

        # 3) Reset sparse weights with a Kaiming-like approach;
        # i.e., initialize each sparse weight individually using Kaiming uniform initialization,
        # explicitly adapting to each node's fan-in.
        for layer_idx in self.sparse_values:
            for prev_layer_idx in self.sparse_values[ layer_idx ]:
                # Initialize sparse weights
                initialized_values = torch.empty_like( self.sparse_values[ layer_idx ][ prev_layer_idx ] )
                # Get nodes receiving input (current layer)
                indices = getattr( self, f"_sparse_indices_{layer_idx}_{prev_layer_idx}" )
                target_indices = indices[ 0 ]
                # Get nodes from the current layer
                next_layer_nodes = [ n for n in self.sorted_nodes if self.node_layers[ n ] == layer_idx ]
                for count_idx, target_node_idx in enumerate( target_indices ):
                    target_node = next_layer_nodes[ target_node_idx ]
                    fan_in = fan_in_of_nodes[ target_node ]
                    bound = ( 6.0 / fan_in )**0.5
                    initialized_values[ count_idx ].uniform_( -bound, bound )

                # Assign initialized values back
                with torch.no_grad():
                    self.sparse_values[ layer_idx ][ prev_layer_idx ].copy_( initialized_values )

    def get_parameter_groups( self, weight_decay: float = 0.0 ):
        """
        Create two parameter groups for the optimizer:
        1) 'wd_params': parameters that should have weight decay (e.g. dense weights, sparse weights).
        2) 'no_wd_params': parameters that should NOT have weight decay (e.g. all biases, BatchNorm).

        Parameters
        ----------
        weight_decay : float or None
            - The custom weight decay factor that should be used.

        Returns
        -------
        param_groups : list of dict
            A parameter-group structure suitable for passing to torch.optim. 
            e.g.:
                optimizer = torch.optim.Adam(param_groups, lr=..., weight_decay=...) 
                # or pass 0 for weight_decay and rely on the parameter-group definitions.
        """
        wd_params = []
        no_wd_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # We'll exclude anything named "bias" or coming from batch norm modules.
            if ( "bias" in name ) or ( "batchnorms" in name ): # or (".bn" in name)
                                                               # --> No weight decay
                no_wd_params.append( param )
            else:
                                                               # --> Apply weight decay
                wd_params.append( param )

        param_groups = [ { "params": wd_params, "weight_decay": weight_decay }, { "params": no_wd_params, "weight_decay": 0.0 } ]
        return param_groups

    def forward( self, x = None, capture_activations = False, override_layer_idx = None, override_activations = None ):
        """
        A unified forward pass that can do:
        1) Normal full forward from input x, if override_layer_idx is None.
        2) Partial forward from 'override_layer_idx' onward, 
            using 'override_activations[override_layer_idx]' as that layer's activation,
            and 'override_activations[l]' for all layers < override_layer_idx.

        Parameters
        ----------
        x : torch.Tensor or None
            If not overriding early layers, this is your input shape (batch_size, input_layer_size).
        capture_activations : bool
            Whether to store a list of layer activations for interpretability.
        override_layer_idx : int or None
            If not None, we skip computing layers <= override_layer_idx from x, 
            and we read them from override_activations[l]. 
        override_activations : dict or list or None
            A precomputed store of layer activations you want to use. 
            e.g. override_activations[l] -> shape (num_nodes_in_layer_l, batch_size).

        Returns
        -------
        output : torch.Tensor of shape (batch_size, output_size)
        all_acts (optional) : list or dict of shape (#layers + 1), 
                            each is (num_nodes_in_that_layer, batch_size).
        """

        # We'll keep an internal structure to hold the final activations for each layer
        # a list indexed by layer (0..self.number_of_layers)
        # Each activation has shape = (num_nodes_in_that_layer, batch_size).
        layer_activations = [ None ] * ( self.number_of_layers + 1 )

        # 0) If override_layer_idx is None, we do a normal full forward from x.
        #    Otherwise, for layers < override_layer_idx, we reuse override_activations.

        if override_layer_idx is not None:
            # For each l in [0 .. override_layer_idx], copy from override_activations
            for l in range( override_layer_idx + 1 ):
                layer_activations[ l ] = override_activations[ l ]

            # We start computation from the next layer
            start_layer = override_layer_idx + 1
        else:
            # We start computation from the first (hidden) layer because layer_activations[0] is input
            start_layer = 1
            input_layer_size = len( self.input_nodes )
            assert x is not None, "x must be provided if override_layer_idx is None"
            assert x.shape[ 1 ] == input_layer_size, "Input size mismatch"

            # input is (batch_size, input_layer_size) -> (input_layer_size, batch_size)
            input_tensor = x.t()
            if capture_activations:
                # make it a leaf with grad
                input_tensor = input_tensor.clone().detach().requires_grad_( True )

            layer_activations[ 0 ] = input_tensor

        # 1) Now proceed from 'start_layer' up to self.number_of_layers
        for layer_idx in range( start_layer, self.number_of_layers + 1 ):
            # Build the sum of sparse mm from all parents
            contributions = []
            for prev_layer_idx, values in self.sparse_values[ layer_idx ].items():
                # Build the sparse matrix
                indices = getattr( self, f"_sparse_indices_{layer_idx}_{prev_layer_idx}" )
                sizes = getattr( self, f"_sparse_sizes_{layer_idx}_{prev_layer_idx}" )
                rows, cols = map( int, sizes.tolist() )

                # # ―― Skip empty connections completely ――――――――――――――――――――――――――――――
                # if values.numel() == 0 or rows == 0 or cols == 0:
                #     print("Empty connections!")
                #     continue

                # # Debug asserts (catch device mismatch early):
                # assert indices.device == values.device, "indices/values device mismatch!"
                # assert sizes.device   == values.device, "sizes device mismatch!"

                # if indices[0].max() >= rows or indices[1].max() >= cols:
                #     raise ValueError(
                #         f"OOB index in layer {layer_idx}->{prev_layer_idx}: "
                #         f"max row {indices[0].max().item()}/{rows-1}, "
                #         f"max col {indices[1].max().item()}/{cols-1}"
                #     )

                # cudaIllegalMemory FIX:
                # w = torch.sparse_coo_tensor(
                #     indices,
                #     values,
                #     (rows, cols),
                #     device=self.device
                # ) #.coalesce()

                # ―― Guarantee the RHS is contiguous (workaround for PyTorch bug) ――――――――
                rhs = layer_activations[ prev_layer_idx ] #.contiguous()

                # if __debug__:
                #     nnz = values.numel()
                #     dtype = values.dtype
                #     print(f"[DBG] L{layer_idx}->{prev_layer_idx}: "
                #         f"rows={rows}, cols={cols}, nnz={nnz}, dtype={dtype}, "
                #         f"rhs_contig={rhs.is_contiguous()}, hash_limit={rows*cols < 2_147_483_520}")

                # Multiply the weight matrix with the previous layer's activations
                # layer_activations[prev_layer_idx]: shape (num_nodes_in_prev_layer, batch_size)
                # cudaIllegalMemory FIX:
                # contributions.append(torch.sparse.mm(w, rhs))
                contributions.append( spmm( indices, values, rows, cols, rhs ) )

                # cuda illegal memory access error workaround trial
                # w = w.to_dense()
                # contributions.append(w @ rhs)

            # Sum the contributions -> shape (num_nodes_in_this_layer, batch_size)
            z = torch.sum( torch.stack( contributions ), dim = 0 )

            # Add bias
            bias = self.biases[ str( layer_idx ) ]
            # z is (num_nodes_in_this_layer, batch_size), so we broadcast
            z = z + bias.unsqueeze( 1 )

            # Optionally batchnorm
            if self.use_batchnorm:
                bn = self.batchnorms[ str( layer_idx ) ]
                z = bn( z.t() ).t() # (batch_size, features) -> BN -> transpose

            # Activation
            a = self.activation( z )

            # Dropout
            if self.use_dropout:
                dp = self.dropouts[ str( layer_idx ) ]
                a = dp( a.t() ).t()

            if capture_activations:
                a.retain_grad()

            # Store activations
            layer_activations[ layer_idx ] = a

        outputs = [
            self.layers[ f"layer_{layer_idx}" ]( layer_activations[ layer_idx ].t() )
            for layer_idx in range( self.output_from_layers, self.number_of_layers + 1 )
        ]

        # print( f"COMPUTING OUTPUT:" )
        # for layer_idx in range( 0, self.number_of_layers + 1 ):
        #     print( f"Layer {layer_idx}, Activations: {layer_activations[ layer_idx ].t()}" )

        # print( f"MODEL OUTPUTS: {outputs}")

        # Outputs so far: list of (batch_size, output_size) shaped tensors of length #output_items
        # We average out by #output_items.
        # This only applies if output_method == 'layerwise'. Here, it is not really probabilistically valid,
        # because the average of logits is not the same as averaging of probabilities mapped back to logit space.
        # The problem is that the second choice might be computationally unstable.
        # The main difference between layerwise and nodewise is that in
        # - layerwise: all layers have the same magnitude of contribution, and in
        # - nodewise: all nodes have the same contibution.
        if self.output_method == 'layerwise':
            output = torch.mean( torch.stack( outputs, dim = 0 ), dim = 0 )
        else:
            output = torch.sum( torch.stack( outputs, dim = 0 ), dim = 0 )

        if capture_activations:
            return output, layer_activations
        else:
            return output


    def integrated_gradients_input( self, x, baseline, steps = 50, output_index = None ):
        """
        Input-level Integrated Gradients for a DAG model that outputs (batch_size, output_size).
        We do single-sample interpretability, so x.shape => (1, input_size).

        We return IG of shape (1, input_size).
        """
        diff = x - baseline
        total_grad = torch.zeros_like( x ).to( x.device )

        for alpha in torch.linspace( 0.0, 1.0, steps, device = x.device ):
            x_step = baseline + alpha * diff
            x_step.requires_grad_( True )

            # forward => (batch_size, output_size)
            output = self.forward( x_step ) # shape (1, output_size) if batch_size=1

            # if we have multiple outputs, and we don't specify which one we want, we sum over them
            if output_index is None:
                scalar_out = output.sum()
            else:
                # or if we specify which one we want, we use that index (that class)
                scalar_out = output[ :, output_index ].sum()

            self.zero_grad()
            if x_step.grad is not None:
                x_step.grad.zero_()

            scalar_out.backward()
            # accumulate gradient w.r.t. x_step
            total_grad += x_step.grad.detach()

        ig = diff * total_grad / steps
        return ig

    def integrated_gradients( self,
                              x: torch.Tensor,
                              baseline: torch.Tensor,
                              *,
                              steps: int = 50,
                              output_index: int | None = None,
                              sample_indices: list[ int ] | None = None ):
        """
        Integrated Gradients for every node-activation layer in the DAG model.

        Parameters
        ----------
        x : torch.Tensor
            Actual input, shape (batch_size, input_size).
        baseline : torch.Tensor
            Baseline input (same shape).
        steps : int
            Number of interpolation steps in [0,1].
        output_index : int or None
            If the final output is multi-dimensional, specify which dimension to backprop.
            If None, we sum all outputs to form a scalar.
        sample_indices : list[int] | None, optional
            Indices of `x` inside the **larger** dataset.  Length must equal
            `x.shape[0]`.  If *None* we fall back to `range(batch_size)`.

        Returns
        -------
        G_importance : nx.DiGraph
            Same as before – every node has an ``"importance"`` attribute that is a
            *numpy* array of length ``batch_size``.
        df_importance : pd.DataFrame
            Long-format table with three columns:

            =========  ==========================================
            column     description
            =========  ==========================================
            sample_id  index of the data point in the full dataset
            node_id    node label as used in the graph
            importance Integrated-Gradients value for that pair
            =========  ==========================================

            The index is just a running integer; use ``pivot`` / ``pivot_table`` or
            ``set_index`` if you prefer a wide matrix with one row per *sample*
            and one column per *node*.
        """

        if sample_indices is None:
            sample_indices = list( range( x.shape[ 0 ] ) )
        if len( sample_indices ) != x.shape[ 0 ]:
            raise ValueError( "sample_indices must have the same length as the batch size" )

        # Create a *copy* of the original graph so we do not mutate the template.
        G_importance = nx.DiGraph( self.graph )

        # Storage for the dataframe – one record per (sample,node) pair
        records: list[ dict ] = []

        # --- 1. Forward passes to get activations on baseline and real input ----
        _, baseline_acts = self.forward( baseline, capture_activations = True )
        _, actual_acts = self.forward( x, capture_activations = True )

        # ---- 2. Loop over layers and integrate the gradients ------------------
        for layer_idx in range( self.number_of_layers + 1 ):
            a_base = baseline_acts[ layer_idx ].t() # (B, layer_dim)
            a_actual = actual_acts[ layer_idx ].t() # (B, layer_dim)

            diff = a_actual - a_base
            total_grad = torch.zeros_like( a_actual, device = x.device )

            # Line-integral approximation
            for alpha in torch.linspace( 0.0, 1.0, steps, device = x.device ):
                a_step = ( a_base + alpha * diff ).t().detach().clone().requires_grad_( True )

                # Override activations up to this layer
                override_acts = { l: actual_acts[ l ].detach() for l in range( layer_idx ) }
                override_acts[ layer_idx ] = a_step

                output = self.forward( x = None, capture_activations = False, override_layer_idx = layer_idx, override_activations = override_acts )

                scalar_out = output.sum() if output_index is None else output[ :, output_index ].sum()

                self.zero_grad()
                if a_step.grad is not None:
                    a_step.grad.zero_()

                scalar_out.backward()
                total_grad += a_step.grad.detach().t()

            # Integrated gradients for this layer
            ig_layer = diff * total_grad / steps # (B, layer_dim)

            # Store into graph and dataframe
            current_layer_nodes = [ n for n in self.sorted_nodes if self.node_layers[ n ] == layer_idx ]

            for node_col, node in enumerate( current_layer_nodes ):
                node_importances = ig_layer[ :, node_col ].detach().cpu().numpy()
                # Graph attribute
                G_importance.nodes[ node ][ "importance" ] = node_importances

                # Data-frame records (one row per sample)
                records.extend( {
                    "sample_id": sample_indices[ b ],
                    "node_id": node,
                    "importance": float( node_importances[ b ] )
                } for b in range( len( sample_indices ) ) )

        df_importance = pd.DataFrame.from_records( records, columns = [ "sample_id", "node_id", "importance" ] )
        return G_importance, df_importance
