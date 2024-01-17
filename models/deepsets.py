import jax
from typing import Callable, List
import flax.linen as nn
import jax.numpy as jnp
import jraph
from jraph._src import utils

from utils.graph_utils import fourier_features
from models.mlp import MLP


# @jax.vmap
# def update_fn(*args):
#     size = args[0].shape[-1]
#     if len(
#     if args[1] is not None:
#         inputs = jnp.concatenate([args[0], args[1]], axis=1)
#     else:  # If lone node
#         inputs = jnp.concatenate([args[0]], axis=1)
#     return MLP([size, size])(jnp.concatenate(inputs, axis=-1))
def get_node_mlp_updates(d_hidden, n_layers, activation):
    """Get a node MLP update  function

    Args:
        mlp_feature_sizes (int): number of features in the MLP
        name (str, optional): name of the update function. Defaults to None.

    Returns:
        Callable: update function
    """

    def update_fn(
        nodes: jnp.ndarray,
        globals: jnp.ndarray,
    ) -> jnp.ndarray:
        """update node features

        Args:
            nodes (jnp.ndarray): node features
            sent_attributes (jnp.ndarray): attributes sent to neighbors
            received_attributes (jnp.ndarray): attributes received from neighbors
            globals (jnp.ndarray): global features

        Returns:
            jnp.ndarray: updated node features
        """
        inputs = jnp.concatenate([nodes], axis=1)
        return MLP([d_hidden] * n_layers, activation=activation)(inputs)

    return update_fn

def get_global_mlp_updates(d_hidden, n_layers, activation):
    """Get a node MLP update  function

    Args:
        mlp_feature_sizes (int): number of features in the MLP
        name (str, optional): name of the update function. Defaults to None.

    Returns:
        Callable: update function
    """

    def update_fn(
        nodes: jnp.ndarray,
    ) -> jnp.ndarray:
        """update node features

        Args:
            nodes (jnp.ndarray): node features
            sent_attributes (jnp.ndarray): attributes sent to neighbors
            received_attributes (jnp.ndarray): attributes received from neighbors
            globals (jnp.ndarray): global features

        Returns:
            jnp.ndarray: updated node features
        """
        inputs = jnp.concatenate([nodes], axis=1)
        return MLP([d_hidden] * n_layers, activation=activation)(inputs)

    return update_fn



class DeepSets(nn.Module):
    """DeepSets Network"""

    # Attributes for all MLPs
    message_passing_steps: int = 3
    d_hidden: int = 64
    n_layers: int = 3
    activation: str = "gelu"

    message_passing_agg: str = "sum"  # "sum", "mean", "max"
    readout_agg: str = "mean"
    mlp_readout_widths: List[int] = (8, 2)  # Factor of d_hidden for global readout MLPs
    task: str = "graph"  # "graph" or "node"
    readout_only_positions: bool = False  # Graph-level readout only uses positions
    n_outputs: int = 1  # Number of outputs for graph-level readout
    norm: str = "layer"
    get_node_reps: bool = False

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Apply equivariant graph convolutional layers to graph

        Args:
            graphs (jraph.GraphsTuple): Input graph

        Returns:
            jraph.GraphsTuple: Updated graph
        """

        processed_graphs = graphs

        if processed_graphs.globals is not None:
            processed_graphs = processed_graphs._replace(globals=processed_graphs.globals.reshape(1, -1))

        activation = getattr(nn, self.activation)
        
        # Apply message-passing rounds
        for _ in range(self.message_passing_steps):
            update_node_fn = get_node_mlp_updates(self.d_hidden, self.n_layers, activation)
            update_global_fn = get_global_mlp_updates(self.d_hidden, self.n_layers, activation)
            
            graph_net = jraph.DeepSets(
                update_node_fn=update_node_fn,
                update_global_fn=update_global_fn,
                aggregate_nodes_for_globals_fn=jraph.segment_mean,
                )
           
            processed_graphs = graph_net(processed_graphs)
            
            # Optional normalization
            if self.norm == 'layer':
                norm = nn.LayerNorm() #pairnorm
            else:
                norm = Identity()  # No normalization
            processed_graphs = processed_graphs._replace(nodes=norm(processed_graphs.nodes))
        node_reps = processed_graphs.nodes
            
        if self.readout_agg not in ["sum", "mean", "mmax"]:
            raise ValueError(f"Invalid global aggregation function {self.message_passing_agg}")

        readout_agg_fn = getattr(jnp, f"{self.readout_agg}")

        if self.task == "node":
            return processed_graphs, node_reps
        
        elif self.task == "graph":
             # Aggregate residual node features; only use positions, optionally
            if self.readout_only_positions:
                agg_nodes = readout_agg_fn(processed_graphs.nodes[:, :3], axis=0)
            else:
                agg_nodes = readout_agg_fn(processed_graphs.nodes, axis=0)

            # Readout and return
            out = MLP([w * self.d_hidden for w in self.mlp_readout_widths] + [self.n_outputs,])(agg_nodes)  
            return out, node_reps

        else:
            raise ValueError(f"Invalid task {self.task}")