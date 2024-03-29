import jax
import jax.numpy as np
import flax.linen as nn
import e3nn_jax as e3nn
from models.mlp import MLP

from utils.graph_utils import apply_pbc

from typing import Callable, List, Tuple


def _index_max(i: np.ndarray, x: np.ndarray, out_dim: int) -> np.ndarray:
    return np.zeros((out_dim,) + x.shape[1:], x.dtype).at[i].max(x)


class EquivariantTransformerBlock(nn.Module):
    irreps_node_output: e3nn.Irreps
    list_neurons: Tuple[int, ...]
    act: Callable[[np.ndarray], np.ndarray]
    num_heads: int = 1

    @nn.compact
    def __call__(
        self,
        edge_src: np.ndarray,  # [E] dtype=int32
        edge_dst: np.ndarray,  # [E] dtype=int32
        edge_weight_cutoff: np.ndarray,  # [E] dtype=float
        edge_attr: e3nn.IrrepsArray,  # [E, D] dtype=float
        node_feat: e3nn.IrrepsArray,  # [N, D] dtype=float
    ) -> e3nn.IrrepsArray:
        r"""Equivariant Transformer.

        Args:
            edge_src (array of int32): source index of the edges
            edge_dst (array of int32): destination index of the edges
            edge_weight_cutoff (array of float): cutoff weight for the edges (typically given by ``soft_envelope``)
            edge_attr (e3nn.IrrepsArray): attributes of the edges (typically given by ``spherical_harmonics``)
            node_f (e3nn.IrrepsArray): features of the nodes

        Returns:
            e3nn.IrrepsArray: output features of the nodes
        """

        def f(x, y, filter_ir_out=None, name=None):
            out1 = e3nn.concatenate([x, e3nn.tensor_product(x, y.filter(drop="0e"))]).regroup().filter(keep=filter_ir_out)
            out2 = e3nn.flax.MultiLayerPerceptron(self.list_neurons + (out1.irreps.num_irreps,), self.act, output_activation=False, name=name)(y.filter(keep="0e"))
            return out1 * out2

        edge_key = f(node_feat[edge_src], edge_attr, node_feat.irreps, name="mlp_key")
        edge_logit = e3nn.flax.Linear(f"{self.num_heads}x0e", name="linear_logit")(e3nn.tensor_product(node_feat[edge_dst], edge_key, filter_ir_out="0e")).array  # [E, H]
        node_logit_max = _index_max(edge_dst, edge_logit, node_feat.shape[0])  # [N, H]
        exp = edge_weight_cutoff[:, None] * np.exp(edge_logit - node_logit_max[edge_dst])  # [E, H]
        z = e3nn.scatter_sum(exp, dst=edge_dst, output_size=node_feat.shape[0])  # [N, H]
        z = np.where(z == 0.0, 1.0, z)
        alpha = exp / z[edge_dst]  # [E, H]

        edge_v = f(node_feat[edge_src], edge_attr, self.irreps_node_output, "mlp_val")  # [E, D]
        edge_v = edge_v.mul_to_axis(self.num_heads)  # [E, H, D]
        edge_v = edge_v * np.sqrt(jax.nn.relu(alpha))[:, :, None]  # [E, H, D]
        edge_v = edge_v.axis_to_mul()  # [E, D]

        node_out = e3nn.scatter_sum(edge_v, dst=edge_dst, output_size=node_feat.shape[0])  # [N, D]
        return e3nn.flax.Linear(self.irreps_node_output, name="linear_out")(node_out)  # [N, D]


class EquivariantTransformer(nn.Module):
    irreps_out: e3nn.Irreps
    d_hidden: int = 64
    n_layers: int = 4
    activation: str = "gelu"
    num_heads: int = 1
    mlp_readout_widths: List[int] = (8, 2)
    n_outputs: int = 1

    @nn.compact
    def __call__(
        self,
        positions: e3nn.IrrepsArray,  # [N, 3] dtype=float
        features: e3nn.IrrepsArray,  # [N, D] dtype=float
        senders: np.array,
        receivers: np.array,
        cutoff: float = 10.,
    ):
        r"""Equivariant Transformer.

        Args:
            positions (e3nn.IrrepsArray): positions of the nodes
            features (e3nn.IrrepsArray): features of the nodes
            senders (np.array): graph senders array
            receivers (np.array): graph receivers array
            cutoff (float): cutoff radius

        Returns:
            e3nn.IrrepsArray: output features of the nodes
        """

        list_neurons = self.n_layers * (self.d_hidden,)

        unit_cell = np.array([[1.,0.,0.,],[0.,1.,0.], [0.,0.,1.]])
        vectors = positions[senders] - positions[receivers] 
        vectors = apply_pbc(vectors.array, unit_cell)
        vectors = e3nn.IrrepsArray("1o", vectors)
        dist = np.linalg.norm(vectors.array, axis=1) / cutoff
        
        # check bessel params
        edge_attr = e3nn.concatenate([e3nn.bessel(dist, 4), e3nn.spherical_harmonics(list(range(1, 3 + 1)), vectors, True)])
        edge_weight_cutoff = e3nn.soft_envelope(dist)

        features = EquivariantTransformerBlock(
            irreps_node_output=e3nn.Irreps("1o") + self.irreps_out,
            list_neurons=list_neurons,
            act=getattr(jax.nn, self.activation),
            num_heads=self.num_heads,
        )(senders, receivers, edge_weight_cutoff, edge_attr, features)
        

        displacements, features = features.slice_by_mul[:1], features.slice_by_mul[1:]

        # Aggregate nodes
        agg_nodes = np.mean(features.array, axis=0)

        # Readout prediction
        out = MLP([w * self.d_hidden for w in self.mlp_readout_widths] + [self.n_outputs,])(agg_nodes)

        # Return updated graph
        return out, features