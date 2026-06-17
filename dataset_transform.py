from model.graph_partition import to_sparse, combine_subgraphs, cal_coarsen_adj
from model.graph_partition_rl import SubgraphsData
from model.graph_partition_rl import metis_subgraph as metis_subgraph_rl

import torch
from pe import random_walk, RWSE, LapPE


class PositionalEncodingTransform(object):
    def __init__(self, rw_dim=0, lap_dim=0):
        super().__init__()
        self.rw_dim = rw_dim
        self.lap_dim = lap_dim

    def __call__(self, data):
        if self.rw_dim > 0:
            data.rw_pos_enc = RWSE(
                data.edge_index, self.rw_dim, data.num_nodes)
        if self.lap_dim > 0:
            data.lap_pos_enc = LapPE(
                data.edge_index, self.lap_dim, data.num_nodes)
        return data


class GraphPartitionTransform(object):
    def __init__(self, static_weights, n_patches, drop_rate=0.0, num_hops=1, is_directed=False, patch_rw_dim=0, patch_num_diff=0, divergence_type='js',
                 lambda_intra=0.6, lambda_balance=0.3,
                 ablate_R_intra=False, ablate_R_balance=False, ablate_R_comm=False,
                 ablate_policy_network=False, ablate_value_network=False):
        super().__init__()
        self.static_weights = static_weights
        self.n_patches = n_patches
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.is_directed = is_directed
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.divergence_type = divergence_type

        self.lambda_intra = lambda_intra
        self.lambda_balance = lambda_balance

        self.ablate_R_intra = ablate_R_intra
        self.ablate_R_balance = ablate_R_balance
        self.ablate_R_comm = ablate_R_comm
        self.ablate_policy_network = ablate_policy_network
        self.ablate_value_network = ablate_value_network
    def _diffuse(self, A):
        if self.patch_num_diff == 0:
            return A
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)
        RW = A * Dinv
        M = RW
        M_power = M

        for _ in range(self.patch_num_diff-1):
            M_power = torch.matmul(M_power, M)
        return M_power

    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})
        node_masks, edge_masks = metis_subgraph_rl(
            data, n_patches=self.n_patches, drop_rate=self.drop_rate, num_hops=self.num_hops,
            is_directed=self.is_directed, static_weights=self.static_weights,
            divergence_type=self.divergence_type, lambda_intra=self.lambda_intra,
            lambda_balance=self.lambda_balance, ablate_R_intra=self.ablate_R_intra,
            ablate_R_balance=self.ablate_R_balance, ablate_R_comm=self.ablate_R_comm,
            ablate_policy_network=self.ablate_policy_network,
            ablate_value_network=self.ablate_value_network)
        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks)
        combined_subgraphs = combine_subgraphs(
            data.edge_index, subgraphs_nodes, subgraphs_edges, num_selected=self.n_patches, num_nodes=data.num_nodes)

        coarsen_adj = cal_coarsen_adj(node_masks)
        coarsen_rows_batch, coarsen_cols_batch = torch.nonzero(
            coarsen_adj, as_tuple=True)
        data.coarsen_edge_attr = coarsen_adj[coarsen_rows_batch,
                                             coarsen_cols_batch]
        data.subgraphs_batch_row = coarsen_rows_batch
        data.subgraphs_batch_col = coarsen_cols_batch
        if self.patch_num_diff > -1 or self.patch_rw_dim > 0:
            if self.patch_rw_dim > 0:
                data.patch_pe = random_walk(coarsen_adj, self.patch_rw_dim)
            if self.patch_num_diff > -1:
                data.coarsen_adj = self._diffuse(coarsen_adj).unsqueeze(0)

        subgraphs_batch = subgraphs_nodes[0]
        mask = torch.zeros(self.n_patches).bool()
        mask[subgraphs_batch] = True
        data.subgraphs_batch = subgraphs_batch
        data.subgraphs_batch_edge = subgraphs_edges[0]
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.mask = mask.unsqueeze(0)

        data.__num_nodes__ = data.num_nodes
        return data
