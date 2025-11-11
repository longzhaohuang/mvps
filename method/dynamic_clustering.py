import torch
from torch import nn
import math
from typing import Dict

def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def cluster_dpc_knn(token_dict : Dict, cluster_num : int, k : int, token_mask : torch.Tensor = None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        x = token_dict['x']
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num

def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """
    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    # out_dict['map_size'] = token_dict['map_size']
    # out_dict['init_grid_size'] = token_dict['init_grid_size']
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    return out_dict

class DynamicClustering(nn.Module):
    def __init__(self, sample_ratio = 0.01 , k = 5):
        super().__init__()
        self.sample_ratio = sample_ratio # 对 Patch Token 的下采样率
        self.k            = k            # knn 的点数
        self.score        = nn.Linear(768, 1)

    def forward(self, patch_token, anomaly_map, device, return_dict = False):
        # token_dict 为 TCFormer 中的一个数据类型用于保存 Patch Token 的信息, 其包含了
        # (['x', 'token_num', 'map_size', 'init_grid_size', 'idx_token', 'agg_weight', 'token_score'])

        # token_score  = torch.softmax(anomaly_map, dim=2)[:, :, 1:] # 获取不同 Token 的异常分数， 这一点应该是衡量重要程度的， 有待考证
        token_score = self.score(patch_token)
        token_weight = token_score.exp()
        
        B,N,C = patch_token.shape
        cluster_num  = max(math.ceil(N * self.sample_ratio) , 1)

        token_dict = {
            "x"           : patch_token,
            "token_score" : token_weight,
            "agg_weight"  : patch_token.new_ones(B, N, 1),
            "token_num"   : B,
            "idx_token"   : torch.arange(N)[None, :].repeat(B, 1).to(device)
        }

        idx_cluster, cluster_num = cluster_dpc_knn(token_dict, cluster_num, self.k)
        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)

        if return_dict:
            return down_dict, token_dict
        else:
            return down_dict["x"]
        
if __name__ == '__main__':
    batch_size = 2
    token_num  = 1369
    token_channel = 768
    device = "cuda"

    anomlay_map = torch.randn(batch_size, token_num, 2).to(device)
    patch_token = torch.randn(batch_size, token_num, token_channel).to(device)

    dynamic_clustering = DynamicClustering()
    cluster_center = dynamic_clustering(patch_token, anomlay_map, device)
    print(cluster_center.shape)
