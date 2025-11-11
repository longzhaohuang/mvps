from typing import Union, List, Optional
import numpy as np
import torch
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
import math

from .clip_model import CLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

CLS_PROMPTING_LENGTH = 3
IMG_PROMPTING_LENGTH = 3
DYN_PROMPTING_LENGTH = 3

class ProjectLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_replicas):
        super(ProjectLayer, self).__init__()

        self.head = nn.ModuleList([
            nn.Linear(input_dim, output_dim) 
            for _ in range(num_replicas)
        ])
        self.num_replicas = num_replicas

    def forward(self, tokens):
        out_tokens = []
        for i in range(self.num_replicas):
            # temp = self.head[i](tokens[:,i,:])
            temp = self.head[i](tokens)
            out_tokens.append(temp)
        out_tokens = torch.stack(out_tokens, dim=1)
        return out_tokens

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads = 1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model need be divided by num_heads"

        self.d_heads = d_model // num_heads  # 整除才是整数
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model) 
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.scale = math.sqrt(self.d_heads)

    def forward(self, q, k, v, mask = None):
        # q (batch_size, seq_len, d_model)
        batch_size = q.size(0)
        seq_len = q.size(1)

        Q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.d_heads).transpose(1, 2) # （batch_size, num_heads, seq_len, d_heads)
        K = self.k_proj(k).view(batch_size, -1, self.num_heads, self.d_heads).transpose(1, 2)
        V = self.v_proj(v).view(batch_size, -1, self.num_heads, self.d_heads).transpose(1, 2)

        score = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # （batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            score = score.masked_fill(mask==0, float('-inf'))

        attn_weight = torch.softmax(score, dim=-1)
        out = torch.matmul(attn_weight, V)  # （batch_size, num_heads, seq_len, d_heads)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_heads)
        out = self.o_proj(out)

        return out, attn_weight

class Adapter(nn.Module):
    def __init__(self, text_channel, visual_channel, last_layer : bool = False):
        super(Adapter, self).__init__()

        self.fc1 = nn.Linear(visual_channel, text_channel)
        self.last_layer = last_layer
        if self.last_layer: return 
        self.fc2 = nn.Linear(visual_channel, visual_channel)

        # 对 token 进行选择
        self.token_select_3x3 = nn.Sequential(
            nn.ReplicationPad2d(1),    # 避免空白补 0 所引入的误差
            nn.Conv2d(text_channel, 1, (3,3), (1,1), (0,0)),
            nn.Sigmoid()
        )

        # 对 channel 进行调节
        self.channel_select = nn.Linear(visual_channel, visual_channel)

    def forward(self, ln_x, out_x = None, aux_information=None):
        numtokens, Bs, _ = ln_x.shape
        numtokens = 1370
        feat_w = feat_h  = int((numtokens - 1) ** .5)
        x_stage1    =  self.fc1(ln_x[1:numtokens])

        if self.last_layer: 
            return x_stage1, None

        x_stage1_featmap = x_stage1.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
        # 此处不传播回去，只注入特征:
        token_selection_mask = self.token_select_3x3(x_stage1_featmap.detach())
        
        if out_x is not None:
            x_stage2     = out_x[1:numtokens]
        else :
            x_stage2     = ln_x[1:numtokens]
        
        x_stage2         = F.relu(self.fc2(x_stage2))
        x_stage2_featmap = x_stage2.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
        x_stage2_featmap = x_stage2_featmap * token_selection_mask
        x_stage2         = x_stage2_featmap.reshape(Bs, -1, feat_h * feat_w).permute(2,0,1)
        x_stage2         = self.channel_select(x_stage2)
        return x_stage1, x_stage2

class PromptLayer(nn.Module):
    def __init__(self, channel, length, depth, is_text, prompting_type, enabled=True):
        super(PromptLayer, self).__init__()
        self.channel = channel
        self.length = length
        self.depth = depth
        self.is_text = is_text
        self.enabled = enabled
        self.bs      = 0
        self.attn_mask = None

        self.prompting_type = prompting_type

        if self.enabled: # only when enabled, the parameters should be constructed
            if 'D' in prompting_type: # dynamic prompts
                self.dynamic_prompts = [0.] # place holder
            
            # Conference Version :
            self.scale_prompt = [0.]
            # Expand Version :
            # self.patch_scale_prompt = [0.]  # Patch 需要利用的尺度提示
            # self.cls_scale_prompt = [0.]    # cls   需要利用的尺度提示

            self.prompt_layer_id = 1
            self.prompt_id = 0

    def set_dynamic_prompts(self, dynamic_prompts):
        self.dynamic_prompts = dynamic_prompts

    # Conference Version :
    def set_scale_prompt(self, scale_prompts):
        self.scale_prompt = scale_prompts
    
    def forward_visual(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, prompt_id : int = 0):
        # Expand Paper:
        if self.enabled:
            length = self.length
            if self.prompt_id != prompt_id:
                self.prompt_layer_id = 0
                self.prompt_id = prompt_id

            if indx == 0:
                scale_context = self.scale_prompt[0][0,...].half()
                x = torch.cat([x, scale_context], dim = 0)
            else:
                if self.prompt_layer_id < (self.depth - 1):
                    prefix = x[:x.shape[0] - length, :, :]
                    scale_context = self.scale_prompt[self.prompt_id][self.prompt_layer_id,...].half()
                    x = torch.cat([prefix, scale_context], dim=0)             
                    self.prompt_layer_id += 1
                else:
                    x = x
        else :
            x = x
        
        if (self.attn_mask is None or self.bs != x.shape[1]) and self.enabled is True:
            with torch.no_grad():
                del self.attn_mask
                N = x.shape[0]
                attn_mask = torch.zeros((N, N)).to(x.device)
                attn_mask[0:1, N-IMG_PROMPTING_LENGTH:]                       = float('-inf')
                attn_mask[1:N-(IMG_PROMPTING_LENGTH+CLS_PROMPTING_LENGTH), N-(IMG_PROMPTING_LENGTH+CLS_PROMPTING_LENGTH):N-IMG_PROMPTING_LENGTH] = float('-inf')
                attn_mask[N-(IMG_PROMPTING_LENGTH+CLS_PROMPTING_LENGTH):N-IMG_PROMPTING_LENGTH, N-IMG_PROMPTING_LENGTH:]                         = float('-inf')
                attn_mask[N-IMG_PROMPTING_LENGTH:, N-(IMG_PROMPTING_LENGTH+CLS_PROMPTING_LENGTH):N-IMG_PROMPTING_LENGTH]                         = float('-inf')
                attn_mask.requires_grad = False
                self.bs = x.shape[1]
            self.attn_mask = attn_mask        
        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x=v_x, attn_mask=self.attn_mask)

        if self.enabled:
            tokens = x[:x.shape[0] - length, : , :]
            return x, tokens, attn_tmp
        else:
            tokens = x
            return x, tokens, attn_tmp

    def forward(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, prompt_id : int = 0):
        if not self.is_text:
            return self.forward_visual(resblock, indx, x, k_x, v_x, attn_mask, prompt_id)

class MutilPrompt_TextEmbebddingLayer(nn.Module):
    def __init__(self, fixed, length):
        super(MutilPrompt_TextEmbebddingLayer, self).__init__()
        self.tokenizer = _Tokenizer()
        self.ensemble_text_features = {}
        self.fixed = fixed
        # 修改为简要的提示
        self.prompt_normal = ['normal {}']
        self.prompt_abnormal = ['abnormal {}' ]
        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        self.prompt_templates = ['{}']
        self.prompt_placeholder = " ".join(["X"] * length) # 此处进行了修改，另外一半的提示用来储存细粒度提示 ， 会议版本只有 1 倍

        self.prompt_normal_length = len(self.prompt_normal) * len(self.prompt_templates)
        self.prompt_abnormal_length = len(self.prompt_abnormal) * len(self.prompt_templates)

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False, placeholder_setting = True) -> Union[
        torch.IntTensor, torch.LongTensor]:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        # 占位符用于后续的软提示替换
        if placeholder_setting:
            all_tokens = [[sot_token] + self.tokenizer.encode(self.prompt_placeholder + ' ') + self.tokenizer.encode(text) + [eot_token] for text in texts] 
        else :
            all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts] 
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def encode_text(self, model, texts, device):
        normal_sentence = []
        abnoraml_sentence = []
        for i in range(len(self.prompt_state)):
            for text in texts:
                text = text.replace('_', ' ')
                prompted_state = [state.format(text) for state in self.prompt_state[i]]
                for s in prompted_state:
                    for template in self.prompt_templates:
                        if i == 0:
                            normal_sentence.append(template.format(s))
                        else : 
                            abnoraml_sentence.append(template.format(s))
        
        normal_sentence = self.tokenize(normal_sentence, context_length=77).to(device)
        abnoraml_sentence = self.tokenize(abnoraml_sentence, context_length=77).to(device)
        
        normal_class_embeddings = model.encode_text(normal_sentence, 0)
        abnormal_class_embeddings = model.encode_text(abnoraml_sentence, 1)
        
        class_embeddings = [normal_class_embeddings, abnormal_class_embeddings]
        
        if len(self.prompt_state) != 2:
            raise RuntimeError("len(self.prompt_state) must equal to 2")

        text_features = [torch.stack([i, j], dim=1) for i , j in zip(class_embeddings[0], class_embeddings[1])] # -> PN * [B, N, C]

        return text_features

    def forward(self, model, texts, device):
        text_features = self.encode_text(model, texts, device)
        for prompt_id in range(len(text_features)):
            text_features[prompt_id] = F.normalize(text_features[prompt_id], dim=-1)
            text_features[prompt_id] = text_features[prompt_id].permute(0, 2, 1)
        return text_features

class MutilPrompt_TextPromptLayer(nn.Module):
    def __init__(self, channel, length, prompt_depth, prompt_num, prompting_type, enabled=True):
        super(MutilPrompt_TextPromptLayer, self).__init__()

        self.channel = channel
        self.length = length

        self.prompt_num = prompt_num
        self.prompt_depth = prompt_depth
        self.enabled = enabled
        self.prompting_type = prompting_type

        if self.enabled:
            self.learnable_prompt = nn.Parameter(torch.empty(2, self.prompt_num, IMG_PROMPTING_LENGTH, channel))
            for i in range(2):
                for j in range(self.prompt_num):
                    nn.init.normal_(self.learnable_prompt[i, j], std=0.02)

            if 'S' in prompting_type:
                self.static_prompts = nn.Parameter(torch.empty(self.prompt_depth - 1, self.prompt_num, self.length, channel))
                for i in range(self.prompt_depth - 1):
                    for j in range(self.prompt_num):
                        nn.init.normal_(self.static_prompts[i, j], std=0.02)

            self.dynamic_prompts = None    
            self.frist_dynamic_prompts = None
            self.memory_prompt = [] # 记录提示 

    def set_dynamic_prompts(self, dynamic_prompts):
        self.dynamic_prompts = dynamic_prompts
    
    def set_first_dynamic_prompts(self, dynamic_prompts):
        self.frist_dynamic_prompts = dynamic_prompts

    def reset_memory_prompt(self, ):
        self.memory_prompt = []

    def forward(self, resblock, indx, x = None, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, state_idx = 0):
        if self.enabled:
            if 0 < indx < self.prompt_depth:
                # 此处沿用 AdaCLIP 去得到实例感知的可学习提示
                if 'S' in self.prompting_type and 'D' in self.prompting_type:                     # both
                    B = x.shape[1] // (self.prompt_num * 2)
                    static_prompts  = self.static_prompts[indx-1]                                   # [PN, N, C]
                    textual_context  = static_prompts.repeat(B * 2, 1, 1)                         # [B * PN, N , C]
                    if self.dynamic_prompts is not None:
                        textual_context = textual_context + self.dynamic_prompts
                else:
                    print('You should at least choose one type of prompts when the prompting branches are not none.')
                    raise NotImplementedError

            if indx == 0:
                prefix = x[:1, :, :]
                suffix = x[1 + IMG_PROMPTING_LENGTH + DYN_PROMPTING_LENGTH:, :, :]
                _ , BPN, C  = x.shape
                B = BPN // (self.prompt_num * 2)
                static_prompts  = self.learnable_prompt[state_idx]                              # [PN, N, C]
                learnable_prompt = static_prompts.repeat(B * 2, 1, 1).permute(1, 0, 2).half()
                if self.frist_dynamic_prompts is not None or DYN_PROMPTING_LENGTH != 0: 
                    dynamic_prompt = self.frist_dynamic_prompts[state_idx].permute(1, 0, 2).half()
                    # learnable_prompt = learnable_prompt + dynamic_prompt
                    learnable_prompt = torch.cat([learnable_prompt, dynamic_prompt], dim=0)
                x = torch.cat([prefix, learnable_prompt, suffix], dim=0)
            elif indx < self.prompt_depth:
                prefix = x[:1, :, :]
                suffix = x[1 + self.length:, :, :]
                textual_context = textual_context.permute(1, 0, 2).half()
                x = torch.cat([prefix, textual_context, suffix], dim=0)
                self.memory_prompt.append(textual_context)
            else : 
                x = x
        else:
            x = x

        if resblock is not None:
            x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask)
            return x, attn_tmp
        else:
            raise RuntimeError("Must input resblock")

class MVPCL(nn.Module):
    def __init__(self, k_clusters, prompt_num, deta_k):
        super(MVPCL, self).__init__()
        # 聚类算法此时使用了多尺度聚类，同时聚类中心数量会如下公式进行递减
        # self.k_clusters - deta_k * prompt_id
        self.k_clusters = k_clusters
        self.prompt_num = prompt_num
        
        self.n_aggregate_patch_tokens = [(self.k_clusters - deta_k * prompt_id) * 5 for prompt_id in range(self.prompt_num)]
        self.cluster_performer = [KMeans(n_clusters = self.k_clusters - deta_k * prompt_id, n_init="auto") for prompt_id in range(self.prompt_num)]

        if self.n_aggregate_patch_tokens[-1] <= 0:
            raise RuntimeError("reset these paramter (k_cluseter, group_cnt, deta_k)")

    def forward(self, patch_token: torch.Tensor, anomaly_map: torch.Tensor, prompt_id: int):
        anomaly_map = torch.softmax(anomaly_map, dim=2)[:, :, 1] # B, L

        # extract most abnormal feats
        k = min(anomaly_map.shape[1], self.n_aggregate_patch_tokens[prompt_id])
        top_k_indices = torch.topk(anomaly_map, k=k, dim=1).indices
        
        selected_tokens = patch_token.gather(dim=1, index=top_k_indices.unsqueeze(-1).expand(-1, -1, patch_token.shape[-1]))

        batch_cluster_centers = []

        # use kmeans to extract these centriods
        # Perform K-Means clustering
        for b in range(selected_tokens.shape[0]):
            cluster_labels = self.cluster_performer[prompt_id].fit_predict(selected_tokens[b, :, :].detach().cpu().numpy())

            # Initialize a list to store the cluster centers
            cluster_centers = []

            # Extract cluster centers for each cluster
            for cluster_id in range(self.cluster_performer[prompt_id].n_clusters):
                collected_cluster_data = selected_tokens[b, :, :][cluster_labels == cluster_id]
                cluster_center = torch.mean(collected_cluster_data, dim=0, keepdim=True)
                cluster_centers.append(cluster_center)

            # Normalize the cluster centers
            cluster_centers = torch.cat(cluster_centers, dim=0)
            batch_cluster_centers.append(cluster_centers)

        batch_cluster_centers = torch.stack(batch_cluster_centers, dim=0) # [B, K, D]
        batch_cluster_centers = F.normalize(batch_cluster_centers, dim=-1) # [B, K, D]
        return batch_cluster_centers

# Conference Paper : 
# class HMPT(nn.Module):
#     def __init__(self, text_channel=768, visual_channel=1024, num_proto=DYN_PROMPTING_LENGTH):
#         super(HMPT, self).__init__()
        
#         self.up_scale_adapter        = nn.Linear(text_channel, visual_channel)
#         self.down_scale_adapter      = nn.Linear(visual_channel, text_channel)
#         self.proto_normal_linear     = ProjectLayer(visual_channel, text_channel, num_proto)
#         self.proto_abnormal_linear   = ProjectLayer(visual_channel, text_channel, num_proto)

#         self.proto_normal_global     = nn.Parameter(torch.empty(1, num_proto, text_channel))
#         self.proto_abnormal_global   = nn.Parameter(torch.empty(1, num_proto, text_channel))
#         for i in range(num_proto):
#             nn.init.normal_(self.proto_normal_global[:, i, :], std=0.02)
#             nn.init.normal_(self.proto_abnormal_global[:, i, :], std=0.02)

#         # 低->高 跨尺度注意力
#         self.low2high_n = CrossAttention(
#             d_model=visual_channel,
#             num_heads=1
#         )
#         self.low2high_ab = CrossAttention(
#             d_model=visual_channel,
#             num_heads=1
#         )

#         self.prompt_num    = DYN_PROMPTING_LENGTH
#         self.prompt_normal = self.prompt_abnormal = None
#         self.patch_token = None

#     def forward(self, patch_tokens, image_feature, patch_token_id):
#         B = patch_tokens[-1].size(0)
#         global_info           = image_feature.unsqueeze(1)
#         scale_global_info     = self.up_scale_adapter(global_info)        
#         low_feat       = patch_tokens[patch_token_id][:, 1:, :]
#         low_feat       = torch.cat([scale_global_info, low_feat], dim=1)
#         normal_proto   = self.low2high_n(q=scale_global_info, k=low_feat, v=low_feat)[0]  + scale_global_info
#         abnormal_proto = self.low2high_ab(q=scale_global_info, k=low_feat, v=low_feat)[0] + scale_global_info
        
#         self.prompt_normal   = normal_proto
#         self.prompt_abnormal = abnormal_proto
#         self.patch_token     = low_feat

#         return F.normalize(self.proto_normal_linear(normal_proto.squeeze(1)), dim=-1)   + self.proto_normal_global.repeat(B, 1, 1),      \
#                F.normalize(self.proto_abnormal_linear(abnormal_proto.squeeze(1)), dim=-1) + self.proto_abnormal_global.repeat(B, 1, 1),  \
#                F.normalize(self.proto_normal_linear(scale_global_info.squeeze(1)), dim=-1) + self.proto_normal_global.repeat(B, 1, 1),   \
#                F.normalize(self.proto_abnormal_linear(scale_global_info.squeeze(1)), dim=-1) + self.proto_abnormal_global.repeat(B, 1, 1)

# Conference Paper : 
class HMPT(nn.Module):
    def __init__(self, text_channel=768, visual_channel=1024, num_proto=DYN_PROMPTING_LENGTH):
        super(HMPT, self).__init__()
        
        self.proto_normal_global     = nn.Parameter(torch.empty(1, num_proto, text_channel))
        self.proto_abnormal_global   = nn.Parameter(torch.empty(1, num_proto, text_channel))
        for i in range(num_proto):
            nn.init.normal_(self.proto_normal_global[:, i, :], std=0.02)
            nn.init.normal_(self.proto_abnormal_global[:, i, :], std=0.02)
        
        self.proto_normal_linear     = ProjectLayer(text_channel, visual_channel, num_proto)
        self.proto_abnormal_linear   = ProjectLayer(text_channel, visual_channel, num_proto)
        self.low2high_n = CrossAttention(
            d_model=visual_channel,
            num_heads=1
        )
        self.low2high_ab = CrossAttention(
            d_model=visual_channel,
            num_heads=1
        )
        self.down_scale_adapter_normal = nn.Linear(visual_channel, text_channel)
        self.down_scale_adapter_abnormal = nn.Linear(visual_channel, text_channel)

        self.prompt_num    = DYN_PROMPTING_LENGTH
        self.prompt_normal = self.prompt_abnormal = None
        self.patch_token = None
        
    def forward(self, patch_tokens, image_feature, patch_token_id):
        B = patch_tokens[-1].size(0)
        # STEP.0 : 先生成一组原型 Proto ， 这组原型 Proto 使用 Linear 生成吗 ?
        global_info           = image_feature
        proto_normal          = self.proto_normal_linear(global_info)
        proto_abnormal        = self.proto_abnormal_linear(global_info)
        # STEP.1 : 然后使用 Cross-Attention 让原型获取不同阶段的细粒度信息 (为了更加稳定的进行监督让proto_normal/proto_abnormal信息无法传递到proto_normal_linear/proto_abnormal_linear当中)
        low_feat              = patch_tokens[patch_token_id][:, 1:, :]
        normal_proto_fg       = self.low2high_n(q=proto_normal, k=low_feat, v=low_feat)[0]
        abnormal_proto_fg     = self.low2high_ab(q=proto_abnormal, k=low_feat, v=low_feat)[0]
        # STEP.2 : 使用一个损失函数去监督得到的细粒度原型
        self.prompt_normal    = normal_proto_fg
        self.prompt_abnormal  = abnormal_proto_fg
        self.patch_token      = low_feat
        # STEP.3 : 对获得的所有原型进行降维
        proto_normal   = self.down_scale_adapter_normal(proto_normal)
        proto_abnormal = self.down_scale_adapter_abnormal(proto_abnormal)
        normal_proto_fg = self.down_scale_adapter_normal(normal_proto_fg)
        abnormal_proto_fg = self.down_scale_adapter_abnormal(abnormal_proto_fg)
        # STEP.4 : 返回所有值
        return F.normalize(normal_proto_fg, dim=-1)   + self.proto_normal_global.repeat(B, 1, 1),      \
               F.normalize(abnormal_proto_fg, dim=-1) + self.proto_abnormal_global.repeat(B, 1, 1),  \
               F.normalize(proto_normal, dim=-1) + self.proto_normal_global.repeat(B, 1, 1),   \
               F.normalize(proto_abnormal, dim=-1) + self.proto_abnormal_global.repeat(B, 1, 1)

class MVPS(nn.Module):
    def __init__(
        self,
        freeze_clip : CLIP,
        text_channel : int , 
        visual_channel: int,
        prompting_length: int,
        prompting_depth: int,
        prompting_branch: str, 
        prompting_type: str,
        use_hsf: bool,
        k_clusters: int,
        output_layers: list,
        device: str,
        image_size: int,
        alpha : float,
    ) -> None:
        super(MVPS, self).__init__()

        # 提示的数量需要和 output_layer 去保持一致
        self.prompt_num    = len(output_layers)
        self.output_layers = output_layers
        self.freeze_clip   = freeze_clip
        self.visual        = self.freeze_clip.visual
        self.transformer   = self.freeze_clip.transformer
        self.token_embedding      = self.freeze_clip.token_embedding
        self.positional_embedding = self.freeze_clip.positional_embedding
        self.ln_final             = self.freeze_clip.ln_final
        self.text_projection      = self.freeze_clip.text_projection
        self.attn_mask            = self.freeze_clip.attn_mask
        self.visual_channel       = visual_channel
        self.text_channel         = text_channel

        # 设置提示微调参数
        self.prompting_branch = prompting_branch
        self.prompting_type   = prompting_type
        self.prompting_depth  = prompting_depth
        self.prompting_length = prompting_length
        self.cls_prompting_length = CLS_PROMPTING_LENGTH
        self.img_prompting_length = IMG_PROMPTING_LENGTH
        self.use_hsf          = use_hsf
        self.k_clusters       = k_clusters

        if 'L' in self.prompting_branch:
            self.enable_text_prompt = True
        else : 
            self.enable_text_prompt = False

        if 'V' in self.prompting_branch:
            self.enable_visual_prompt = True
        else:
            self.enable_visual_prompt = False

        self.text_embedding_layer = MutilPrompt_TextEmbebddingLayer(
            fixed=(not self.enable_text_prompt),
            length=CLS_PROMPTING_LENGTH+DYN_PROMPTING_LENGTH
        )
        
        self.text_prompter = MutilPrompt_TextPromptLayer(
            text_channel,
            CLS_PROMPTING_LENGTH+IMG_PROMPTING_LENGTH,
            prompt_depth  =prompting_depth,
            prompting_type=prompting_type,
            enabled       =self.enable_text_prompt,
            prompt_num    =self.prompt_num
        )
        self.visual_prompter = PromptLayer(
            visual_channel, 
            prompting_length, 
            prompting_depth, 
            is_text=False,
            prompting_type=prompting_type,
            enabled=self.enable_visual_prompt,
        )

        # For location:
        self.patch_token_layer = nn.ModuleList(
            [
                *[Adapter(text_channel, visual_channel, last_layer=False) for _ in range(3)],
                *[Adapter(text_channel, visual_channel, last_layer=True)  for _ in range(5)],
            ] 
        )
        # For detection:
        self.cls_token_layer = nn.ModuleList(
            [nn.Linear(visual_channel, text_channel) for _ in range(1)]
        )

        # 用于对齐视觉和文本提示, 级联优化.
        self.DPTO = nn.ModuleList()
        for _ in range(self.prompt_num * 2):
            self.DPTO.append(nn.Linear(text_channel, visual_channel))
        
        # 多视角提示学习
        self.mvpcl = MVPCL(prompt_num=self.prompt_num, k_clusters=10, deta_k=2)
        
        # 层级特征生成
        self.hmpt = nn.ModuleList()
        for _ in range(self.prompt_num):
            self.hmpt.append(
                HMPT(text_channel, visual_channel)
            )

        self.image_size = image_size
        self.device = device
        self.alpha = 0.5
        
        # 一些可视化的变量
        self.qk_token_visualize = []
        self.vv_tokne_visualize = []
        self.fusion_token_visualize = None
        self.qk_attn_map_visualize = []
        self.vv_attn_map_visualize = []

    def empty_visualize_cache(self):
        del self.qk_token_visualize
        del self.vv_tokne_visualize
        del self.qk_attn_map_visualize
        del self.vv_attn_map_visualize 
        self.qk_token_visualize = []
        self.vv_tokne_visualize = []
        self.fusion_token_visualize = None
        self.qk_attn_map_visualize = []
        self.vv_attn_map_visualize = []

    def generate_and_set_dynamic_promtps(self, image, texts):
        with torch.no_grad():
            # extract image features
            image_features , patch_tokens = self.visual.forward(image, self.output_layers, single_output=False)
        
        B = image_features[-1].shape[0]
        text_prompt_fg_normal = []
        text_prompt_fg_abnormal = []
        text_prompt_gl_normal = []
        text_prompt_gl_abnormal = []
        for patch_token_id in range(self.prompt_num):
            text_prompt = self.hmpt[patch_token_id].forward(patch_tokens, image_features[-1], patch_token_id)
            text_prompt_fg_normal.append(text_prompt[0])
            text_prompt_fg_abnormal.append(text_prompt[1])
            text_prompt_gl_normal.append(text_prompt[2])
            text_prompt_gl_abnormal.append(text_prompt[3])

        text_prompt_gl_normal = torch.cat(text_prompt_gl_normal, dim=0)        
        text_prompt_fg_normal = torch.cat(text_prompt_fg_normal, dim=0)
        text_prompt_fg_abnormal = torch.cat(text_prompt_fg_abnormal, dim=0)
        text_prompt_gl_abnormal = torch.cat(text_prompt_gl_abnormal, dim=0)

        normal_list = []
        abnormal_list = []
        for i in range(B):
            normal_list.append(text_prompt_fg_normal[i::B, ...])
            normal_list.append(text_prompt_gl_normal[i::B, ...])
            abnormal_list.append(text_prompt_fg_abnormal[i::B, ...])
            abnormal_list.append(text_prompt_gl_abnormal[i::B, ...])
            
        text_prompt_normal   = torch.cat(normal_list, dim=0)
        text_prompt_abnormal = torch.cat(abnormal_list, dim=0)

        self.text_prompter.set_first_dynamic_prompts([text_prompt_normal, text_prompt_abnormal])

        return image_features, patch_tokens, None
    
    def encode_text(self, text, idx):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        _, B, _ = x.shape

        x = torch.repeat_interleave(x, repeats=self.prompt_num * 2, dim=1)
        text_max_id = torch.repeat_interleave(text.argmax(dim=-1), repeats=self.prompt_num * 2, dim=0)
        for indx, r in enumerate(self.transformer.resblocks):
            x, attn_tmp = self.text_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=self.attn_mask, state_idx=idx)
        prompt_set = x.permute(1, 0, 2)
        prompt_set = self.ln_final(prompt_set)
        prompt_set = prompt_set[torch.arange(prompt_set.shape[0]), text_max_id] @ self.text_projection            # [BPN, C]
        prompt_set_batch = []
        for batch_id in range(B):
            prompt_set_per_batch = prompt_set[batch_id*(self.prompt_num * 2):(batch_id+1)*(self.prompt_num * 2), ...] # [PN,C]
            prompt_set_batch.append(prompt_set_per_batch)
        prompt_set_batch = torch.stack(prompt_set_batch, dim=1)                                                   # [PN, B, C]
        return prompt_set_batch

    def custom_attn(self, attn_layer, x, attn_mask=None, mid_simi=None, attn_type="fused-attn"):
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        v_weight = attn_layer.in_proj_weight[2*embed_dim:3*embed_dim, :]
        v_bias = attn_layer.in_proj_bias[2*embed_dim:3*embed_dim]
        v = F.linear(x, v_weight, v_bias)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if attn_type == "fused-attn" and attn_mask is not None: 
            with torch.no_grad():
                attn_mask /= torch.sum(attn_mask, dim=-2, keepdim=True)
                attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)
                attn_mask = (attn_mask + attn_mask.transpose(-2, -1)) / 2
                attn_mask -= (attn_mask.mean(-1, keepdim=True) * 0.5)
                attn_mask = torch.clamp(attn_mask, 0)
                attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)
                attn_mask = attn_mask.flatten(0, 1)
                attn_weights = torch.repeat_interleave(attn_mask, dim=0, repeats=v.shape[0] // attn_mask.shape[0])
            attn_output     = torch.bmm(attn_weights, v)
            attn_output     = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
            return attn_output, attn_weights
        elif attn_type == "v-v":
            attn_weights = torch.bmm(v * scale, v.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
            return attn_output, attn_weights
        else:
            raise NotImplementedError("NO ATTN TYPE HAS BEEN IMPLEMNT")

    def encode_image(self, image, ori_patch_tokens):
        x = image
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1],
                          self.visual.grid_size[0],
                          self.visual.patch_size[0],
                          self.visual.grid_size[1],
                          self.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.visual.grid_size[0] * self.visual.grid_size[1], -1)
            x = self.visual.patchnorm_pre_ln(x)
            x = self.visual.conv1(x)
        else:
            x = self.visual.conv1(x)                   # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)                     # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)

        x = x + self.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        proj_patch_tokens  = []   # vv-token 用于分割
        proj_cls_tokens    = []
        semantic_tokens = []   # qk-token 用于语义信息
        prompt_id = 0

        # Conference paper version :
        # for indx, r in enumerate(self.visual.transformer.resblocks):
        #     if (indx + 1) in self.output_layers:
        #         x, tokens, attn_tmp = \
        #             self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None, prompt_id=prompt_id) # 在推理阶段不使用对角注意力机制
        #         N = tokens.shape[0]

        #         # For Segmentation : 
        #         seg_adapt_med = self.patch_token_layer[prompt_id](tokens + ori_patch_tokens[prompt_id].permute(1, 0 ,2))
        #         proj_patch_tokens.append(seg_adapt_med[1:,:,:].permute(1,0,2))
        #         # For Detection:
        #         pooled = self.cls_token_layer[prompt_id](tokens[0,:,:])
        #         proj_cls_tokens.append(pooled)

        #         # Combine the knowledge of Segmentation and Detection          
        #         prompt_id += 1
        #     else : 
        #         x, tokens, attn_tmp = \
        #             self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None, prompt_id=prompt_id)

        # Expand paper version : Building VV Attention Pattern
        num_tokens, batch_size, embed_dim = x.shape
        img_features = torch.zeros([len(self.output_layers)] + [num_tokens, batch_size, embed_dim], device=x.device, dtype=x.dtype)    # 不加入可训练参数
        prompt_id = 0
        fusion_attn_map = 0
        x_for_vv  = None
        for indx, r in enumerate(self.visual.transformer.resblocks):
            if (indx + 1) in self.output_layers:
                img_features[prompt_id] = tokens.clone()
                x , tokens , attn_tmp = \
                    self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None, prompt_id=prompt_id)        
                prompt_id += 1
            else:
                if prompt_id >= self.prompt_num: break
                x , tokens , attn_tmp = \
                    self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None, prompt_id=prompt_id)                
            with torch.no_grad():              
                fusion_attn_map = fusion_attn_map + attn_tmp[:,:,:num_tokens,:num_tokens].detach()              
        
        proj_cls_tokens.append(self.cls_token_layer[-1](tokens[0,:,:]))

        alpha = 0.5
        prompt_id = 0
        shallow_information = None
        for indx, r in enumerate(self.visual.transformer.resblocks):
            if (indx + 1) in self.output_layers:
                x_for_vv = img_features[prompt_id].clone()
                if shallow_information is not None:
                    x_for_vv[1:num_tokens] = 0.5 * x_for_vv[1:num_tokens] + 0.5 * shallow_information
                ln_x, attn = self.custom_attn(r.attn, r.ln_1(x_for_vv), attn_mask=None, attn_type="v-v")
                ln_x  = alpha * F.normalize(ln_x, dim=-1) + (1 - alpha) * F.normalize(ori_patch_tokens[prompt_id].permute(1, 0 ,2), dim=-1)
                local_token, shallow_information = self.patch_token_layer[prompt_id](ln_x) 
                proj_patch_tokens.append(local_token.permute(1, 0, 2))
                prompt_id += 1

                # ============ visualize ============ 
                with torch.no_grad():
                    B = ln_x.shape[1]
                    _,N,_ = attn.shape
                    attn_map_mean = attn.detach().view(B, -1, N, N)
                    attn_map_mean = attn_map_mean.mean(dim=1)
                    self.vv_attn_map_visualize.append(attn_map_mean.detach())  
                # ============ visualize ============

        model_r   = self.visual.transformer.resblocks[-1]
        prompt_id = 0
        for indx, r in enumerate(self.visual.transformer.resblocks):
            if (indx + 1) in self.output_layers:
                x_for_fusion = img_features[prompt_id].clone()
                ln_x, attn = self.custom_attn(model_r.attn, model_r.ln_1(x_for_fusion), attn_mask=fusion_attn_map, mid_simi=None, attn_type="fused-attn")
                ln_x  = alpha * F.normalize(ln_x, dim=-1) + (1 - alpha) * F.normalize(ori_patch_tokens[self.prompt_num + prompt_id].permute(1, 0 ,2), dim=-1)
                local_token, shallow_information = self.patch_token_layer[self.prompt_num + prompt_id](ln_x)
                semantic_tokens.append(local_token.permute(1, 0, 2))
                prompt_id += 1

                # ============ visualize ============ 
                with torch.no_grad():
                    B = ln_x.shape[1]
                    _,N,_ = attn.shape
                    attn_map_mean = attn.detach().view(B, -1, N, N)
                    attn_map_mean = attn_map_mean.mean(dim=1)
                    self.qk_attn_map_visualize.append(attn_map_mean.detach())  
                # ============ visualize ============

        return proj_cls_tokens , proj_patch_tokens, semantic_tokens

    def extract_feat(self, image, cls_name):
        _ , ori_patch_token, _ = self.generate_and_set_dynamic_promtps(image, None) # generate and set dynamic prompts for corresponding prompters
        if self.enable_text_prompt:
            text_features = self.text_embedding_layer(self, cls_name, self.device)
        else:
            with torch.no_grad():
                text_features = self.text_embedding_layer(self, cls_name, self.device)
        
        # PD * [TN, B * PN, C] -> [PN / 2, B, TN * 2, C] :
        text_scale_prompt = self.text_prompter.memory_prompt
        if len(text_scale_prompt) // 2 != self.prompting_depth - 1:
            raise RuntimeError("TEXT SCALE PROMPT LIST LENGTH WRONG")
        text_scale_prompt = text_scale_prompt[:len(text_scale_prompt) // 2]
        for idx in range(len(text_scale_prompt)):
            text_scale_prompt[idx] = text_scale_prompt[idx].permute(1 , 0 , 2) # PD * [TN, B * PN, C]  -> PD * [B * PN, TN , C]
        BPN , _ ,  _ =  text_scale_prompt[0].shape
        B = BPN // (self.prompt_num * 2)

        new_text_scale_prompt_pd = []
        for pd_id in range(self.prompting_depth - 1):
            new_text_scale_prompt_b = []
            for batch_id in range(B):
                text_scale_prompt_ = text_scale_prompt[pd_id][batch_id * self.prompt_num * 2:(batch_id + 1) * self.prompt_num * 2, ...] # [PN, TN , C]
                text_scale_prompt_ = (text_scale_prompt_[:self.prompt_num, ...] + text_scale_prompt_[self.prompt_num:, ...]) / 2
                new_text_scale_prompt_b.append(text_scale_prompt_)
            new_text_scale_prompt_b = torch.stack(new_text_scale_prompt_b, dim = 2)                                             # [PN, TN, B , C]
            new_text_scale_prompt_pd.append(new_text_scale_prompt_b)
        new_text_scale_prompt_pd = torch.stack(new_text_scale_prompt_pd, dim=1)                                                 # [PN, PD, TN, B , C]                                      

        visual_scale_prompt = []

        for prompt_id in range(self.prompt_num):
            visual_scale_prompt.append(
                torch.cat(
                    [
                        self.DPTO[prompt_id](
                            new_text_scale_prompt_pd[prompt_id, :, :IMG_PROMPTING_LENGTH, ...]
                        ), 
                        self.DPTO[self.prompt_num + prompt_id](
                            new_text_scale_prompt_pd[prompt_id, :, IMG_PROMPTING_LENGTH:, ...]
                        )
                    ], dim=1
                )
            )

        self.visual_prompter.set_scale_prompt(
            visual_scale_prompt
        )    
    
        if self.enable_visual_prompt:
            proj_cls_tokens, proj_patch_tokens, semantic_map_token = self.encode_image(image, ori_patch_token)
        else:
            with torch.no_grad():
                proj_cls_tokens, proj_patch_tokens, semantic_map_token = self.encode_image(image)
        
        for idx in range(len(proj_cls_tokens)):
            proj_cls_tokens[idx] = proj_cls_tokens[idx] / proj_cls_tokens[idx].norm(dim=-1, keepdim=True)
        
        # 清空缓存
        self.text_prompter.reset_memory_prompt()

        return proj_cls_tokens, proj_patch_tokens, semantic_map_token, text_features

    def visual_text_similarity(self, image_feature, patch_token, text_feature, aggregation):
        vv_tokens, semantic_tokens = patch_token        

        fusion_tokens = []
        anomaly_maps  = []
        for prompt_id in range(self.prompt_num):
            fusion_token = semantic_tokens[prompt_id] + vv_tokens[prompt_id]
            fusion_token = F.normalize(fusion_token , dim=-1)
            anomaly_map  = 100 * fusion_token @ text_feature[prompt_id]
            fusion_tokens.append(fusion_token)
            anomaly_maps.append(anomaly_map)

        anomaly_scores           = []
        anomaly_scores_global    = []
        anomaly_scores_finegrain = []
        cls_alpha = 0.5
        for prompt_id in range(self.prompt_num):  
            clustered_feature     = \
                self.mvpcl.forward(
                    fusion_tokens[prompt_id],
                    anomaly_maps[prompt_id], 
                    prompt_id
                ) # [B,K,D]            
            K                     = clustered_feature.shape[1]
            fusion_feature        = cls_alpha * clustered_feature + cls_alpha * image_feature[-1].unsqueeze(1).repeat(1,K,1)
            anomaly_score_global  = 100 * fusion_feature @  text_feature[self.prompt_num + prompt_id]   # [B,K,2]
            anomaly_score_global  = torch.mean(anomaly_score_global, dim=1)                               # [B,2]
            anomaly_score_global  = torch.softmax(anomaly_score_global, dim=-1)
            anomaly_scores_global.append(anomaly_score_global)
            anomaly_score_finegrain = 100 * clustered_feature @  text_feature[prompt_id]   # [B,K,2]
            anomaly_score_finegrain = torch.softmax(anomaly_score_finegrain, dim=-1)
            max_idx                 = torch.argmax(anomaly_score_finegrain[:, :, 1], dim=1)
            anomaly_score_finegrain = anomaly_score_finegrain[torch.arange(anomaly_score_finegrain.size(0)), max_idx]
            anomaly_scores_finegrain.append(anomaly_score_finegrain)
        anomaly_scores = [*anomaly_scores_global, *anomaly_scores_finegrain]

        # NOTE: this bilinear interpolation is not unreproducible and may occasionally lead to unstable ZSAD performance.
        for i in range(len(anomaly_maps)):
            B, L, C = anomaly_maps[i].shape
            H = int(np.sqrt(L))
            anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
            anomaly_maps[i] = F.interpolate(anomaly_maps[i], size=self.image_size, mode='bilinear', align_corners=True)

        if aggregation: # in the test stage, we firstly aggregate logits from all hierarchies and then do the softmax normalization
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
            
            anomaly_scores_global = torch.mean(torch.stack(anomaly_scores_global, dim = 1), dim = 1)
            anomaly_scores_global = anomaly_scores_global[:, 1]
            anomaly_scores_finegrain = torch.mean(torch.stack(anomaly_scores_finegrain, dim = 1), dim = 1)
            anomaly_scores_finegrain = anomaly_scores_finegrain[:, 1]
            anomaly_score = 0.9 * anomaly_scores_global + 0.1 * anomaly_scores_finegrain
            return anomaly_map, anomaly_score
        else: # otherwise, we do the softmax normalization for individual hierarchies
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            return anomaly_maps, anomaly_scores

    @torch.cuda.amp.autocast()
    def forward(self, image, cls_name, aggregation=True):
        # extract features for images and texts
        image_features, patch_tokens, semantic_map_token, text_features = self.extract_feat(image, cls_name)
        anomaly_map, anomaly_score = self.visual_text_similarity(image_features, [patch_tokens, semantic_map_token], text_features, aggregation)

        if aggregation:
            anomaly_map = anomaly_map # tensor
            anomaly_score = anomaly_score
            anomaly_map = anomaly_map.squeeze(1)

            return anomaly_map, anomaly_score
        else:
            anomaly_maps = anomaly_map # list
            anomaly_score = anomaly_score
            return anomaly_maps, anomaly_score