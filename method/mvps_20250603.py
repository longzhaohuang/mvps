from typing import Union, List, Optional
import numpy as np
import torch
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans

from .clip_model import CLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .transformer import ResidualAttentionBlock

# class Adapter(nn.Module):
#     def __init__(self, text_channel, visual_channel, last_layer : bool = False):
#         super(Adapter, self).__init__()
	   
#         self.fc1 = nn.Linear(visual_channel,  visual_channel)
#         self.fc2 = nn.Linear(text_channel,  visual_channel)
#         self.fc3 = nn.Linear(visual_channel,  text_channel)

#         self.pad = nn.ReplicationPad2d(1)
#         self.agg_conv = nn.Sequential(
#             nn.Conv2d(visual_channel, visual_channel, (3,3), (1,1), (0,0)),
#             nn.BatchNorm2d(visual_channel),
#             nn.ReLU()
#         )
#         self.weight_conv = nn.Sequential(
#             nn.Conv2d(visual_channel, 1, (1,1), (1,1), (0,0)),
#             nn.ReLU()
#         )

#     def forward(self, x, text_feat = None):
#         numtokens, Bs, _ = x.shape
#         feat_w = feat_h  = int(numtokens ** .5)
# 	    # Viusal -> Text
#         seg_token = F.relu(self.fc1(x))

#         seg_token_feat_map = seg_token.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w) 
#         seg_token_feat_map = self.pad(seg_token_feat_map)
#         seg_token_agg_feat_map = self.agg_conv(seg_token_feat_map) 
#         seg_token_output = seg_token_agg_feat_map.reshape(Bs, -1, feat_h * feat_w).permute(2,0,1)
#         seg_token_output_last = self.fc3(seg_token_output)
#         seg_token_agg_weight_map = self.weight_conv(seg_token_agg_feat_map) 
	    
#         # Text  -> Visual
#         align_token          = F.relu(self.fc2(seg_token))
#         align_token_feat_map = align_token.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w) * seg_token_agg_weight_map
#         align_token_output   = align_token_feat_map.reshape(Bs, -1, feat_h * feat_w).permute(2,0,1)
#         return seg_token_output_last, align_token_output

# class Adapter(nn.Module):
#     def __init__(self, text_channel, visual_channel, last_layer : bool = False):
#         super(Adapter, self).__init__()

#         self.last_layer = last_layer
#         self.fc1 = nn.Linear(visual_channel, text_channel)
#         if self.last_layer: return

#         self.fc2 = nn.Linear(text_channel  , visual_channel)
#         # 尝试将 FC2 替换为对 特征的聚合
#         # self.simple_conv = nn.Sequential(
#         #     nn.Conv2d(text_channel, visual_channel, (3,3), (1,1), (1,1)),
#         #     nn.BatchNorm2d(visual_channel),
#         #     nn.ReLU()
#         # )
#         self.simple_conv = nn.Sequential(
#             nn.Conv2d(text_channel, text_channel, (3,3), (1,1), (1,1)),
#             nn.BatchNorm2d(text_channel),
#             nn.ReLU(),
#             nn.Conv2d(text_channel, visual_channel, (1,1), (1,1), (0,0)),
#             nn.ReLU(),
#         )

#         # 方案7431: 去掉聚合特性, 使用 1x1 卷积
#         # 没有保留 VISA 的精度
#         # self.simple_conv = nn.Sequential(
#         #     nn.Conv2d(text_channel, 1, (1,1), (1,1), (0,0)),
#         #     nn.BatchNorm2d(1),
#         #     nn.ReLU()
#         # ) 
#         # 方案7431-1
#         # self.simple_conv = nn.Sequential(
#         #     nn.Conv2d(text_channel, 1, (1,1), (1,1), (0,0)),
#         #     nn.ReLU()
#         # )

#         # 方案7431-2: 去除 Padding 的影响
#         # self.pad = nn.ReplicationPad2d(1)
#         # self.simple_conv = nn.Sequential(
#         #     nn.Conv2d(text_channel, visual_channel, (3,3), (1,1), (0,0)),
#         #     nn.BatchNorm2d(visual_channel),
#         #     nn.ReLU()
#         # )


#     def forward(self, x, text_feat = None):
#         cls_token, patch_token = x[0:1,:,:], x[1:,:,:]
#         numtokens, Bs, _ = patch_token.shape
#         feat_w = feat_h  = int(numtokens ** .5)
        
#         x_stage1    = F.relu(self.fc1(x))
#         if self.last_layer: return x_stage1[1:,::]

#         x_stage2_fc = F.relu(self.fc2(x_stage1))

#         # ============ Adapter as Gate unit ============
#         x_stage1_reshape = x_stage1[1:,:,:].permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
#         x_stage2_input   = x_stage2_fc[1:,:,:].permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)

#         # x_stage2_weight  = self.simple_conv(x_stage1_reshape)
#         # x_stage2_weight = self.pad(x_stage2_weight)
#         # x_stage2_weight  = self.simple_conv(x_stage1_reshape_pad)

#         x_stage2_weight  = self.simple_conv(x_stage1_reshape)
#         x_stage2         = x_stage2_input * x_stage2_weight
#         x_stage2         = x_stage2.reshape(Bs, -1, feat_h * feat_w).permute(2,0,1)
#         # ============ Adapter as Gate unit ============
#         x_stage2         = torch.cat([cls_token, x_stage2], dim=0)
#         return x_stage1[1:,::], x_stage2 

# class Adapter(nn.Module):
#     def __init__(self, text_channel, visual_channel, last_layer : bool = False):
#         super(Adapter, self).__init__()

#         self.last_layer = last_layer
#         self.fc1 = nn.Linear(visual_channel, text_channel)
#         self.simple_conv1 = nn.Sequential(
#             nn.Conv2d(text_channel, text_channel, (3,3), (1,1), (1,1)),
#             nn.BatchNorm2d(text_channel),
#             nn.ReLU()
#         )
#         if self.last_layer: return

#         self.fc2 = nn.Linear(text_channel  , visual_channel)
#         # 尝试将 FC2 替换为对 特征的聚合
#         # self.simple_conv = nn.Sequential(
#         #     nn.Conv2d(text_channel, visual_channel, (3,3), (1,1), (1,1)),
#         #     nn.BatchNorm2d(visual_channel),
#         #     nn.ReLU()
#         # )
#         self.simple_conv2 = nn.Sequential(
#             nn.Conv2d(text_channel, 1, (1,1), (1,1), (0,0)),
#             nn.ReLU()
#         )

#         # 方案7431: 去掉聚合特性, 使用 1x1 卷积
#         # 没有保留 VISA 的精度
#         # self.simple_conv = nn.Sequential(
#         #     nn.Conv2d(text_channel, 1, (1,1), (1,1), (0,0)),
#         #     nn.BatchNorm2d(1),
#         #     nn.ReLU()
#         # ) 
#         # 方案7431-1
#         # self.simple_conv = nn.Sequential(
#         #     nn.Conv2d(text_channel, 1, (1,1), (1,1), (0,0)),
#         #     nn.ReLU()
#         # )

#         # 方案7431-2: 去除 Padding 的影响
#         # self.pad = nn.ReplicationPad2d(1)
#         # self.simple_conv = nn.Sequential(
#         #     nn.Conv2d(text_channel, visual_channel, (3,3), (1,1), (0,0)),
#         #     nn.BatchNorm2d(visual_channel),
#         #     nn.ReLU()
#         # )


#     def forward(self, x, text_feat = None):
#         numtokens, Bs, _ = x.shape
#         feat_w = feat_h  = int(numtokens ** .5)
        
#         x_stage1    = F.relu(self.fc1(x))
        
#         # ============ Adapter as Gate unit ============
#         x_stage1_reshape = x_stage1.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
#         # x_stage2_weight  = self.simple_conv(x_stage1_reshape)
#         # x_stage2_weight = self.pad(x_stage2_weight)
#         # x_stage2_weight  = self.simple_conv(x_stage1_reshape_pad)
#         x_stage2_weight  = self.simple_conv1(x_stage1_reshape)
#         seg_token = x_stage2_weight.reshape(Bs, -1, feat_h * feat_w).permute(2,0,1)
#         if self.last_layer: return seg_token
#         x_stage2_fc = F.relu(self.fc2(seg_token))
#         x_stage2_input   = x_stage2_fc.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
#         x_stage2_weight  = self.simple_conv2(x_stage2_weight)
#         x_stage2         = x_stage2_input * x_stage2_weight
#         x_stage2         = x_stage2.reshape(Bs, -1, feat_h * feat_w).permute(2,0,1)
#         # ============ Adapter as Gate unit ============
#         return x_stage1, x_stage2 

# class Adapter(nn.Module):
#     def __init__(self, text_channel, visual_channel, last_layer : bool = False):
#         super(Adapter, self).__init__()

#         self.last_layer = last_layer
#         self.fc1 = nn.Linear(visual_channel, text_channel)
#         if self.last_layer: return

#         self.fc2 = nn.Linear(text_channel  , visual_channel)
#         self.simple_conv = nn.Sequential(
#             nn.Conv2d(text_channel, visual_channel, (3,3), (1,1), (1,1)),
#             nn.BatchNorm2d(visual_channel),
#             nn.ReLU()
#         )

#     def forward(self, x, text_feat = None):
#         patch_token = x
#         numtokens, Bs, _ = patch_token.shape
#         feat_w = feat_h  = int(numtokens ** .5)
        
#         x_stage1    = F.relu(self.fc1(patch_token))
#         if self.last_layer: return x_stage1
#         x_stage2_fc = F.relu(self.fc2(x_stage1))
#         # ============ Adapter as Gate unit ============
#         x_stage1_reshape = x_stage1.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
#         x_stage2_input   = x_stage2_fc.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
#         x_stage2_weight  = self.simple_conv(x_stage1_reshape)
#         x_stage2         = x_stage2_input * x_stage2_weight
#         x_stage2         = x_stage2.reshape(Bs, -1, feat_h * feat_w).permute(2,0,1)
#         # ============ Adapter as Gate unit ============
#         return x_stage1, x_stage2 

def mean_filter(input_tensor, kernel_size):
    kernel = torch.ones((1, 1, kernel_size[0], kernel_size[1]), device=input_tensor.device) / (kernel_size[0] * kernel_size[1])     # Build Mean Filter Kernel
    kernel = kernel.expand(input_tensor.size(1), 1, *kernel_size)                                                                   # Expand Mean Filter Kerenl
    return F.conv2d(input_tensor, kernel, padding=(kernel_size[0] // 2, kernel_size[1] // 2), groups=input_tensor.size(1))          # Execute convolution

# 增强跨领域的能力
class Adapter(nn.Module):
    def __init__(self, text_channel, visual_channel, last_layer:bool=False):
        super(Adapter, self).__init__()

        self.fc1 = nn.Linear(visual_channel, text_channel)
        self.last_layer = last_layer
        if last_layer: return
        self.fc2 = nn.Linear(text_channel  , visual_channel)
        self.fc3 = nn.Sequential(
            nn.Linear(text_channel  , visual_channel),
            nn.ReLU()
        )

    def forward(self, x, frozen_information = None):
        # ========= get shape information =========
        numtokens, Bs, _ = x.shape
        feat_w = feat_h  = int(numtokens ** .5)
        # ========= get shape information =========

        x_stage1_fc = F.relu(self.fc1(x))
        if self.last_layer: return x_stage1_fc
        x_stage2_fc = F.relu(self.fc2(x_stage1_fc))

        # ========= RESHAPE =========
        x_reshape           = x.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
        x_stage1_fc_reshape = x_stage1_fc.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
        x_stage2_fc_reshape = x_stage2_fc.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
        frozen_information_reshape = frozen_information.permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
        # ========= RESHAPE =========
        # ========= Project Forzen Information Here =========
        x_stage2_fc_reshape = x_stage2_fc_reshape + frozen_information_reshape
        # ========= Project Forzen Information Here =========

        # ========= MEAN FILTER =========
        x_stage1_fc_reshape_agg = \
                0.5 * 0.33 * mean_filter(x_stage1_fc_reshape, (3, 3)) + \
                0.5 * 0.33 * mean_filter(x_stage1_fc_reshape, (1, 3)) + \
                0.5 * 0.33 * mean_filter(x_stage1_fc_reshape, (3, 1)) + \
                0.5 * x_stage1_fc_reshape
        x_stage1_fc_reshape_agg = x_stage1_fc_reshape_agg.reshape(Bs, -1, feat_h * feat_w).permute(2,0,1)
        x_stage1_fc_reshape_agg_ad  = self.fc3(x_stage1_fc_reshape_agg).permute(1,2,0).reshape(Bs, -1, feat_h, feat_w)
        # ========= MEAN FILTER =========

        # ========= DOT MULTIPLY =========
        shallow_fusion_information = x_stage2_fc_reshape * x_stage1_fc_reshape_agg_ad
        shallow_fusion_information = shallow_fusion_information.reshape(Bs, -1, feat_h * feat_w).permute(2,0,1)
        # ========= DOT MULTIPLY =========
        
        return x_stage1_fc, shallow_fusion_information

class ProjectLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_replicas, stack=False, is_array=True):
        super(ProjectLayer, self).__init__()

        self.head = nn.ModuleList([
            nn.Linear(input_dim, output_dim) 
            for _ in range(num_replicas)
        ])
        self.num_replicas = num_replicas
        self.stack = stack
        self.is_array = is_array

    def forward(self, tokens):
        out_tokens = []
        for i in range(self.num_replicas):
            if self.is_array:
                if len(tokens[i].shape) == 3: # For Patch Tokens
                    temp = self.head[i](tokens[i][:, 1:, :]) # for ViT, we exclude the class token and only extract patch tokens here.
                else: # For CLS Tokens
                    temp = self.head[i](tokens[i])
            else:
                if len(tokens.shape) == 3:
                    temp = self.head[i](tokens[:,i,:])
                else :
                    temp = self.head[i](tokens)

            out_tokens.append(temp)

        if self.stack:
            out_tokens = torch.stack(out_tokens, dim=1)

        return out_tokens

# class PromptLayer(nn.Module):
#     def __init__(self, channel, length, depth, is_text, prompting_type, enabled=True, stat_length=20):
#         super(PromptLayer, self).__init__()

#         self.channel = channel
        
#         self.dpto_length = length
#         self.stat_length = stat_length # DeFault : 20
        
#         self.depth = depth
#         self.is_text = is_text
#         self.enabled = enabled

#         self.prompting_type = prompting_type

#         if self.enabled: # only when enabled, the parameters should be constructed
#             if 'S' in prompting_type: # static prompts
#                 # learnable
#                 self.static_prompts = nn.ParameterList(
#                     [nn.Parameter(torch.empty(self.stat_length, self.channel))
#                      for _ in range(self.depth)])

#                 for single_para in self.static_prompts:
#                     nn.init.normal_(single_para, std=0.02)

#             if 'D' in prompting_type: # dynamic prompts
#                 self.dynamic_prompts = [0.] # place holder
#             self.scale_prompt = [0.]
#             self.prompt_layer_id = 1
#             self.prompt_id = 0

#     def set_dynamic_prompts(self, dynamic_prompts):
#         self.dynamic_prompts = dynamic_prompts

#     def set_scale_prompt(self, scale_prompts):
#         self.scale_prompt = scale_prompts

#     def forward_visual(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, prompt_id : int = 0, diag_attn : int = -1):
#         if self.enabled:
#             # length = self.length

#             # only prompt the first J layers
#             if indx < self.depth:
#                 static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
#                 visual_context = static_prompts
            
#             if indx == 0:  # for the first layer
#                 visual_context = visual_context.permute(1, 0, 2).half()
#                 x = torch.cat([x, visual_context], dim=0)
#             else:
#                 if indx < self.depth:  # replace with learnalbe tokens
#                     prefix = x[0:x.shape[0] - (self.stat_length + self.dpto_length), :, :]            # prefix
#                     suffix = x[x.shape[0] - self.dpto_length:, : , :]                                 # suffix
#                     visual_context = visual_context.permute(1, 0, 2).half()
#                     x = torch.cat([prefix, visual_context, suffix], dim=0)
#                 else:  # keep the same
#                     x = x

#             if self.prompt_id != prompt_id:
#                 self.prompt_layer_id = 0
#                 self.prompt_id = prompt_id

#             if indx == 0:
#                 scale_context = self.scale_prompt[0][0,::].half()
#                 x = torch.cat([x, scale_context], dim = 0)
#             else:
#                 if self.prompt_layer_id < (self.depth - 1):
#                     prefix = x[0:x.shape[0] - self.dpto_length, :, :]                                    # prefix
#                     scale_context = self.scale_prompt[self.prompt_id][self.prompt_layer_id,::].half()
#                     x = torch.cat([prefix, scale_context], dim=0)
#                     self.prompt_layer_id += 1
#                 else:
#                     x = x
            
#         else:
#             x = x

#         x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask, diag_attn=diag_attn)
        
#         if diag_attn != -1:
#             tokens = x[1]
#             x      = x[0]
#             return x, tokens, attn_mask
#         if self.enabled:
#             tokens = x[0:x.shape[0] - (self.stat_length + self.dpto_length), :, :]
#             return x, tokens, attn_tmp
#         else:
#             tokens = x
#             return x, tokens, attn_tmp

#     def forward(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, prompt_id : int = 0, diag_attn : int = -1):
#         if not self.is_text:
#             return self.forward_visual(resblock, indx, x, k_x, v_x, attn_mask, prompt_id,  diag_attn)

class PromptLayer(nn.Module):
    def __init__(self, channel, length, depth, is_text, prompting_type, enabled=True, stat_length : int = 0):
        super(PromptLayer, self).__init__()

        stat_length = 0 # 该超参数不会被使用到

        self.channel = channel
        self.length = length
        self.depth = depth
        self.is_text = is_text
        self.enabled = enabled

        self.prompting_type = prompting_type

        if self.enabled: # only when enabled, the parameters should be constructed
            # if 'S' in prompting_type: # static prompts
            #     # learnable
            #     self.static_prompts = nn.ParameterList(
            #         [nn.Parameter(torch.empty(self.length, self.channel))
            #          for _ in range(self.depth)])

            #     for single_para in self.static_prompts:
            #         nn.init.normal_(single_para, std=0.02)

            if 'D' in prompting_type: # dynamic prompts
                self.dynamic_prompts = [0.] # place holder
            self.scale_prompt = [0.]
            self.prompt_layer_id = 1
            self.prompt_id = 0

    def set_dynamic_prompts(self, dynamic_prompts):
        self.dynamic_prompts = dynamic_prompts

    def set_scale_prompt(self, scale_prompts):
        self.scale_prompt = scale_prompts

    def forward_visual(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, prompt_id : int = 0, diag_attn : int = -1):
        if self.enabled:
            length = self.length

            if self.prompt_id != prompt_id:
                self.prompt_layer_id = 0
                self.prompt_id = prompt_id

            if indx == 0:
                scale_context = self.scale_prompt[0][0,::].half()
                x = torch.cat([x, scale_context], dim = 0)
            else:
                if self.prompt_layer_id < (self.depth - 1):
                    prefix = x[0:x.shape[0] - length, :, :]
                    scale_context = self.scale_prompt[self.prompt_id][self.prompt_layer_id,::].half()
                    x = torch.cat([prefix, scale_context], dim=0)
                    self.prompt_layer_id += 1
                else:
                    x = x
            
        else:
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask, diag_attn=diag_attn)
        
        if diag_attn != -1:
            tokens = x[1]
            x      = x[0]
            return x, tokens, attn_mask
        if self.enabled:
            tokens = x[0:x.shape[0] - length, :, :]
            return x, tokens, attn_tmp
        else:
            tokens = x
            return x, tokens, attn_tmp

    def forward(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, prompt_id : int = 0, diag_attn : int = -1):
        if not self.is_text:
            return self.forward_visual(resblock, indx, x, k_x, v_x, attn_mask, prompt_id,  diag_attn)

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
        self.prompt_placeholder = " ".join(["X"] * length) 

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
            all_tokens = [[sot_token] + self.tokenizer.encode(self.prompt_placeholder + ' ' + "object") + [eot_token] for text in texts] 
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
        length = [self.prompt_normal_length, self.prompt_abnormal_length]
        bs = len(texts)

        for state_id in range(len(class_embeddings)):
            for class_embeddings_id in range(len(class_embeddings[state_id])):
                class_embedding = class_embeddings[state_id][class_embeddings_id].reshape(bs, length[state_id], -1)
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                class_embedding = class_embedding.mean(dim=1)
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                class_embeddings[state_id][class_embeddings_id] = class_embedding
    
        if len(self.prompt_state) != 2:
            raise RuntimeError("len(self.prompt_state) must equal to 2")

        text_features = [torch.stack([i, j], dim=1) for i , j in zip(class_embeddings[0], class_embeddings[1])]

        return text_features

    def forward(self, model, texts, device):
        text_features = self.encode_text(model, texts, device)
        for text_feature_id in range(len(text_features)):
            text_features[text_feature_id] = F.normalize(text_features[text_feature_id], dim=-1)
            text_features[text_feature_id] = text_features[text_feature_id].permute(0,2,1)
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
            self.learnable_prompt = nn.ModuleList()
            for _ in range(2):
                self.learnable_prompt.append(    
                    torch.nn.ParameterList([
                        nn.Parameter(torch.empty(self.length, channel))
                        for _ in range(self.prompt_num)
                    ])            
                )
            for learn_para in self.learnable_prompt:
                for single_para in learn_para:
                    nn.init.normal_(single_para, std=0.02)
                        
            if 'S' in prompting_type:
                self.static_prompts = nn.ModuleList()
                for _ in range(self.prompt_num):
                    self.static_prompts.append(
                        nn.ParameterList(
                            [
                                nn.Parameter(torch.empty(self.length, channel))
                                for _ in range(self.prompt_depth - 1)
                            ]                         
                        )    
                    )
                
                for static_prompt in self.static_prompts:
                    for single_para in static_prompt:
                        nn.init.normal_(single_para, std=0.02)

            if 'D' in prompting_type:
                self.dynamic_prompts = [0.]    
            self.frist_dynamic_prompts = None
            self.memory_prompt = [] # 记录提示 

    def set_dynamic_prompts(self, dynamic_prompts):
        self.dynamic_prompts = dynamic_prompts
    
    def set_first_dynamic_prompts(self, dynamic_prompts):
        self.frist_dynamic_prompts = dynamic_prompts

    def reset_memory_prompt(self, ):
        self.memory_prompt = []

    def forward(self, resblock, indx, prompt_id = 0, x = None, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None, state_idx = 0):
        if self.enabled:
            if 0 < indx < self.prompt_depth:
                if 'S' in self.prompting_type and 'D' in self.prompting_type: # both
                    B, tuning_length, C = self.dynamic_prompts[prompt_id].shape
                    _ , BT, C  = x.shape
                    T = BT // B 
                    static_prompts = self.static_prompts[prompt_id][indx-1].unsqueeze(0).expand(x.shape[1], -1, -1)                    
                    dynamic_context = self.dynamic_prompts[prompt_id].unsqueeze(1).repeat_interleave(T, dim=1).view(-1, tuning_length, C)
                    textual_context = dynamic_context + static_prompts
                elif 'S' in self.prompting_type:  # static
                    static_prompts = self.static_prompts[prompt_id][indx-1].unsqueeze(0).expand(x.shape[1], -1, -1)
                    textual_context = static_prompts
                elif 'D' in self.prompting_type:  # dynamic
                    B, tuning_length, C = self.dynamic_prompts[prompt_id].shape
                    _ , BT, C  = x.shape
                    T = BT // B 
                    textual_context = self.dynamic_prompts[prompt_id].unsqueeze(1).repeat_interleave(T, dim=1).view(-1, tuning_length, C)
                else:
                    print('You should at least choose one type of prompts when the prompting branches are not none.')
                    raise NotImplementedError

            if indx == 0:
                prefix = x[:1, :, :]
                suffix = x[1 + self.length:, :, :]

                B, tuning_length, C = self.frist_dynamic_prompts[state_idx][prompt_id].shape
                _ , BT, C  = x.shape
                T = BT // B
                dynamic_context = self.frist_dynamic_prompts[state_idx][prompt_id].unsqueeze(1).repeat_interleave(T, dim=1).view(-1, tuning_length, C)
                dynamic_context = dynamic_context.permute(1,0,2)

                learnable_prompt = self.learnable_prompt[state_idx][prompt_id].unsqueeze(1).repeat(1,prefix.shape[1],1) + dynamic_context
                 
                x = torch.cat([prefix, learnable_prompt, suffix], dim=0)            
            elif indx < self.prompt_depth:
                prefix = x[:1, :, :]
                suffix = x[1 + self.length:, :, :]
                textual_context = textual_context.permute(1, 0, 2)
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

class HMPT(nn.Module):
    def __init__(self, text_channel=768, visual_channel=1024, hidden_dim=256):
        super(HMPT, self).__init__()

        self.visual_rca = \
            ResidualAttentionBlock(
                d_model=visual_channel,
                n_head=8,
                is_cross_attention=True,
                n_dropout=.1
            )
        self.scale_adapter = nn.Sequential(
            nn.Linear(visual_channel, hidden_dim),
            nn.Dropout(.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, visual_channel)
        )

    def forward(self, model, patch_tokens, texts, device, patch_token_id):
        gobal_cls_token = patch_tokens[-1][:,0,:]
        if patch_token_id == len(patch_tokens) - 1:
            return [gobal_cls_token, gobal_cls_token]
        
        scale_patch_token = patch_tokens[patch_token_id][:,1:,:]
        gobal_cls_token = self.scale_adapter(gobal_cls_token)
        q =  gobal_cls_token.unsqueeze(1).permute(1,0,2)
        kv = torch.cat([gobal_cls_token.unsqueeze(1), scale_patch_token], dim=1).permute(1,0,2)
        
        gobal_cls_token = \
            self.visual_rca(
                q_x = q, 
                k_x = kv, 
                v_x = kv)[0].permute(1,0,2)[:,0,:]
        return [gobal_cls_token, gobal_cls_token]

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
        self.prompt_num = len(output_layers)
        self.output_layers = output_layers
        self.freeze_clip = freeze_clip
        self.visual = self.freeze_clip.visual
        self.transformer = self.freeze_clip.transformer
        self.token_embedding = self.freeze_clip.token_embedding
        self.positional_embedding = self.freeze_clip.positional_embedding
        self.ln_final = self.freeze_clip.ln_final
        self.text_projection = self.freeze_clip.text_projection
        self.attn_mask = self.freeze_clip.attn_mask

        # 设置提示微调参数
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.use_hsf = use_hsf
        # Replace k_clusters with stat_length
        self.k_clusters = k_clusters
        self.stat_length= 10

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
            length=self.prompting_length
        )
        self.text_prompter = MutilPrompt_TextPromptLayer(
            text_channel,
            prompting_length,
            prompt_depth=prompting_depth,
            prompting_type=prompting_type,
            enabled=self.enable_text_prompt,
            prompt_num=self.prompt_num
        )
        self.visual_prompter = PromptLayer(
            visual_channel, 
            prompting_length, 
            prompting_depth, 
            is_text=False,
            prompting_type=prompting_type,
            enabled=self.enable_visual_prompt,
            stat_length=self.stat_length
        )

        # For location:
        # self.patch_token_layer = nn.ModuleList(
        #     # [nn.Linear(visual_channel, text_channel) for _ in range(self.prompt_num + 1)]
        #     [
        #         *[Adapter(text_channel, visual_channel) for _ in range(self.prompt_num - 1)],
        #         *[nn.Linear(visual_channel, text_channel) for _ in range(2)],
        #     ] 
        # )
        self.patch_token_layer = nn.ModuleList(
            [
                *[Adapter(text_channel, visual_channel) for _ in range(self.prompt_num - 1)],
                *[Adapter(text_channel, visual_channel, last_layer=True) for _ in range(2)]
                # *[nn.Linear(visual_channel, text_channel) for _ in range(1)],
                # *[nn.Linear(visual_channel, text_channel) for _ in range(1)],
            ] 
        )

        # For detection:
        self.cls_token_layer = nn.ModuleList(
            [nn.Linear(visual_channel, text_channel) for _ in range(1)]
        )

        self.DPTO = nn.ModuleList()
        for i in range(self.prompt_num):
            self.DPTO.append(
                nn.Linear(
                    text_channel,
                    visual_channel
                )
            )
        
        self.dynamic_text_prompt_generator = nn.ModuleList()
        for _ in range(3):
            self.dynamic_text_prompt_generator.append(
                ProjectLayer(
                    visual_channel,
                    text_channel,
                    prompting_length,
                    stack=True,
                    is_array=False
                )
            )
        
        self.mvpcl = MVPCL(prompt_num=self.prompt_num, k_clusters=10, deta_k=0)
        self.hmpt = HMPT(
            text_channel,
            visual_channel,
            256
        )

        self.image_size = image_size
        self.device = device
        self.alpha = 0.5

        # USED TO GET EACH LAYER
        # self.weight_linear = nn.Linear(
        #     text_channel, 4
        # )
        # from .upsampler import IOUScorePrediction
        # self.weight_linear = IOUScorePrediction()

        # self.fusion_weight = None

        # 聚类中心的操作
        # from .GroupVit_struct import GroupingBlock_ligth
        # self.group_token_proj = nn.ModuleList()
        # for i in range(self.prompt_num):
        #     self.group_token_proj.append(
        #         nn.Linear(
        #             visual_channel,
        #             text_channel
        #         )
        #     ) 
        # self.stat_length -> 8
        # self.group_clustering = nn.ModuleList()
        # for i in range(self.prompt_num):
        #     self.group_clustering.append(
        #         GroupingBlock_ligth(
        #             dim       = visual_channel  , 
        #             out_dim   = visual_channel  ,
        #             num_heads = 1               , 
        #             num_group_token = self.stat_length,
        #             norm_layer = nn.LayerNorm
        #         )
        #     )
        
    def generate_and_set_dynamic_promtps(self, image, texts):
        with torch.no_grad():
            # extract image features
            image_features , patch_tokens = self.visual.forward(image, self.output_layers, single_output=False)

        text_normal_prompt = []
        text_abnormal_prompt = []
        tuning_text_prompt = []
        for patch_token_id in range(len(patch_tokens)):
            text_prompt = self.hmpt.forward(
                self,
                patch_tokens,
                texts,
                self.device,
                patch_token_id=patch_token_id
            )
            text_normal_prompt.append(self.dynamic_text_prompt_generator[0](text_prompt[0]))
            text_abnormal_prompt.append(self.dynamic_text_prompt_generator[1](text_prompt[1]))
            
            common_text_prompt = text_prompt[0] + text_prompt[1]
            common_text_prompt = F.normalize(common_text_prompt, dim = -1)
            tuning_text_prompt.append(
                self.dynamic_text_prompt_generator[-1](
                    common_text_prompt
                )
            )
        
        first_text_promt = [text_normal_prompt, text_abnormal_prompt]

        self.text_prompter.set_first_dynamic_prompts(first_text_promt)
        self.text_prompter.set_dynamic_prompts(tuning_text_prompt)
        
        return image_features, patch_tokens
    
    def encode_text(self, text, idx):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        prompt_set = []
        for prompt_id in range(self.prompt_num):
            for indx, r in enumerate(self.transformer.resblocks):
                x, attn_tmp = self.text_prompter(r, indx, prompt_id, x, k_x=None, v_x=None, attn_mask=self.attn_mask, state_idx=idx)
            prompt_set.append(x)
        
        prompt_set = [x.permute(1, 0, 2) for x in prompt_set]  # LND -> NLD
        prompt_set = [self.ln_final(x) for x in prompt_set]  # [batch_size, n_ctx, transformer.width]

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        prompt_set = [x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection for x in prompt_set]
        return prompt_set

    def custom_attn(self, attn_layer, x, attn_mask=None, mid_simi=None, attn_type="fused-attn"):
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if attn_type == "fused-attn" and attn_mask is not None:
            # sum to 1
            attn_mask /= torch.sum(attn_mask, dim=-2, keepdim=True)
            attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)
            attn_mask = (attn_mask + attn_mask.transpose(-2, -1)) / 2
            # attn_mask -= attn_mask.amax(-2, keepdim=True) * 0.1
            attn_mask = torch.clamp(attn_mask, 0)
            attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)
            attn_mask = attn_mask.flatten(0, 1)
            attn_weights = torch.repeat_interleave(attn_mask, dim=0, repeats=v.shape[0] // attn_mask.shape[0])
        elif attn_type == "q-q":
            attn_weights = torch.bmm(q * scale, q.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif attn_type == "k-k":
            attn_weights = torch.bmm(k * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif attn_type == "v-v":
            attn_weights = torch.bmm(v * scale, v.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif attn_type == "vanilla":
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif attn_type == "sim_ec" and mid_simi is not None:      #  use sim enhance vanilla attn
            mid_simi = (mid_simi - torch.mean(mid_simi)) * 3.0
            mid_simi[mid_simi < 0.0] = float('-inf')
            mid_simi = mid_simi.repeat(num_heads, 1, 1)
            attn_weights = F.softmax(mid_simi, dim=-1)
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights += F.softmax(attn_weights, dim=-1)
            attn_weights /= 2
        else:
            identity = torch.eye(v.shape[-2], dtype=v.dtype, device=v.device)[None]
            attn_weights = torch.repeat_interleave(identity, dim=0, repeats=v.shape[0])

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        if attn_type == "vanilla" or attn_type == "sim_ec":
            attn_output = attn_layer.out_proj(attn_output)
        # attn_output = attn_layer.out_proj(attn_output)

        return attn_output, attn_weights

    def adaptively_aggregate(self, maskclip_feats: torch.Tensor, corrs: torch.Tensor):
        # maskclip_feats : [1369, 1, 1024],  corr : [1, 1369, 1369]
        corrs_normalized = corrs / (corrs.sum(dim=-1, keepdim=True) + 1e-6)
        maskclip_feats_ref = torch.matmul(corrs_normalized, maskclip_feats.permute(1, 0, 2))
        return maskclip_feats_ref.permute(1, 0, 2)

    def encode_image(self, image, ori_patch_tokens,  text_features):
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
            x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        proj_patch_tokens  = []
        proj_cls_tokens    = []
        align_patch_tokens = []
        group_tokens       = []
        prompt_id = 0

        # Conference paper version :
        # for indx, r in enumerate(self.visual.transformer.resblocks):
        #     if (indx + 1) in self.output_layers:
        #         x, tokens, attn_tmp = \
        #             self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None, prompt_id=prompt_id, diag_attn=-1) # 在推理阶段不使用对角注意力机制
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

        #  尝试使用 shallow patch token 进行融合
        shallow_fusion_cls_patch_token = None

        #  分开两个阶段, 第一个阶段将结果全部推理出来， 第二个阶段再将结果通过最后一个 Transformer Layer
        #  第一个阶段将结果全部推理出来
        num_tokens, batch_size, embed_dim = x.shape
        #  不介入微调
        # img_features = torch.zeros([len(self.output_layers)] + [num_tokens, batch_size, embed_dim], device=x.device, dtype=x.dtype)
        #  没使用静态提示微调
        img_features = torch.zeros([len(self.output_layers)] + [num_tokens + self.prompting_length, batch_size, embed_dim], device=x.device, dtype=x.dtype)
        #  使用静态提示微调
        # img_features = torch.zeros([len(self.output_layers)] + [num_tokens + (self.prompting_length + self.stat_length), batch_size, embed_dim], device=x.device, dtype=x.dtype)
        for indx, r in enumerate(self.visual.transformer.resblocks):
            if (indx + 1) in self.output_layers:
                text_feat = text_features[prompt_id]
                img_features[prompt_id] = x
                cls_token, patch_token, scale_token = x[0:1,:,:], x[1:num_tokens,:,:], x[num_tokens:,:,:]
                if shallow_fusion_cls_patch_token is not None:
                    patch_token = patch_token + shallow_fusion_cls_patch_token

                # cls_patch_token, scale_token = x[0:num_tokens,:,:], x[num_tokens:,:,:]
                # if shallow_fusion_cls_patch_token is not None:
                #     cls_patch_token = cls_patch_token + shallow_fusion_cls_patch_token

                # ======================== VV Attn For Seg    ========================
                # To retina local feature , we use x from shallow layer to input the last layer:
                # if shallow_fusion_patch_token is not None:
                #     patch_token = patch_token + shallow_fusion_patch_token
                  
                # x_for_vv = torch.cat([cls_patch_token, scale_token], dim = 0)
                x_for_vv = torch.cat([cls_token, patch_token, scale_token], dim=0)
                ln_patch_token, attn = self.custom_attn(r.attn, r.ln_1(x_for_vv), attn_mask=None, attn_type="v-v")
                # For Segmentation : 
                forzen_information = ori_patch_tokens[prompt_id][:,1:num_tokens,:].permute(1, 0 ,2)
                patch_token_after_vvAttn = (ln_patch_token[1:num_tokens,:,:] + forzen_information) / 2
                if prompt_id != self.prompt_num - 1:
                    local_token, shallow_fusion_cls_patch_token = self.patch_token_layer[prompt_id](patch_token_after_vvAttn, forzen_information)
                else :
                    local_token = self.patch_token_layer[prompt_id](patch_token_after_vvAttn)
                proj_patch_tokens.append(local_token.permute(1, 0, 2))
                # ======================== VV Attn For Seg    ========================

                # ======================== QK Attn ========================
                x , tokens , attn_tmp = \
                    self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None, prompt_id=prompt_id)
                # ======================== QK Attn ========================
                prompt_id += 1
            else :
                x , tokens, attn_tmp = \
                    self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None, prompt_id=prompt_id)

        #  第二个阶段将前置结果通过最后一个 Transformer Layer
        #  ======================== Get Global Information from last layer ========================
        model_res = self.visual.transformer.resblocks[-1]
        for prompt_id in range(self.prompt_num):            
            # V2.5:
            ln_x, attn = self.custom_attn(model_res.attn, model_res.ln_1(img_features[prompt_id]), attn_mask=None, mid_simi=None, attn_type="vanilla")
            output_x = img_features[prompt_id] + model_res.ls_1(ln_x)
            output_x = output_x + model_res.ls_2(model_res.mlp(model_res.ln_2(output_x)))
            # local_token, shallow_fusion_patch_token = self.patch_token_layer[-1](output_x[1:num_tokens,:,:])
            forzen_information = ori_patch_tokens[self.prompt_num + prompt_id][:,1:num_tokens,:].permute(1, 0 ,2)
            patch_token_after_qkAttn = (output_x[1:num_tokens,:,:] + forzen_information) / 2
            local_token = self.patch_token_layer[-1](patch_token_after_qkAttn, forzen_information)
            align_patch_tokens.append(local_token.permute(1, 0, 2))
            proj_cls_tokens.append(self.cls_token_layer[-1](output_x[0,:,:]))
        #  ======================== Get Global Information from last layer ========================

        return proj_cls_tokens , proj_patch_tokens, align_patch_tokens, group_tokens

    def extract_feat(self, image, cls_name):
        ori_image_feature , ori_patch_tokens = self.generate_and_set_dynamic_promtps(image, cls_name) # generate and set dynamic prompts for corresponding prompters
        
        if self.enable_text_prompt:
            text_features = self.text_embedding_layer(self, cls_name, self.device)
        else:
            with torch.no_grad():
                text_features = self.text_embedding_layer(self, cls_name, self.device)

        visual_scale_prompt = []
        text_scale_prompt = self.text_prompter.memory_prompt[:self.prompt_num * (self.prompting_depth - 1)]
        pl , B,  C =  text_scale_prompt[0].shape
        text_scale_prompt = torch.stack(
            text_scale_prompt , 
            dim = 0
        ).reshape(self.prompt_num, self.prompting_depth - 1, pl, B, C) # [12, pl , B, C] -> [4, 3, pl, B, C]
        for prompt_id in range(self.prompt_num):
            visual_scale_prompt.append(
                self.DPTO[prompt_id](
                    text_scale_prompt[prompt_id, ::]
                )
            )
        self.visual_prompter.set_scale_prompt(
            visual_scale_prompt
        )            

        if self.enable_visual_prompt:
            proj_cls_tokens, proj_patch_tokens, align_patch_tokens, group_tokens = self.encode_image(image, ori_patch_tokens, text_features)
        else:
            with torch.no_grad():
                proj_cls_tokens, proj_patch_tokens, align_patch_tokens, group_tokens = self.encode_image(image, ori_patch_tokens, text_features)
        
        self.text_prompter.reset_memory_prompt()

        # proj_cls_tokens, proj_patch_tokens = self.proj_visual_tokens(image_features, patch_tokens, [ori_image_features, ori_patch_tokens])
        for idx in range(len(proj_cls_tokens)):
            proj_cls_tokens[idx] = proj_cls_tokens[idx] / proj_cls_tokens[idx].norm(dim=-1, keepdim=True)
        for idx in range(len(align_patch_tokens)):
            align_patch_tokens[idx] = align_patch_tokens[idx] / align_patch_tokens[idx].norm(dim=-1, keepdim=True)
        for idx in range(len(proj_patch_tokens)):
            proj_patch_tokens[idx] = proj_patch_tokens[idx] / proj_patch_tokens[idx].norm(dim=-1, keepdim=True)

        return proj_cls_tokens, proj_patch_tokens, align_patch_tokens, group_tokens, text_features, ori_image_feature

    def visual_text_similarity(self, image_feature, patch_token, text_feature, aggregation, texts, ori_image_feature):
        # =================== Using the weight to fuse all anomaly map  ===================
        # fusion_weight = self.weight_linear(ori_image_feature[-1].detach())
        # fusion_weight = torch.softmax(fusion_weight, dim=-1)
        # self.fusion_weight = fusion_weight
        # =================== Using the weight to fuse all anomaly map  ===================

        vv_tokens, qk_tokens, group_token = patch_token
        # vv_tokens is in shallow layerd
        # qk_tokens is in last    layer
        # group_token is clustering center For vv_tokens 

        anomaly_maps  = []
        seg_alpha     = .8
        for prompt_id in range(self.prompt_num):  
            anomaly_map_vv      = 100 * vv_tokens[prompt_id] @ text_feature[prompt_id]
            anomaly_map_qk      = 100 * qk_tokens[prompt_id] @ text_feature[prompt_id]
            anomaly_map         = (seg_alpha) * (anomaly_map_vv) + (1 - seg_alpha) * (anomaly_map_qk)
            anomaly_maps.append(anomaly_map)

        anomaly_scores = []
        
        cls_alpha = .5
        for prompt_id in range(self.prompt_num):  
            clustered_feature     = \
                self.mvpcl.forward(
                    qk_tokens[prompt_id], 
                    anomaly_maps[prompt_id], 
                    prompt_id
                ) # [B,K,D]            
            K                     = clustered_feature.shape[1]
            clustered_feature     = cls_alpha * clustered_feature + (1 - cls_alpha) * image_feature[prompt_id].unsqueeze(1).repeat(1,K,1)
            anomaly_score_cluster = 100 * clustered_feature @ text_feature[prompt_id] # [B,K,2]
            anomaly_score_cluster = torch.mean(anomaly_score_cluster, dim=1)          # [B,2]
            anomaly_score_cluster = torch.softmax(anomaly_score_cluster, dim=-1)      # [B,2]
            anomaly_scores.append(anomaly_score_cluster)

        # NOTE: this bilinear interpolation is not unreproducible and may occasionally lead to unstable ZSAD performance.
        for i in range(len(anomaly_maps)):
            B, L, C = anomaly_maps[i].shape
            H = int(np.sqrt(L))
            anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
            anomaly_maps[i] = F.interpolate(anomaly_maps[i], size=self.image_size, mode='bilinear', align_corners=True)

        if aggregation: # in the test stage, we firstly aggregate logits from all hierarchies and then do the softmax normalization
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=1)

            # =================== Using the weight to fuse all anomaly map  ===================
            # fusion_anomaly_map = 0
            # for i in range(len(anomaly_maps)):
            #     fusion_anomaly_map = fusion_anomaly_map + fusion_weight[:,i:i+1].unsqueeze(-1).unsqueeze(-1) * anomaly_maps[i]
            #     anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            # anomaly_map = torch.softmax(fusion_anomaly_map, dim=1)
            # =================== Using the weight to fuse all anomaly map  ===================

            anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
            anomaly_score = torch.mean(torch.stack(anomaly_scores, dim = 1), dim = 1)
            anomaly_score = anomaly_score[:, 1]

            return anomaly_map, anomaly_score
        else: # otherwise, we do the softmax normalization for individual hierarchies
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)

            # =================== Using the weight to fuse all anomaly map  ===================
            # fusion_anomaly_map = 0
            # for i in range(len(anomaly_maps)):
            #     fusion_anomaly_map = fusion_anomaly_map + fusion_weight[:,i:i+1].unsqueeze(-1).unsqueeze(-1) * anomaly_maps[i].detach()
            #     anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            # fusion_anomaly_map = torch.softmax(fusion_anomaly_map, dim=1)
            # anomaly_maps.append(fusion_anomaly_map)
            # =================== Using the weight to fuse all anomaly map  ===================

            return anomaly_maps, anomaly_scores

    @torch.cuda.amp.autocast()
    def forward(self, image, cls_name, aggregation=True):
        # extract features for images and texts
        image_features, patch_tokens, align_patch_tokens, group_tokens, text_features, ori_image_features = self.extract_feat(image, cls_name)
        anomaly_map, anomaly_score = self.visual_text_similarity(image_features, [patch_tokens, align_patch_tokens, group_tokens], text_features, aggregation, cls_name, ori_image_features)

        if aggregation:
            anomaly_map = anomaly_map # tensor
            anomaly_score = anomaly_score
            anomaly_map = anomaly_map.squeeze(1)

            return anomaly_map, anomaly_score
        else:
            anomaly_maps = anomaly_map # list
            anomaly_score = anomaly_score

            return anomaly_maps, anomaly_score
