import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from operator import mul
from copy import deepcopy
from torch.nn.modules.utils import _pair
from torch.nn.modules.loss import CrossEntropyLoss
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import *


class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output




class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionLayer(nn.Module):  # 交叉注意力+残差
    def __init__(self, d_model, nhead, dropout=0.1, ):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv, kv)
        q = q + self.dropout(self.mlp(self.norm(q)))
        return q




class FeatureVariance(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, features, text_features):
        # features: [batch_size, seq_len, feature_dim]
        # text_features: [num_text, feature_dim]
        # Reshape features to [batch_size * seq_len, feature_dim] for batch matrix multiplication
        #features_norm = F.normalize(features, p=2, dim=-1)
        #text_features_norm = F.normalize(text_features, p=2, dim=-1)

        features_flat = features.view(-1, self.feature_dim)
        # Calculate dot product similarity
        similarity_scores = torch.matmul(features_flat, text_features.t())  # [batch_size * seq_len, num_text]
        # Calculate variance of similarity scores along text feature dimension
        variance = torch.var(similarity_scores, dim=1, unbiased=False)  # [batch_size * seq_len]
        # Reshape back to [batch_size, seq_len]
        return variance.view(features.size(0), features.size(1))

class LocalFeatureAdd(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.variance_calc = FeatureVariance(feature_dim)

    def forward(self, low_features, high_features, text_features):
        low_variances = self.variance_calc(low_features, text_features)
        global_low_variance=low_variances[:,0].unsqueeze(1)
        low_variances_ratio=low_variances/global_low_variance
        high_variances = self.variance_calc(high_features, text_features)
        global_high_variance=high_variances[:,0].unsqueeze(1)
        high_variances_ratio=high_variances/global_high_variance
        ratio=low_variances_ratio/high_variances_ratio
        enhance_ratio=torch.clamp(ratio, min=1)
        #print(enhance_ratio.shape)
        enhanced_high_feature=high_features*enhance_ratio.unsqueeze(2)
        return enhanced_high_feature






class Stage1LocalAlignment(nn.Module):
    def __init__(self, d_model, num_heads=12, dropout=0.1, num_cmt_layers_first=1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        #self.spatial_attn = SpatialAttention(d_model)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout) for _ in range(num_cmt_layers_first)
        ])
        self.dropout = nn.Dropout(dropout)
        #self.se_attn_layers = SelfAttentionLayer(d_model, num_heads, dropout=dropout)

    def forward(self, q, kv_low_level):
        kv_low_level=self.norm(kv_low_level)
       # kv_low_level=self.se_attn_layers(kv_low_level)
        for cross_attn in self.cross_attn_layers:
            q = cross_attn(q, kv_low_level)
        return q





class Stage2GlobalFusion(nn.Module):
    def __init__(self, d_model, num_heads=12, dropout=0.1, num_cmt_layers_second=3):
        super().__init__()
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout) for _ in range(num_cmt_layers_second)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv_high_level):
        kv_high_level = self.norm(kv_high_level)
        for cross_attn in self.cross_attn_layers:
            q = cross_attn(q, kv_high_level)
        return q




class GatedFusion(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)  # Changing activation to ReLU for experimentation
        )

        self.norm = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, feature1, feature2):
        combined_features = torch.cat([feature1, feature2], dim=-1)
        gate_feature = self.attention_gate(combined_features)
        #gated_features = gate_values * feature1 + (1 - gate_values) * feature2
        # gated_features = self.dropout(gated_features)
        # output = self.norm(gated_features)
        return gate_feature



'''
class FeatureAggregator(nn.Module):
    def __init__(self, d_model=768, text_dim=768, hidden_dim=256, num_heads=8, dropout=0.1):
        super(FeatureAggregator, self).__init__()
        # 确保 hidden_dim 是 num_heads 的整数倍
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.query_projection = nn.Linear(d_model, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.enhance_projection = nn.Linear(hidden_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, high_features, low_features, text_features):
        # 计算低层和高层特征之间的差异
        diff_features = F.relu(low_features - high_features)  # 只保留正差异

        # 投影低层差异特征和文本特征到共同空间
        projected_diff = self.query_projection(diff_features)  # [batch_size, seq_len, hidden_dim]
        projected_text = self.text_projection(text_features)  # [num_texts, hidden_dim]

        # 注意力机制输入需要是 [seq_len, batch_size, hidden_dim]
        projected_diff = projected_diff.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        projected_text = projected_text.unsqueeze(1)  # [num_texts, 1, hidden_dim]
        key_value = projected_text.expand(-1, projected_diff.size(0), -1)  # [num_texts, seq_len, hidden_dim]

        attn_output, _ = self.attention(projected_diff, key_value, key_value)
        attn_output = attn_output.transpose(0, 1)  # 转换回 [batch_size, seq_len, hidden_dim]

        # 使用注意力输出加强原始低层特征
        enhanced_diff = self.enhance_projection(attn_output)
        enhanced_features = low_features + enhanced_diff
        enhanced_features = self.norm(enhanced_features)

        return enhanced_features

'''



class MSCI(nn.Module):

    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.cross_attn_dropout = config.cross_attn_dropout if hasattr(config, 'cross_attn_dropout') else 0.1
        self.prim_loss_weight = config.prim_loss_weight if hasattr(config, 'prim_loss_weight') else 1

        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.num_layers = self.clip.visual.transformer.layers  # 总层数
        self.enable_pos_emb = True
        self.selected_low_layers = config.selected_low_layers  # 默认为3
        self.selected_high_layers = config.selected_high_layers
        dtype = self.clip.dtype
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)

        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model

        self.additional_visual_params = self.add_visual_tunable_params()

        output_dim = self.clip.visual.output_dim
        self.local_feature_add = LocalFeatureAdd(output_dim)
        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()

        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)

        self.cmt = nn.ModuleList([CrossAttentionLayer(output_dim, output_dim // 64, self.cross_attn_dropout) for _ in
                                  range(config.cmt_layers)])
        self.lamda = nn.Parameter(torch.ones(output_dim) * config.init_lamda)
        self.lamda_2 = nn.Parameter(torch.ones(output_dim) * config.init_lamda)
        self.patch_norm = nn.LayerNorm(output_dim)

        # 定义拼接后的线性变换层
        self.concat_projection_low = nn.Sequential(
            nn.Linear(self.selected_low_layers * output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.concat_projection_high = nn.Sequential(
            nn.Linear(self.selected_high_layers * output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)

        )
        self.stage1_local_alignment = Stage1LocalAlignment(d_model=self.clip.visual.output_dim,
                                                           num_heads=config.stage_1_num_heads,
                                                           dropout=self.config.stage_1_dropout,
                                                           num_cmt_layers_first=config.stage_1_num_cmt_layers)
        self.stage2_global_fusion = Stage2GlobalFusion(d_model=self.clip.visual.output_dim,
                                                       num_heads=config.stage_2_num_heads,
                                                       dropout=config.stage_2_dropout,
                                                       num_cmt_layers_second=config.stage_2_num_cmt_layers)
        # self.stage3_fine_grained_enhancement = Stage3FineGrainedEnhancement(d_model=self.clip.visual.output_dim,
        # num_heads=8, dropout=0.2)

        self.fusion_module = GatedFusion(output_dim, dropout=self.config.fusion_dropout)

        # self.fusion_module_2=GatedFusion(output_dim,dropout=self.config.fusion_dropout)
        #self.lamda_2 = nn.Parameter(torch.ones(output_dim) * config.init_lamda)
        #self.feature_aggregator = FeatureAggregator(output_dim,output_dim)




    def multi_stage_cross_attention(self, q, low_level_features, high_level_features):
        # 第一阶段：局部特征对齐



        q_1 = self.stage1_local_alignment(q, low_level_features)
        q_2 = self.stage2_global_fusion(q_1, high_level_features)

        #q_fused = self.fusion_module(q, q_1)
        # print('q_1',q)
        # 第二阶段：全局特征融合


        # print('q_2', q)
        return q_1, q_2
        # return q_1,q_2

    def aggregate_features_low(self, visual_features):
        """
        使用拼接和线性变换来聚合不定数量的视觉特征层。
        :param visual_features: List[Tensor]，每个 tensor 大小为 [batch_size, seq_len, feature_dim]
        :return: Tensor, 大小为 [batch_size, seq_len, feature_dim]
        """
        # 检查输入是否为非空列表
        assert isinstance(visual_features, list) and len(
            visual_features) > 0, "Input should be a non-empty list of tensors."

        # 获取输入特征的基本维度信息
        batch_size, seq_len, feature_dim = visual_features[0].shape
        num_selected_layers = len(visual_features)

        # 将所有选择的层特征拼接在一起
        concat_features = torch.cat(visual_features, dim=-1)  # [batch_size, seq_len, num_selected_layers * feature_dim]

        # 通过线性层映射回 feature_dim 维度
        aggregated_features = self.concat_projection_low(concat_features)  # [batch_size, seq_len, feature_dim]

        return aggregated_features

    def aggregate_features_high(self, visual_features):
        """
        使用拼接和线性变换来聚合不定数量的视觉特征层。
        :param visual_features: List[Tensor]，每个 tensor 大小为 [batch_size, seq_len, feature_dim]
        :return: Tensor, 大小为 [batch_size, seq_len, feature_dim]
        """
        # 检查输入是否为非空列表
        assert isinstance(visual_features, list) and len(
            visual_features) > 0, "Input should be a non-empty list of tensors."

        # 获取输入特征的基本维度信息
        batch_size, seq_len, feature_dim = visual_features[0].shape
        num_selected_layers = len(visual_features)

        # 将所有选择的层特征拼接在一起
        concat_features = torch.cat(visual_features, dim=-1)  # [batch_size, seq_len, num_selected_layers * feature_dim]

        # 通过线性层映射回 feature_dim 维度
        aggregated_features = self.concat_projection_high(concat_features)  # [batch_size, seq_len, feature_dim]

        return aggregated_features

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width,
                                        bottleneck=self.config.adapter_dim,
                                        dropout=self.config.adapter_dropout
                                        ) for _ in range(adapter_num)])


        return params

    '''
    x_first torch.Size([16, 3, 224, 224])
x: torch.Size([16, 3, 224, 224])
x: torch.Size([16, 1024, 256])
x: torch.Size([16, 256, 1024])
x: torch.Size([16, 257, 1024])
x: torch.Size([16, 257, 1024])
x: torch.Size([257, 16, 1024])
img_feature torch.Size([16, 257, 1024]) 


    '''

    def encode_image(self, x: torch.Tensor):
        # print('x_first',x.shape)

        return self.encode_image_with_adapter(x)

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        low_level_features = []
        mid_level_features = []
        high_level_features = []

        for i_block in range(self.clip.visual.transformer.layers):
            # MHA
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual

            # FFN
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual

            # 保存每一层的特征
            x_feature = x.permute(1, 0, 2)  # LND -> NLD
            x_feature = self.clip.visual.ln_post(x_feature)
            if self.clip.visual.proj is not None:
                x_feature = x_feature @ self.clip.visual.proj  # [batch_size, seq_len, feature_dim]

            # 添加位置编码
            #x_feature = self.add_positional_encoding(x_feature, x_feature.size(1), x_feature.size(-1))

            # 提取特定层的特征
            if i_block < self.config.selected_low_layers:  # Low-level features (前8个Transformer块)
                low_level_features.append(x_feature)
                # high_level_features.append(x_feature)

            elif self.config.selected_low_layers <= i_block < 24 - self.config.selected_high_layers:  # Mid-level features (中间8个Transformer块)
                mid_level_features.append(x_feature)
            else:  # High-level features (最后8个Transformer块)
                high_level_features.append(x_feature)

        img_feature = high_level_features[-1]
        # print(high_level_features.shape)
       # stacked_low_level_features = torch.stack(low_level_features, dim=0)
       # stacked_high_level_features = torch.stack(high_level_features, dim=0)
        #max_pooled_low_level_features = torch.max(stacked_low_level_features, dim=0)[0]  # 通过 [0] 来提取第一个返回值 values
        #mean_pooled_high_level_features = torch.mean(stacked_high_level_features, dim=0)
       # low_level_features=self.aggregate_features_low(low_level_features)
        #high_level_features=self.aggregate_features_high(high_level_features)

        '''
        stacked_low_level_features = torch.stack(low_level_features, dim=0)
        stacked_high_level_features = torch.stack(high_level_features, dim=0)

        # 应用最大池化于底层特征
        max_pooled_low_level_features = torch.max(stacked_low_level_features, dim=0)[0]#通过 [0] 来提取第一个返回值 values
        mean_pooled_high_level_features = torch.mean(stacked_high_level_features, dim=0)

        #print(len(low_level_features))
        #print(len(high_level_features))
        #print(len(low_level_features))
        #print(len(high_level_features))
         '''

        #low_level_features = self.aggregate_features_low(low_level_features)


        if self.config.selected_high_layers==1:
            high_level_features = high_level_features[-1]

        else:
            high_level_features=self.aggregate_features_high(high_level_features)
       
        if self.config.selected_low_layers == 1:
            low_level_features = low_level_features[0]

        else:
            low_level_features = self.aggregate_features_low(low_level_features)
        
        return img_feature[:, 0, :],low_level_features, high_level_features

    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template,
                                   context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                                context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1: 1 + n_ctx[0], :].to(self.clip.dtype)  # 组合上下文前缀放入
        attr_ctx_vectors = embedding[1, 1: 1 + n_ctx[1], :].to(self.clip.dtype)  # 属性上下文前缀放入
        obj_ctx_vectors = embedding[2, 1: 1 + n_ctx[2], :].to(self.clip.dtype)  # 物体上下文前缀放入

        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(
                class_token_ids.cuda()
            ).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)

        # 组合替换
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
            obj_idx + self.offset
            ].type(self.clip.dtype)
        token_tensor[0][
        :, 1: len(self.comp_ctx_vectors) + 1, :
        ] = self.comp_ctx_vectors.type(self.clip.dtype)

        # 属性替换

        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
                                                :self.offset
                                                ].type(self.clip.dtype)
        token_tensor[1][
        :, 1: len(self.attr_ctx_vectors) + 1, :
        ] = self.attr_ctx_vectors.type(self.clip.dtype)

        # 物体替换
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
                                                self.offset:
                                                ].type(self.clip.dtype)
        token_tensor[2][
        :, 1: len(self.obj_ctx_vectors) + 1, :
        ] = self.obj_ctx_vectors.type(self.clip.dtype)

        return token_tensor

    def loss_calu(self, predict, target):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target

        # 检查 predict 的类型
        if isinstance(predict, tuple) and len(predict) == 2:
            # 训练阶段，predict 包含 logits 和特征
            logits, (high_features, low_features,text_features) = predict
            comp_logits, attr_logits, obj_logits = logits
        else:
            # 评估/测试阶段，predict 只包含 logits
            logits = predict
            comp_logits, attr_logits, obj_logits = logits
            high_features = None
            low_features = None
            text_features = None

        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()

        # 计算分类损失
        loss_comp = loss_fn(comp_logits, batch_target)
        loss_attr = loss_fn(attr_logits, batch_attr)
        loss_obj = loss_fn(obj_logits, batch_obj)
        loss = loss_comp * self.config.pair_loss_weight + \
               loss_attr * self.config.attr_loss_weight + \
               loss_obj * self.config.obj_loss_weight


        '''
        # 仅在训练阶段计算协方差损失
        if high_features is not None and low_features is not None and text_features is not None:
           loss_cov= self.local_feature_add(high_features, low_features,text_features)
           #loss += loss_cov * self.config.covariance_loss_weight
           #print(loss_cov)

        '''


            #print("损失不加")

        return loss

    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits = predict
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(comp_logits.shape[-1]):
            weighted_attr_pred = 1 if self.config.attr_inference_weight == 0 else attr_pred[:, pairs[i_comp][
                                                                                                   0]] * self.config.attr_inference_weight
            weighted_obj_pred = 1 if self.config.obj_inference_weight == 0 else obj_pred[:, pairs[i_comp][
                                                                                                1]] * self.config.obj_inference_weight
            comp_logits[:, i_comp] = comp_logits[:,
                                     i_comp] * self.config.pair_inference_weight + weighted_attr_pred * weighted_obj_pred
        return comp_logits

    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_features = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            text_features.append(idx_text_features)
        return text_features

    def forward_for_open(self, batch, text_feats):
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        # l, _ = idx.shape
        batch_img, low_level_features, high_level_features = self.encode_image(batch_img.type(self.clip.dtype))
        att_feature = self.attr_disentangler(batch_img)
        obj_feature = self.obj_disentangler(batch_img)

        batch_img_features = [batch_img, att_feature, obj_feature]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        logits = list()
        for i_element in range(self.token_ids.shape[0]):
            idx_text_features = text_feats[i_element]
            idx_text_features = idx_text_features / idx_text_features.norm(
                dim=-1, keepdim=True
            )
            # CMT
            cmt_text_features = idx_text_features.unsqueeze(0).expand(b, -1, -1)

            # 多阶段交叉注意力
            cmt_1, cmt_2 = self.multi_stage_cross_attention(
                cmt_text_features,
                low_level_features,
                high_level_features

            )
            # cmt1是全局特征交互形成的
            # cmt2是低层局部特征补齐形成的

            # print('weight:',self.stage1_local_alignment.spatial_attn.conv.weight)
            #cmt=self.fusion_module(cmt_1, cmt_2)
            cmt_text_features = idx_text_features + self.lamda * cmt_1+self.lamda_2 * cmt_2
            cmt_text_features = cmt_text_features / cmt_text_features.norm(dim=-1, keepdim=True)

            logits.append(
                torch.einsum(
                    "bd, bkd->bk",
                    normalized_img_features[i_element],
                    cmt_text_features * self.clip.logit_scale.exp()
                )
            )
        return logits

    def forward(self, batch, idx, return_features=False):
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        l, _ = idx.shape

        # 编码图像并提取不同层次的特征
        batch_img, low_level_features, high_level_features = self.encode_image(batch_img.type(self.clip.dtype))

        att_feature = self.attr_disentangler(batch_img)
        obj_feature = self.obj_disentangler(batch_img)

        batch_img_features = [batch_img, att_feature, obj_feature]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        token_tensors = self.construct_token_tensors(idx)

        logits = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )

            cmt_text_features = idx_text_features.unsqueeze(0).expand(b, -1, -1)

            # print(low_level_features.shape)
            # print(high_level_features.shape)
            # print(fine_grained_features.shape)

            # 多阶段交叉注意力
            #print(idx_text_features.shape)

          #low_level_features=self.local_feature_add(high_level_features,low_level_features,idx_text_features)



            #enhanced_high_features=self.local_feature_add(low_level_features,high_level_features,idx_text_features)
            cmt_1, cmt_2 = self.multi_stage_cross_attention(
                cmt_text_features,
                low_level_features,
                high_level_features

            )

            #cmt_text_features = self.fusion_module(cmt_1, cmt_2)
            # Cmt_text_features = cmt
            # print('weight:',self.stage1_local_alignment.spatial_attn.conv.weight)

           # fused_cmt = self.fusion_module(cmt_1, cmt_2)  # 确保维度匹配

            #cmt_text_features = idx_text_features + self.lamda * cmt_2.squeeze(1)
            cmt_text_features = idx_text_features + self.lamda * cmt_1.squeeze(1)+self.lamda_2 * cmt_2.squeeze(1)
            # cmt_text_features= cmt_text_features.squeeze(1)
            cmt_text_features = cmt_text_features / cmt_text_features.norm(dim=-1, keepdim=True)

            logits.append(
                torch.einsum(
                    "bd, bkd->bk",
                    normalized_img_features[i_element],
                    cmt_text_features * self.clip.logit_scale.exp()
                )
            )

        if return_features:
            return logits, (high_level_features, low_level_features, idx_text_features)
        else:
            return logits
