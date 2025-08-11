import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .ram import RAM
from .utils import AsymmetricLoss
from collections import OrderedDict

class MultiTaskTaggingLoss(nn.Module):
    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        
        # 不同层次标签的损失函数
        self.global_loss = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)
        self.local_loss = AsymmetricLoss(gamma_neg=5, gamma_pos=1, clip=0.03)
        self.relation_loss = RelationLoss()
        
        # 一致性损失
        self.consistency_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict containing 'global', 'local', 'relation' logits
            targets: dict containing corresponding ground truth labels
        """
        total_loss = 0
        loss_dict = {}
        
        # 全局标签损失
        if 'global' in predictions:
            global_loss = self.global_loss(predictions['global'], targets['global'])
            total_loss += self.loss_weights['global'] * global_loss
            loss_dict['global_loss'] = global_loss
            
        # 局部标签损失
        if 'local' in predictions:
            local_loss = self.local_loss(predictions['local'], targets['local'])
            total_loss += self.loss_weights['local'] * local_loss
            loss_dict['local_loss'] = local_loss
            
        # 关系标签损失
        if 'relation' in predictions:
            relation_loss = self.relation_loss(predictions['relation'], targets['relation'])
            total_loss += self.loss_weights['relation'] * relation_loss
            loss_dict['relation_loss'] = relation_loss
            
        # 层次一致性损失：确保不同层次标签的一致性
        if 'global' in predictions and 'local' in predictions:
            global_prob = torch.sigmoid(predictions['global'])
            local_prob = torch.sigmoid(predictions['local'])
            
            # 全局标签应该与局部标签的聚合保持一致
            consistency_loss = self.consistency_loss(
                global_prob, 
                torch.max(local_prob, dim=-1)[0].unsqueeze(-1).expand_as(global_prob)
            )
            total_loss += self.loss_weights['consistency'] * consistency_loss
            loss_dict['consistency_loss'] = consistency_loss
            
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

class RelationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, relation_logits, relation_targets):
        # 关系标签的特殊损失计算
        return self.bce_loss(relation_logits, relation_targets)

class CrossScaleAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, global_feat, balanced_feat, local_feat):
        # 将不同尺度特征作为query, key, value
        B, N_g, D = global_feat.shape
        B, N_b, D = balanced_feat.shape
        B, N_l, D = local_feat.shape
        
        # 全局特征作为query，其他作为key和value
        q = self.q_proj(global_feat).view(B, N_g, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 拼接balanced和local特征作为key和value
        kv_feat = torch.cat([balanced_feat, local_feat], dim=1)  # [B, N_b+N_l, D]
        k = self.k_proj(kv_feat).view(B, N_b+N_l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_feat).view(B, N_b+N_l, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_g, D)
        
        return self.out_proj(attn_output)

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, vision_width, scales=[224, 336, 448]):
        super().__init__()
        self.scales = scales
        
        # 不同尺度的特征投影层
        self.scale_projectors = nn.ModuleDict({
            f'scale_{scale}': nn.Sequential(
                nn.Linear(vision_width, 1536),
                nn.GELU(),
                nn.Linear(1536, 1536),
                nn.LayerNorm(1536)
            ) for scale in scales
        })
        
        # 跨尺度注意力融合
        self.cross_scale_attention = CrossScaleAttention(1536, num_heads=8)
        
        # 尺度特定的池化策略
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局场景理解
        self.local_pool = nn.AdaptiveMaxPool1d(64)  # 局部对象检测
        
    def forward(self, image_features, scale_info):
        """
        Args:
            image_features: [B, N, D] 来自视觉编码器的特征
            scale_info: 当前处理的尺度信息
        """
        batch_size = image_features.shape[0]
        
        # 多尺度特征提取
        multiscale_features = {}
        
        for scale in self.scales:
            # 根据尺度调整特征
            if scale == 224:  # 低分辨率 - 全局特征
                pooled_features = self.global_pool(image_features.transpose(1, 2)).transpose(1, 2)
                projected = self.scale_projectors[f'scale_{scale}'](pooled_features)
                multiscale_features['global'] = projected
                
            elif scale == 336:  # 中分辨率 - 平衡特征
                projected = self.scale_projectors[f'scale_{scale}'](image_features)
                multiscale_features['balanced'] = projected
                
            elif scale == 448:  # 高分辨率 - 局部特征
                local_features = self.local_pool(image_features.transpose(1, 2)).transpose(1, 2)
                projected = self.scale_projectors[f'scale_{scale}'](local_features)
                multiscale_features['local'] = projected
        
        # 跨尺度特征融合
        fused_features = self.cross_scale_attention(
            multiscale_features['global'],
            multiscale_features['balanced'], 
            multiscale_features['local']
        )
        
        return fused_features, multiscale_features

class TaggingHead(nn.Module):
    def __init__(self, hidden_size, num_classes, tag_type):
        super().__init__()
        self.tag_type = tag_type
        
        # 不同类型标签的特定处理
        if tag_type == "global":
            # 全局标签：关注整体场景
            self.feature_processor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2)
            )
        elif tag_type == "local":
            # 局部标签：关注具体对象
            self.feature_processor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_size, hidden_size // 2)
            )
            
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, features):
        processed_features = self.feature_processor(features)
        # 对于全局标签，使用平均池化；对于局部标签，使用最大池化
        if self.tag_type == "global":
            pooled = torch.mean(processed_features, dim=1)
        else:
            pooled = torch.max(processed_features, dim=1)[0]
            
        logits = self.classifier(pooled)
        return logits

class RelationTaggingHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        
        # 关系建模：对象-动作-对象三元组
        self.subject_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.predicate_proj = nn.Linear(hidden_size, hidden_size // 2) 
        self.object_proj = nn.Linear(hidden_size, hidden_size // 2)
        
        # 关系分类器
        self.relation_classifier = nn.Linear(hidden_size // 2 * 3, num_classes)
        
        # 空间注意力机制
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True
        )
        
    def forward(self, features):
        B, N, D = features.shape
        
        # 空间注意力增强特征
        enhanced_features, _ = self.spatial_attention(features, features, features)
        
        # 提取主体、谓词、客体特征
        subject_feat = self.subject_proj(enhanced_features)
        predicate_feat = self.predicate_proj(enhanced_features)
        object_feat = self.object_proj(enhanced_features)
        
        # 组合关系特征
        relation_feat = torch.cat([
            torch.mean(subject_feat, dim=1),
            torch.mean(predicate_feat, dim=1),
            torch.mean(object_feat, dim=1)
        ], dim=-1)
        
        relation_logits = self.relation_classifier(relation_feat)
        return relation_logits

class MultiScaleRAM(RAM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 多尺度特征提取器
        self.multiscale_extractor = MultiScaleFeatureExtractor(
            vision_width=1024,  # CLIP-L特征维度
            scales=[224, 336, 448]  # 低、中、高分辨率
        )
        
        # 分层标签预测头
        self.global_head = TaggingHead(1536, self.num_class, "global")
        self.local_head = TaggingHead(1536, self.num_class, "local")
        self.relation_head = RelationTaggingHead(1536, self.num_class)
        
        # 多任务损失权重
        self.loss_weights = {
            'global': 1.0,
            'local': 1.0, 
            'relation': 0.8,
            'consistency': 0.5
        }
        
    def forward(self, image, caption=None, global_tags=None, local_tags=None, relation_tags=None, clip_feature=None):
        """
        多尺度多任务前向传播
        """
        # 1. 多尺度特征提取
        image_features = self.visual_encoder(image)
        fused_features, multiscale_features = self.multiscale_extractor(image_features, None)
        
        # 2. 分层标签预测
        predictions = {}
        
        # 全局标签预测（场景级别）
        global_logits = self.global_head(multiscale_features['global'])
        predictions['global'] = global_logits
        
        # 局部标签预测（对象级别）
        local_logits = self.local_head(multiscale_features['local'])
        predictions['local'] = local_logits
        
        # 关系标签预测（事件级别）
        relation_logits = self.relation_head(fused_features)
        predictions['relation'] = relation_logits
        
        # 3. 损失计算（训练时）
        if self.training and global_tags is not None:
            targets = {
                'global': global_tags,
                'local': local_tags,
                'relation': relation_tags
            }
            
            multitask_loss = MultiTaskTaggingLoss(self.loss_weights)
            total_loss, loss_dict = multitask_loss(predictions, targets)
            
            # CLIP蒸馏损失
            if clip_feature is not None:
                image_cls_embeds = fused_features[:, 0, :]
                loss_dis = F.l1_loss(image_cls_embeds, clip_feature)
                total_loss += 0.1 * loss_dis
                loss_dict['distillation_loss'] = loss_dis
                
            return total_loss, loss_dict
        
        return predictions
    
    def generate_hierarchical_tags(self, image, threshold_config=None):
        """
        生成分层标签：全局、局部、关系
        """
        if threshold_config is None:
            threshold_config = {
                'global': 0.7,
                'local': 0.6, 
                'relation': 0.5
            }
            
        with torch.no_grad():
            predictions = self.forward(image)
            
            # 解析不同层次的标签
            results = {
                'global_tags': self._parse_tags(predictions['global'], threshold_config['global'], 'global'),
                'local_tags': self._parse_tags(predictions['local'], threshold_config['local'], 'local'),
                'relation_tags': self._parse_relation_tags(predictions['relation'], threshold_config['relation'])
            }
            
            return results
    
    def _parse_tags(self, logits, threshold, tag_type):
        """解析标签预测结果"""
        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).float()
        
        batch_results = []
        for b in range(predictions.shape[0]):
            indices = torch.where(predictions[b] == 1)[0]
            if tag_type == 'global':
                tags = [self.tag_list[idx] for idx in indices if 'scene' in self.tag_list[idx] or 'environment' in self.tag_list[idx]]
            else:
                tags = [self.tag_list[idx] for idx in indices]
            batch_results.append(tags)
            
        return batch_results
    
    def _parse_relation_tags(self, logits, threshold):
        """解析关系标签"""
        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).float()
        
        batch_results = []
        for b in range(predictions.shape[0]):
            indices = torch.where(predictions[b] == 1)[0]
            # 构建关系三元组
            relations = []
            for idx in indices:
                if 'action' in self.tag_list[idx] or 'interaction' in self.tag_list[idx]:
                    relations.append(self.tag_list[idx])
            batch_results.append(relations)
            
        return batch_results