import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional, Union

from models.bricks.basic import MLP
import sys
import os

# 导入NSA稀疏注意力机制 - 直接从本地导入

from models.bricks.nsa_attention import NSAAttention

from torch import optim


class UncertaintyEstimator(nn.Module):
    """
    不确定性评估模块，用于量化模型对RoI区域的检测置信度。
    使用Monte Carlo Dropout或集成方法来近似贝叶斯推理。
    """
    def __init__(self, 
                 in_channels: int, 
                 hidden_dim: int = 256, 
                 num_mc_samples: int = 5,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.num_mc_samples = num_mc_samples
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 不确定性评估头
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        评估特征的不确定性
        
        Args:
            features: 输入的多尺度特征
            
        Returns:
            uncertainty: 每个位置的不确定性评分
        """
        # 启用dropout进行Monte Carlo采样
        self.train()  # 强制启用dropout
        
        # 进行多次前向传播以获取采样
        predictions = []
        for _ in range(self.num_mc_samples):
            # 提取特征
            x = self.feature_extractor(features)
            # 预测
            pred = self.uncertainty_head(x)
            predictions.append(pred)
            
        # 堆叠所有预测
        stacked_preds = torch.stack(predictions, dim=0)  # [num_samples, B, 1, H, W]
        
        # 计算预测方差作为不确定性度量
        mean_pred = torch.mean(stacked_preds, dim=0)  # [B, 1, H, W]
        variance = torch.mean((stacked_preds - mean_pred.unsqueeze(0))**2, dim=0)  # [B, 1, H, W]
        
        # 归一化不确定性
        uncertainty = F.sigmoid(variance)  # 将不确定性映射到 [0, 1]
        
        return uncertainty


class MultiScaleRoIGenerator(nn.Module):
    """
    多尺度RoI生成器，从不同尺度的特征图生成RoI
    
    实现多尺度RoI生成，并基于显著性和不确定性进行筛选
    """
    def __init__(self, 
                 in_channels_list: List[int],
                 hidden_dim: int = 256,
                 num_scales: int = 4,
                 top_k_per_level: List[int] = [100, 100, 100, 100],
                 uncertainty_weight: float = 0.5):
        super().__init__()
        
        self.num_scales = num_scales
        self.top_k_per_level = top_k_per_level
        self.uncertainty_weight = uncertainty_weight
        
        # 显著性预测器
        self.salience_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.Sigmoid()
            ) for in_channels in in_channels_list
        ])
        
        # 不确定性估计器
        self.uncertainty_estimators = nn.ModuleList([
            UncertaintyEstimator(in_channels, hidden_dim)
            for in_channels in in_channels_list
        ])
        
    def forward(self, features_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        生成多尺度RoI
        
        Args:
            features_list: 不同尺度的特征图列表 [P2, P3, P4, P5]
            
        Returns:
            roi_features_list: 选出的RoI特征列表
            roi_indices_list: RoI索引列表
            roi_scores_list: RoI得分列表
        """
        roi_features_list = []
        roi_indices_list = []
        roi_scores_list = []
        
        for level_idx, features in enumerate(features_list):
            # 获取显著性分数
            salience_score = self.salience_predictors[level_idx](features)
            
            # 获取不确定性分数
            uncertainty_score = self.uncertainty_estimators[level_idx](features)
            
            # 结合显著性和不确定性计算综合得分
            # 高显著性或高不确定性区域都应被保留
            composite_score = (1 - self.uncertainty_weight) * salience_score + self.uncertainty_weight * uncertainty_score
            
            # 展平特征和分数以便于选择
            B, C, H, W = features.shape
            flat_features = features.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, H*W, C]
            flat_scores = composite_score.reshape(B, -1)  # [B, H*W]
            
            # 为每个样本选择top-k的索引
            top_k = min(self.top_k_per_level[level_idx], flat_scores.shape[1])
            batch_roi_features = []
            batch_roi_indices = []
            batch_roi_scores = []
            
            for b in range(B):
                scores = flat_scores[b]
                top_values, top_indices = torch.topk(scores, k=top_k)
                
                # 选择对应的特征
                selected_features = flat_features[b, top_indices]  # [top_k, C]
                
                batch_roi_features.append(selected_features)
                batch_roi_indices.append(top_indices)
                batch_roi_scores.append(top_values)
            
            # 堆叠批次结果
            roi_features = torch.stack(batch_roi_features)  # [B, top_k, C]
            roi_indices = torch.stack(batch_roi_indices)    # [B, top_k]
            roi_scores = torch.stack(batch_roi_scores)      # [B, top_k]
            
            # 添加位置编码
            pos_encoding = self.generate_position_encoding(roi_indices, H, W)
            roi_features = roi_features + pos_encoding
            
            roi_features_list.append(roi_features)
            roi_indices_list.append(roi_indices)
            roi_scores_list.append(roi_scores)
            
        return roi_features_list, roi_indices_list, roi_scores_list
    
    def generate_position_encoding(self, indices: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        生成位置编码
        
        Args:
            indices: RoI的扁平索引 [B, num_rois]
            height: 特征图高度
            width: 特征图宽度
            
        Returns:
            position_encoding: RoI的位置编码 [B, num_rois, C]
        """
        # 从平坦索引计算坐标
        y = (indices // width).float() / height  # 归一化行坐标
        x = (indices % width).float() / width    # 归一化列坐标
        
        B, num_rois = indices.shape
        C = 256  # 与特征维度相同
        
        # 生成简单的位置编码
        position_encoding = torch.zeros((B, num_rois, C), device=indices.device)
        
        # 偶数索引使用正弦编码，奇数索引使用余弦编码
        for i in range(C // 4):
            freq = 1.0 / (10000 ** (4 * i / C))
            position_encoding[:, :, 4*i] = torch.sin(x * freq)
            position_encoding[:, :, 4*i+1] = torch.cos(x * freq)
            position_encoding[:, :, 4*i+2] = torch.sin(y * freq)
            position_encoding[:, :, 4*i+3] = torch.cos(y * freq)
            
        return position_encoding


class AdaptiveRoIProcessor(nn.Module):
    """
    自适应RoI处理器，实现RoI交互和尺寸调整
    
    使用NSA稀疏注意力机制进行RoI之间的交互，并预测最优的RoI尺寸
    """
    def __init__(self, 
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_roi_layers: int = 2,
                 use_nsa: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_nsa = use_nsa
        
        # RoI特征融合
        self.roi_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # RoI交互层
        self.roi_interaction_layers = nn.ModuleList([
            NSAAttention(embed_dim, num_heads, dropout) if use_nsa else
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_roi_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_roi_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_roi_layers)
        ])
        
        self.norm_ffn_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_roi_layers)
        ])
        
        # 自适应尺寸预测
        self.size_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 2)  # 预测宽高缩放因子
        )
        
    def forward(self, 
                multi_scale_rois: List[torch.Tensor], 
                multi_scale_scores: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理多尺度RoI，进行交互并预测自适应尺寸
        
        Args:
            multi_scale_rois: 多尺度RoI特征列表
            multi_scale_scores: 多尺度RoI分数列表
            
        Returns:
            processed_rois: 处理后的RoI特征
            roi_scale_factors: RoI尺寸调整因子
        """
        B = multi_scale_rois[0].shape[0]
        
        # 融合多尺度RoI，加权求和
        fused_rois = []
        for b in range(B):
            batch_rois = []
            batch_scores = []
            
            for level_rois, level_scores in zip(multi_scale_rois, multi_scale_scores):
                batch_rois.append(level_rois[b])
                batch_scores.append(level_scores[b])
            
            # 拼接所有尺度的RoI
            all_rois = torch.cat(batch_rois, dim=0)
            all_scores = torch.cat(batch_scores, dim=0)
            
            # 对分数进行Softmax归一化作为权重
            weights = F.softmax(all_scores, dim=0).unsqueeze(1)
            
            # 加权融合
            fused_roi = self.roi_fusion(all_rois * weights)
            fused_rois.append(fused_roi)
            
        # 堆叠批次结果
        rois = torch.stack(fused_rois)  # [B, num_rois, C]
        
        # RoI交互
        output = rois
        for i, (attn_layer, norm_layer, ffn_layer, norm_ffn_layer) in enumerate(
            zip(self.roi_interaction_layers, self.norm_layers, self.ffn_layers, self.norm_ffn_layers)
        ):
            # 自注意力交互
            if self.use_nsa:
                # NSA稀疏注意力
                attn_output = attn_layer(output, output, output)
            else:
                # 标准多头注意力
                attn_output = attn_layer(output, output, output)[0]
                
            # 残差连接与层归一化
            output = norm_layer(output + attn_output)
            
            # 前馈网络
            ffn_output = ffn_layer(output)
            output = norm_ffn_layer(output + ffn_output)
        
        # 预测自适应尺寸
        roi_scale_factors = self.size_predictor(output)  # [B, num_rois, 2]
        roi_scale_factors = torch.sigmoid(roi_scale_factors) * 2.0  # 输出范围为 [0, 2]
        
        return output, roi_scale_factors


class UncertaintyRoIScreening(nn.Module):
    """
    综合模块，整合多尺度RoI生成和自适应RoI处理
    """
    def __init__(self,
                 in_channels_list: List[int],
                 hidden_dim: int = 256,
                 num_scales: int = 4,
                 top_k_per_level: List[int] = [200, 150, 100, 50],
                 uncertainty_weight: float = 0.5,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_roi_layers: int = 2,
                 use_nsa: bool = True):
        super().__init__()
        
        self.multi_scale_roi_generator = MultiScaleRoIGenerator(
            in_channels_list=in_channels_list,
            hidden_dim=hidden_dim,
            num_scales=num_scales,
            top_k_per_level=top_k_per_level,
            uncertainty_weight=uncertainty_weight
        )
        
        self.adaptive_roi_processor = AdaptiveRoIProcessor(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_roi_layers=num_roi_layers,
            use_nsa=use_nsa
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features_list: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        执行完整的RoI处理流程
        
        Args:
            features_list: 不同尺度的特征图列表 [P2, P3, P4, P5]
            
        Returns:
            processed_rois: 处理后的RoI特征
            roi_indices_list: 各尺度RoI索引列表
            roi_scale_factors: RoI尺寸调整因子
        """
        # 生成多尺度RoI
        roi_features_list, roi_indices_list, roi_scores_list = self.multi_scale_roi_generator(features_list)
        
        # 处理RoI
        processed_rois, roi_scale_factors = self.adaptive_roi_processor(roi_features_list, roi_scores_list)
        
        # 最终融合
        processed_rois = self.final_fusion(processed_rois)
        
        return processed_rois, roi_indices_list, roi_scale_factors 