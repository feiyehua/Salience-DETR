import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class NSAAttention(nn.Module):
    """
    原生稀疏注意力实现
    结合了压缩注意力、选择性注意力和滑动窗口注意力三种机制
    """
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.0,
                 block_size: int = 16,        # 压缩注意力的块大小
                 slide_window_size: int = 64, # 滑动窗口大小
                 select_k: int = 8,           # 每个token选择的稀疏连接数量
                 use_gate: bool = True):      # 是否使用门控聚合
        super().__init__()
        assert dim % num_heads == 0, "维度必须能被注意力头数整除"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.slide_window_size = slide_window_size
        self.select_k = select_k
        self.use_gate = use_gate
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 压缩注意力参数
        self.W_K_cmp = nn.Parameter(torch.randn(block_size, 1))
        self.W_V_cmp = nn.Parameter(torch.randn(block_size, 1))
        self.W_pe = nn.Parameter(torch.randn(block_size, dim))
        
        # 门控聚合参数
        if use_gate:
            self.gate_proj = nn.Linear(dim, 3)  # 3种注意力方式的权重
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.constant_(self.v_proj.bias, 0)
        nn.init.constant_(self.out_proj.bias, 0)
        
        # 初始化压缩注意力参数
        nn.init.xavier_uniform_(self.W_K_cmp)
        nn.init.xavier_uniform_(self.W_V_cmp)
        nn.init.normal_(self.W_pe, std=0.02)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询张量 [B, L, C]
            key: 键张量 [B, S, C]
            value: 值张量 [B, S, C]
            key_padding_mask: 键的填充掩码 [B, S]
            
        Returns:
            output: 注意力输出 [B, L, C]
        """
        # 输入检查
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, src_len), \
                f"键填充掩码形状错误: {key_padding_mask.shape}"
            # 将填充掩码转换为float并扩展维度
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).float()  # [B, 1, 1, S]
        
        # 投影查询、键、值
        q = self.q_proj(query)  # [B, L, C]
        k = self.k_proj(key)    # [B, S, C]
        v = self.v_proj(value)  # [B, S, C]
        
        # 重新排列形状以支持多头
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, S, head_dim]
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, S, head_dim]
        
        # 计算三种注意力
        output_cmp = self._compressed_attention(q, k, v, key_padding_mask)
        output_slc = self._selection_attention(q, k, v, key_padding_mask)
        output_win = self._sliding_window_attention(q, k, v, key_padding_mask)
        
        # 门控聚合
        if self.use_gate:
            # 计算门控权重
            gate = self.gate_proj(query)  # [B, L, 3]
            gate = F.sigmoid(gate)
            
            # 聚合三种注意力输出
            output = (gate[:, :, 0:1] * output_cmp + 
                      gate[:, :, 1:2] * output_slc + 
                      gate[:, :, 2:3] * output_win)
        else:
            # 简单平均
            output = (output_cmp + output_slc + output_win) / 3.0
        
        # 最终的线性投影和dropout
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output
    
    def _compressed_attention(self, 
                             q: torch.Tensor, 
                             k: torch.Tensor, 
                             v: torch.Tensor, 
                             key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        压缩注意力实现
        
        压缩注意力将输入序列划分为不重叠的区块，把每个区块压缩为单个令牌，从而减少注意力计算。
        """
        batch_size, num_heads, tgt_len, head_dim = q.shape
        _, _, src_len, _ = k.shape
        
        # 计算压缩方法的所需参数
        stride = self.block_size
        max_idx = (src_len + stride - 1) // stride
        
        # 压缩键和值
        K_cmp_list = []
        V_cmp_list = []
        
        for i in range(max_idx):
            # 获取当前块
            start_idx = i * stride
            end_idx = min(start_idx + self.block_size, src_len)
            actual_size = end_idx - start_idx
            
            if actual_size <= 0:
                continue
                
            # 获取当前块的K和V
            cur_K = k[:, :, start_idx:end_idx, :]  # [B, num_heads, block_size, head_dim]
            cur_V = v[:, :, start_idx:end_idx, :]
            
            # 如果块不足block_size，需要填充
            if actual_size < self.block_size:
                padding = self.block_size - actual_size
                cur_K = F.pad(cur_K, (0, 0, 0, padding))
                cur_V = F.pad(cur_V, (0, 0, 0, padding))
                
            # 添加位置编码
            pe = self.W_pe[:, :head_dim].unsqueeze(0).unsqueeze(0)  # [1, 1, block_size, head_dim]
            cur_K = cur_K + pe
            cur_V = cur_V + pe
            
            # 压缩成单个令牌
            K_cmp = torch.einsum('bhsd,sp->bhpd', cur_K, self.W_K_cmp[:self.block_size])  # [B, num_heads, 1, head_dim]
            V_cmp = torch.einsum('bhsd,sp->bhpd', cur_V, self.W_V_cmp[:self.block_size])  # [B, num_heads, 1, head_dim]
            
            K_cmp_list.append(K_cmp)
            V_cmp_list.append(V_cmp)
        
        # 拼接所有压缩的结果
        K_cmp = torch.cat(K_cmp_list, dim=2)  # [B, num_heads, max_idx, head_dim]
        V_cmp = torch.cat(V_cmp_list, dim=2)  # [B, num_heads, max_idx, head_dim]
        
        # 计算压缩注意力的分数
        scores = torch.einsum('bhqd,bhkd->bhqk', q, K_cmp) / math.sqrt(head_dim)
        
        # 应用掩码（如果有）
        if key_padding_mask is not None:
            # 压缩key_padding_mask以匹配K_cmp
            compressed_mask = key_padding_mask.view(batch_size, 1, 1, src_len)
            # 将掩码分块并取平均
            blocks = []
            for i in range(max_idx):
                start_idx = i * stride
                end_idx = min(start_idx + self.block_size, src_len)
                if start_idx < src_len:
                    block = compressed_mask[:, :, :, start_idx:end_idx].mean(dim=3, keepdim=True)
                    blocks.append(block)
            compressed_mask = torch.cat(blocks, dim=3)
            # 应用掩码
            scores = scores.masked_fill(compressed_mask > 0.5, -1e4)
            
        # 计算注意力权重和输出
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, V_cmp)
        
        # 调整输出形状
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, -1)
        
        return output
    
    def _selection_attention(self, 
                            q: torch.Tensor, 
                            k: torch.Tensor, 
                            v: torch.Tensor, 
                            key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        选择性注意力实现
        
        为每个查询动态选择top-k个最相关的键，只在这些键上计算注意力，实现稀疏连接。
        """
        batch_size, num_heads, tgt_len, head_dim = q.shape
        _, _, src_len, _ = k.shape
        
        # 计算完整注意力分数
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(head_dim)
        
        # 应用掩码（如果有）
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len)
            scores = scores.masked_fill(key_padding_mask > 0.5, -1e4)
        
        # 为每个查询选择top-k键
        k_to_select = min(self.select_k, src_len)
        _, indices = torch.topk(scores, k=k_to_select, dim=-1)  # [B, num_heads, tgt_len, k]
        
        # 收集对应的键和值
        batch_indices = torch.arange(batch_size, device=q.device)[:, None, None, None]
        head_indices = torch.arange(num_heads, device=q.device)[None, :, None, None]
        query_indices = torch.arange(tgt_len, device=q.device)[None, None, :, None]
        
        k_selected = k[batch_indices, head_indices, indices]  # [B, num_heads, tgt_len, k, head_dim]
        v_selected = v[batch_indices, head_indices, indices]  # [B, num_heads, tgt_len, k, head_dim]
        
        # 计算所选键的注意力分数
        selected_scores = torch.einsum('bhqd,bhqkd->bhqk', q, k_selected)
        selected_scores = selected_scores / math.sqrt(head_dim)
        
        # 计算注意力权重和输出
        attn_weights = F.softmax(selected_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.einsum('bhqk,bhqkd->bhqd', attn_weights, v_selected)
        
        # 调整输出形状
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, -1)
        
        return output
    
    def _sliding_window_attention(self, 
                                 q: torch.Tensor, 
                                 k: torch.Tensor, 
                                 v: torch.Tensor, 
                                 key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        滑动窗口注意力实现
        
        限制每个查询只注意其局部窗口内的键，适用于具有局部依赖性的任务。
        """
        batch_size, num_heads, tgt_len, head_dim = q.shape
        _, _, src_len, _ = k.shape
        
        # 如果是自注意力且目标长度等于源长度
        # 为每个位置创建滑动窗口掩码
        if tgt_len == src_len:  # 假设是自注意力
            # 创建相对位置索引
            q_idx = torch.arange(tgt_len, device=q.device)
            k_idx = torch.arange(src_len, device=k.device)
            
            # 计算相对位置
            rel_pos = q_idx.unsqueeze(1) - k_idx.unsqueeze(0)  # [tgt_len, src_len]
            
            # 创建窗口掩码：只有在窗口内的位置才能相互关注
            window_size = min(self.slide_window_size, tgt_len)
            window_mask = (rel_pos.abs() <= window_size // 2)
            
            # 扩展窗口掩码
            window_mask = window_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
            
            # 计算注意力分数
            scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(head_dim)
            
            # 应用窗口掩码和填充掩码
            scores = scores.masked_fill(~window_mask, -1e4)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len)
                scores = scores.masked_fill(key_padding_mask > 0.5, -1e4)
                
            # 计算注意力权重和输出
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)
            
            # 调整输出形状
            output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, -1)
            
        else:
            # 如果不是自注意力（如交叉注意力），回退到全注意力
            scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(head_dim)
            
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len)
                scores = scores.masked_fill(key_padding_mask > 0.5, -1e4)
                
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)
            
            output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, -1)
            
        return output 