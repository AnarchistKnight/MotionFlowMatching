import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# TimeEmbedding
class TimeEmbedding(nn.Module):
    """
    一个简单的线性层来嵌入时间索引 (frame_idx)。
    将 frame_idx 映射到 d_model 维度，以便与动作特征融合。
    """
    def __init__(self, d_model: int, max_frames: int = 5000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model), # 输入是 (batch_size, 1), 映射到 (batch_size, d_model)
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, t: torch.Tensor):
        """
        Args:
            t: 每一帧的索引，形状为 (batch_size, 1)。
        Returns:
            时间嵌入向量，形状为 (batch_size, d_model)。
        """
        return self.mlp(t.float()) # 将 t 转换为浮点数输入线性层


class MotionTransformer(nn.Module):
    """
    一个用于角色动作生成的速度场预测Transformer模型。
    它接收一帧动作数据和对应的帧索引，输出该帧的速度场。
    """
    def __init__(self,
                 num_joints: int = 23,
                 rot6d_dim: int = 6,
                 pos3d_dim: int = 3,
                 d_model: int = 256,
                 num_head: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_frames: int = 5000):
        super().__init__()

        self.num_joints = num_joints
        self.input_feature_dim = rot6d_dim + pos3d_dim # 6 + 3 = 9
        self.d_model = d_model

        # 输入编码层：将每帧的动作数据映射到d_model维度
        # 输入形状：(batch_size, num_joints, input_feature_dim)
        # 展平为 (batch_size, num_joints * input_feature_dim)
        # 然后线性映射到 (batch_size, d_model)
        self.input_projection = nn.Linear(self.num_joints * self.input_feature_dim, d_model)

        # 时间嵌入层：为 frame_idx 添加编码
        self.time_embedding = TimeEmbedding(d_model, max_frames=max_frames)

        # Transformer编码器
        # 注意: TransformerEncoderLayer 默认 batch_first=False，期望输入是 (seq_len, batch_size, feature_dim)
        # 但如果你的输入每一帧都是独立的，可以考虑 batch_first=True
        # 在这里，由于我们处理的是单帧，seq_len=1，所以手动 unsqueeze(0) 或 squeeze(0) 调整维度。
        # 如果你希望直接传入 (batch_size, 1, d_model) 给 encoder，可以设 batch_first=True
        # 为了符合 PyTorch 官方示例和通常做法，我们保持 batch_first=False
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=num_head,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True) # 修改为 batch_first=True 以简化输入处理
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 输出层：将Transformer的输出映射回速度场维度
        # 速度场的维度与输入动作数据相同：num_joints * input_feature_dim
        self.output_projection = nn.Linear(d_model, self.num_joints * self.input_feature_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: 单帧动作数据，形状为 (batch_size, num_joints, rot6d_dim + pos3d_dim)。
               例如：(B, J, 9)
            t: 当前帧的索引，形状为 (batch_size, 1)。
               每个batch样本对应一个帧索引。
        Returns:
            predicted_velocity_field: 预测的速度场，形状为 (batch_size, num_joints, rot6d_dim + pos3d_dim)。
        """
        batch_size = x.shape[0]

        # 1. 输入编码
        # 将 (batch_size, num_joints, input_feature_dim) 展平为 (batch_size, num_joints * input_feature_dim)
        flat_x = x.view(batch_size, -1)
        # 线性映射到 (batch_size, d_model)
        encoded_x = self.input_projection(flat_x) # (batch_size, d_model)

        # 2. 时间嵌入
        # 将 t (frame_idx) 映射到 (batch_size, d_model)
        time_emb = self.time_embedding(t) # (batch_size, d_model)

        # 3. 融合动作特征和时间嵌入
        # 将两者相加，作为Transformer的输入。
        # 如果你希望更复杂的融合方式，例如拼接后通过线性层，也可以。
        fused_input = encoded_x + time_emb # (batch_size, d_model)

        # 4. Transformer编码
        # 由于我们每一帧作为一个独立的序列元素 (seq_len=1)，
        # 并且设置了 batch_first=True，所以输入形状是 (batch_size, seq_len, d_model)
        fused_input = fused_input.unsqueeze(1) # 变为 (batch_size, 1, d_model)

        transformer_output = self.transformer_encoder(fused_input) # (batch_size, 1, d_model)

        # 5. 输出映射
        # 移除 seq_len 维度 (1)
        transformer_output = transformer_output.squeeze(1) # (batch_size, d_model)
        # 映射回速度场维度
        predicted_velocity_field_flat = self.output_projection(transformer_output) # (batch_size, num_joints * input_feature_dim)
        # 恢复到原始动作数据形状
        predicted_velocity_field = predicted_velocity_field_flat.view(batch_size, self.num_joints, self.input_feature_dim)

        return predicted_velocity_field
