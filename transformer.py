import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to the input embeddings.
    Helps the Transformer understand the order of frames in the sequence.
    """
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model) for broadcasting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence embeddings. Shape: (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: Input with positional embeddings added.
        """
        return x + self.pe[:, :x.size(1), :]


class TimeEmbedding(nn.Module):
    """
    Transforms a scalar time value t into a high-dimensional embedding.
    """
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), # Input is a scalar t (t.unsqueeze(1))
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): Time scalar. Shape: (batch_size, 1) or (batch_size,)
        Returns:
            torch.Tensor: Time embedding. Shape: (batch_size, d_model)
        """
        if t.dim() == 1:  # Ensure t is (batch_size, 1)
            t = t.unsqueeze(1)
        return self.mlp(t)


# --- Main Flow Matching Transformer Model ---
class FlowMatchingTransformer(nn.Module):
    @classmethod
    def from_config(cls, config):
        return FlowMatchingTransformer(num_frames=config["num_frames"],
                                       num_joints=config["num_joints"],
                                       joint_dim=config["joint_dim"],
                                       d_model=config["d_model"],
                                       num_head=config["num_head"],
                                       num_encoder_layers=config["num_encoder_layers"],
                                       dim_feedforward=config["dim_feedforward"],
                                       dropout=config["dropout"])

    def __init__(self,
                 num_frames: int,
                 num_joints: int = 23,
                 joint_dim: int = 9,
                 d_model: int = 512,
                 num_head: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.d_model = d_model

        # 1. Input embedding for flattened action data (per frame)
        # Each frame (num_joints * joint_dim) is projected to d_model
        self.input_projection = nn.Linear(num_joints * joint_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, num_frames)

        # 2. Time embedding module
        self.time_embedding_module = TimeEmbedding(d_model)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: input tensors are (batch_size, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 4. Output head for velocity prediction
        # Projects the Transformer output (d_model) back to flattened velocity (num_joints * joint_dim)
        self.output_projection = nn.Linear(d_model, num_joints * joint_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predicts the velocity field for a given action sequence at time t.

        Args:
            x_t (torch.Tensor): The action sequence at time t (interpolated from x0 and x1).
                                Shape: (batch_size, num_frames, num_joints, joint_dim)
            t (torch.Tensor): The scalar time value for each sample in the batch.
                              Shape: (batch_size,) or (batch_size, 1) - float between 0 and 1.
        Returns:
            torch.Tensor: The predicted velocity field for the entire sequence.
                          Shape: (batch_size, num_frames, num_joints, joint_dim)
        """
        batch_size = x_t.shape[0]

        # 1. Prepare action sequence x_t for Transformer input
        # Flatten num_joints and joint_dim into a single feature dimension per frame
        x_t_flat = x_t.view(batch_size, self.num_frames, -1)  # Shape: (batch_size, num_frames, num_joints * joint_dim)

        # Project flattened features to Transformer's d_model dimension
        x_t_embedded = self.input_projection(x_t_flat)  # Shape: (batch_size, num_frames, d_model)

        # Add positional encoding to capture frame order
        x_t_with_pos = self.pos_encoder(x_t_embedded)  # Shape: (batch_size, num_frames, d_model)

        # 2. Generate time embedding
        time_emb = self.time_embedding_module(t)  # Shape: (batch_size, d_model)

        # 3. Integrate time embedding into the sequence
        # We add the time_emb (broadcasting it) to every frame's embedding
        # This makes the Transformer aware of the global time t for the entire sequence
        input_to_transformer = x_t_with_pos + time_emb.unsqueeze(1)  # Shape: (batch_size, num_frames, d_model)

        # 4. Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(input_to_transformer)  # Shape: (batch_size, num_frames, d_model)

        # 5. Project Transformer output back to the velocity field dimension
        # The output head predicts the flattened velocity for each frame
        velocity_flat = self.output_projection(transformer_output)  # Shape: (batch_size, num_frames, num_joints * joint_dim)

        # Reshape back to the original (num_frames, num_joints, joint_dim) structure
        velocity_pred = velocity_flat.view(batch_size, self.num_frames, self.num_joints, self.joint_dim)

        return velocity_pred


def flow(net: nn.Module, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
    t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
    return x_t + (t_end - t_start) * net(t=t_start + (t_end - t_start) / 2,
                                         x_t=x_t + net(x_t=x_t, t=t_start) * (t_end - t_start) / 2)

