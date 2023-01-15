import math
from functools import partial
from typing import Callable
from typing import List, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models.vision_transformer import EncoderBlock
from typing_extensions import OrderedDict


class Encoder(nn.Module):
    """
    Transformer Model Encoder for sequence to sequence translation.
    Code from torch.
    """
    def __init__(
            self,
            pos_embedding: Tensor,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, x: Tensor):
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        x = x + self.pos_embedding
        return self.ln(self.layers(self.dropout(x)))


class TubeViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        video_shape: Union[List[int], np.ndarray],
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        representation_size=None,
    ):
        super(TubeViT, self).__init__()
        self.video_shape = np.array(video_shape)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.kernel_sizes = (
            (8, 8, 8),
            (16, 4, 4),
            (4, 12, 12),
            (1, 16, 16),
        )

        self.strides = (
            (16, 32, 32),
            (6, 32, 32),
            (16, 32, 32),
            (32, 16, 16),
        )

        self.offsets = (
            (0, 0, 0),
            (4, 8, 8),
            (0, 16, 16),
            (0, 0, 0),
        )

        self.conv_proj_weight = nn.Parameter(torch.empty((self.hidden_dim, 3, *self.kernel_sizes[0])).normal_(),
                                             requires_grad=True)

        self.register_parameter('conv_proj_weight', self.conv_proj_weight)

        self.conv_proj_bias = nn.Parameter(torch.zeros(len(self.kernel_sizes), self.hidden_dim), requires_grad=True)
        self.register_parameter('conv_proj_bias', self.conv_proj_bias)

        self.pos_embedding = nn.Parameter(self._generate_position_embedding(), requires_grad=False)
        self.register_parameter('pos_embedding', self.pos_embedding)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.encoder = Encoder(pos_embedding=self.pos_embedding,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               hidden_dim=self.hidden_dim,
                               mlp_dim=mlp_dim,
                               dropout=dropout,
                               attention_dropout=attention_dropout)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(self.hidden_dim, self.num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(self.hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, self.num_classes)

        self.heads = nn.Sequential(heads_layers)

    def _process_input(self, x: Tensor) -> Tensor:
        n, c, t, h, w = x.shape
        tubes = []
        for i in range(len(self.kernel_sizes)):
            if i == 0:
                weight = self.conv_proj_weight
            else:
                weight = F.interpolate(self.conv_proj_weight, self.kernel_sizes[i], mode='trilinear')

            tube = F.conv3d(
                x[:, :, self.offsets[i][0]:, self.offsets[i][1]:, self.offsets[i][2]:],
                weight,
                bias=self.conv_proj_bias[i],
                stride=self.strides[i],
            )

            tube = tube.reshape((n, self.hidden_dim, -1))

            tubes.append(tube)

        x = torch.cat(tubes, dim=-1)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

    def _calc_conv_shape(self, kernel_size, stride, offset) -> np.ndarray:
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)

        output = np.ceil((self.video_shape[1:] - offset - kernel_size + 1) / stride).astype(int)
        output = np.concatenate([np.array([self.hidden_dim / 6]), output]).astype(int)

        return output

    def _generate_position_embedding(self) -> Tensor:
        def _position_embedding_code(t, x, y, j, tau=100_000) -> Tensor:
            w = 1 / (tau**j)
            p_jt = math.sin(t * w), math.cos(t * w)
            p_jx = math.sin(x * w), math.cos(x * w)
            p_jy = math.sin(y * w), math.cos(y * w)
            return Tensor([*p_jt, *p_jx, *p_jy])

        position_embedding = [torch.zeros(self.hidden_dim, 1)]

        for i in range(len(self.kernel_sizes)):
            tube_shape = self._calc_conv_shape(self.kernel_sizes[i], self.strides[i], self.offsets[i])
            tmp = torch.zeros([self.hidden_dim, *tube_shape[1:]])
            for j in range(tube_shape[0]):
                for t in range(tube_shape[1]):
                    for x in range(tube_shape[2]):
                        for w in range(tube_shape[3]):
                            tmp[6 * j:6 * (j + 1), t, x, w] = _position_embedding_code(
                                t=t * self.strides[i][0] + self.offsets[i][0] + self.kernel_sizes[i][0] // 2,
                                x=x * self.strides[i][1] + self.offsets[i][1] + self.kernel_sizes[i][1] // 2,
                                y=w * self.strides[i][2] + self.offsets[i][2] + self.kernel_sizes[i][2] // 2,
                                j=j,
                            )
            tmp = tmp.reshape((self.hidden_dim, -1))
            position_embedding.append(tmp)

        position_embedding = torch.cat(position_embedding, dim=-1)
        position_embedding = position_embedding.permute(1, 0)
        return position_embedding
