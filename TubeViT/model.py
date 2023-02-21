import math
from functools import partial
from typing import Callable
from typing import List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor, optim
from torch.nn import functional as F
from torchmetrics.functional import accuracy, f1_score
from torchvision.models.vision_transformer import EncoderBlock
from typing_extensions import OrderedDict


class Encoder(nn.Module):
    """
    Transformer Model Encoder for sequence to sequence translation.
    Code from torch.
    Move pos_embedding to TubeViT
    """
    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
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
        return self.ln(self.layers(self.dropout(x)))


class SparseTubesTokenizer(nn.Module):
    def __init__(self, hidden_dim: int, kernel_sizes, strides, offsets, space_to_depth_rate: int = 1):
        super().__init__()
        self.space_to_depth_rate = space_to_depth_rate
        self.hidden_dim = int(hidden_dim // space_to_depth_rate)
        self.kernel_sizes = kernel_sizes
        for i in range(len(strides)):
            strides[i][0] = int(strides[i][0] // space_to_depth_rate)

        self.strides = strides
        self.offsets = offsets

        self.conv_proj_weight = nn.Parameter(torch.empty((self.hidden_dim, 3, *self.kernel_sizes[0])).normal_(),
                                             requires_grad=True)

        self.register_parameter('conv_proj_weight', self.conv_proj_weight)

        self.conv_proj_bias = nn.Parameter(torch.zeros(len(self.kernel_sizes), self.hidden_dim), requires_grad=True)
        self.register_parameter('conv_proj_bias', self.conv_proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, t, h, w = x.shape  # CTHW
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
        x = x.permute(0, 2, 1).contiguous()
        return x


class TubeViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        video_shape: Union[List[int], np.ndarray],  # CTHW
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        representation_size=None,
        space_to_depth_rate=1,
    ):
        super(TubeViT, self).__init__()
        self.video_shape = np.array(video_shape)  # CTHW
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.kernel_sizes = [
            [8, 8, 8],
            [16, 4, 4],
            [4, 12, 12],
            [1, 16, 16],
        ]

        self.strides = [
            [16, 32, 32],
            [6, 32, 32],
            [16, 32, 32],
            [32, 16, 16],
        ]

        self.offsets = [
            [0, 0, 0],
            [4, 8, 8],
            [0, 16, 16],
            [0, 0, 0],
        ]
        self.sparse_tubes_tokenizer = SparseTubesTokenizer(self.hidden_dim,
                                                           self.kernel_sizes,
                                                           self.strides,
                                                           self.offsets,
                                                           space_to_depth_rate=space_to_depth_rate)

        self.pos_embedding = self._generate_position_embedding()
        self.register_parameter('pos_embedding', self.pos_embedding)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim), requires_grad=True)
        self.register_parameter('class_token', self.class_token)

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(self.hidden_dim, self.num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(self.hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, self.num_classes)

        self.heads = nn.Sequential(heads_layers)

    def forward(self, x):
        x = self.sparse_tubes_tokenizer(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.pos_embedding

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

    def _calc_conv_shape(self, kernel_size, stride, offset) -> np.ndarray:
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)

        output = np.ceil((self.video_shape[[1, 2, 3]] - offset - kernel_size + 1) / stride).astype(int)
        output = np.concatenate([np.array([self.hidden_dim / 6]),
                                 output]).astype(int)  # 6 elements (a sine and cosine value for each x; y; t)

        return output

    def _generate_position_embedding(self) -> torch.nn.Parameter:
        def _position_embedding_code(t, x, y, j, tau=10_000) -> Tensor:
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
        position_embedding = position_embedding.permute(1, 0).contiguous()
        position_embedding = torch.nn.Parameter(position_embedding, requires_grad=False)
        return position_embedding


class TubeViTLightningModule(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 video_shape,
                 num_layers,
                 num_heads,
                 hidden_dim,
                 mlp_dim,
                 weight_path: str = None,
                 space_to_depth_rate: int = 1,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model = TubeViT(
            num_classes=num_classes,
            video_shape=video_shape,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            space_to_depth_rate=space_to_depth_rate,
        )

        if 'lr' in kwargs:
            self.lr = kwargs['lr']
        else:
            self.lr = 1e-6

        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.example_input_array = Tensor(1, *video_shape)

        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path), strict=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy(y_pred, y, task='multiclass', num_classes=self.num_classes), prog_bar=True)
        self.log('train_f1', f1_score(y_pred, y, task='multiclass', num_classes=self.num_classes), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy(y_pred, y, task='multiclass', num_classes=self.num_classes), prog_bar=True)
        self.log('val_f1', f1_score(y_pred, y, task='multiclass', num_classes=self.num_classes), prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
