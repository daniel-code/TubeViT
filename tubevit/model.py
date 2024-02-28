from functools import partial
from typing import Any, Callable, List, Union

import lightning.pytorch as pl
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchmetrics.functional import accuracy, f1_score
from torchvision.models.vision_transformer import EncoderBlock
from typing_extensions import OrderedDict

from tubevit.positional_encoding import get_3d_sincos_pos_embed


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
    def __init__(self, hidden_dim, kernel_sizes, strides, offsets):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.offsets = offsets

        self.conv_proj_weight = nn.Parameter(
            torch.empty((self.hidden_dim, 3, *self.kernel_sizes[0])).normal_(), requires_grad=True
        )

        self.register_parameter("conv_proj_weight", self.conv_proj_weight)

        self.conv_proj_bias = nn.Parameter(torch.zeros(len(self.kernel_sizes), self.hidden_dim), requires_grad=True)
        self.register_parameter("conv_proj_bias", self.conv_proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, t, h, w = x.shape  # CTHW
        tubes = []
        for i in range(len(self.kernel_sizes)):
            if i == 0:
                weight = self.conv_proj_weight
            else:
                weight = F.interpolate(self.conv_proj_weight, self.kernel_sizes[i], mode="trilinear")

            tube = F.conv3d(
                x[:, :, self.offsets[i][0] :, self.offsets[i][1] :, self.offsets[i][2] :],
                weight,
                bias=self.conv_proj_bias[i],
                stride=self.strides[i],
            )

            tube = tube.reshape((n, self.hidden_dim, -1))

            tubes.append(tube)

        x = torch.cat(tubes, dim=-1)
        x = x.permute(0, 2, 1).contiguous()
        return x


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf

    code from https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """

        # (N, T, H) -> (N, T) -> (N, T, 1)
        att_w = nn.functional.softmax(self.W(x).squeeze(dim=-1), dim=-1).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
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
    ):
        super(TubeViT, self).__init__()
        self.video_shape = np.array(video_shape)  # CTHW
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.kernel_sizes = (
            (8, 8, 8),
            (12, 4, 4),
            (4, 12, 12),
            (1, 16, 16),
        )

        self.strides = (
            (16, 32, 32),
            (6, 32, 32),
            (16, 32, 32),
            (16, 16, 16),
        )

        self.offsets = (
            (0, 0, 0),
            (4, 8, 8),
            (0, 16, 16),
            (0, 0, 0),
        )
        self.sparse_tubes_tokenizer = SparseTubesTokenizer(
            self.hidden_dim, self.kernel_sizes, self.strides, self.offsets
        )

        self.pos_embedding = self._generate_position_embedding()
        self.pos_embedding = torch.nn.Parameter(self.pos_embedding, requires_grad=False)
        self.register_parameter("pos_embedding", self.pos_embedding)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim), requires_grad=True)
        self.register_parameter("class_token", self.class_token)

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.attention_pooling = SelfAttentionPooling(self.hidden_dim)

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

        # Attention pooling
        x = self.attention_pooling(x)

        x = self.heads(x)

        return x

    def _calc_conv_shape(self, kernel_size, stride, offset) -> np.ndarray:
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)
        output = np.floor(((self.video_shape[[1, 2, 3]] - offset - kernel_size) / stride) + 1).astype(int)
        return output

    def _generate_position_embedding(self) -> torch.nn.Parameter:
        position_embedding = [torch.zeros(1, self.hidden_dim)]

        for i in range(len(self.kernel_sizes)):
            tube_shape = self._calc_conv_shape(self.kernel_sizes[i], self.strides[i], self.offsets[i])
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                tube_shape=tube_shape,
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                offset=self.offsets[i],
            )
            position_embedding.append(pos_embed)

        position_embedding = torch.cat(position_embedding, dim=0).contiguous()
        return position_embedding


class TubeViTLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        video_shape,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        lr: float = 3e-4,
        weight_decay: float = 0,
        weight_path: str = None,
        max_epochs: int = None,
        label_smoothing: float = 0.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.num_classes = num_classes
        self.model = TubeViT(
            num_classes=num_classes,
            video_shape=video_shape,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.lr = lr
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.example_input_array = Tensor(1, *video_shape)

        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path), strict=False)
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)
        self.log("train_f1", f1_score(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)
        self.log("val_f1", f1_score(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            [
                {"params": self.model.sparse_tubes_tokenizer.parameters()},
                {"params": self.model.attention_pooling.parameters()},
                {"params": self.model.heads.parameters()},
                {"params": self.model.class_token},
                {"params": self.model.encoder.parameters(), "lr": self.lr * 0.01},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.max_epochs is not None:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.lr, total_steps=self.max_epochs
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        y_hat = self(x)
        y_pred = torch.softmax(y_hat, dim=-1)

        return {"y": y, "y_pred": torch.argmax(y_pred, dim=-1), "y_prob": y_pred}
