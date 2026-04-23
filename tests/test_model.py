import pytest
import torch

from tubevit.model import (
    Encoder,
    SelfAttentionPooling,
    SparseTubesTokenizer,
    TubeViT,
    TubeViTLightningModule,
    _apply_s2d,
    _cosine_with_warmup_lr_lambda,
)

# Minimal valid shape for the hard-coded tube configs:
#   T >= 20  (tube1: offset_t=4 + kernel_t=16 = 20)
#   H/W >= 28 (tube2: offset_h=16 + kernel_h=12 = 28)
VIDEO_SHAPE = [3, 32, 64, 64]  # CTHW
BATCH_SIZE = 2
HIDDEN_DIM = 24  # % 6 == 0 (pos-embed), % 4 == 0 (num_heads)
NUM_HEADS = 4
NUM_LAYERS = 2
MLP_DIM = 48
NUM_CLASSES = 10

# Expected token count for VIDEO_SHAPE = [3, 32, 64, 64]:
#   tube0 k=(8,8,8)   s=(16,32,32) o=(0,0,0)  : floor((32-8)/16+1)*floor((64-8)/32+1)^2  = 2*2*2 =  8
#   tube1 k=(16,4,4)  s=(6,32,32)  o=(4,8,8)  : floor((28-16)/6+1)*floor((56-4)/32+1)^2  = 3*2*2 = 12
#   tube2 k=(4,12,12) s=(16,32,32) o=(0,16,16): floor((32-4)/16+1)*floor((48-12)/32+1)^2 = 2*2*2 =  8
#   tube3 k=(1,16,16) s=(32,16,16) o=(0,0,0)  : floor((32-1)/32+1)*floor((64-16)/16+1)^2 = 1*4*4 = 16
TOTAL_TOKENS = 44

_KERNEL_SIZES = ((8, 8, 8), (16, 4, 4), (4, 12, 12), (1, 16, 16))
_STRIDES = ((16, 32, 32), (6, 32, 32), (16, 32, 32), (32, 16, 16))
_OFFSETS = ((0, 0, 0), (4, 8, 8), (0, 16, 16), (0, 0, 0))

# Paper's reference S2D config (Table 12): temporal 2× for tube 0, spatial 2×2 for tube 1
_S2D_FACTORS = ((2, 1), (1, 2), (1, 1), (1, 1))

# Token count with _S2D_FACTORS applied to VIDEO_SHAPE = [3, 32, 64, 64]:
#   tube0: (2,2,2) → temporal 2×: (1,2,2) = 4 tokens
#   tube1: (3,2,2) → spatial 2×: (3,1,1) = 3 tokens
#   tube2: (2,2,2) → no S2D     = 8 tokens
#   tube3: (1,4,4) → no S2D     = 16 tokens
TOTAL_TOKENS_S2D = 31


@pytest.fixture
def video():
    return torch.randn(BATCH_SIZE, *VIDEO_SHAPE)


@pytest.fixture
def tubevit():
    return TubeViT(
        num_classes=NUM_CLASSES,
        video_shape=VIDEO_SHAPE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        mlp_dim=MLP_DIM,
    )


@pytest.fixture
def tubevit_interpolated():
    return TubeViT(
        num_classes=NUM_CLASSES,
        video_shape=VIDEO_SHAPE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        mlp_dim=MLP_DIM,
        interpolated_kernels=True,
    )


@pytest.fixture
def tubevit_s2d():
    return TubeViT(
        num_classes=NUM_CLASSES,
        video_shape=VIDEO_SHAPE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        mlp_dim=MLP_DIM,
        s2d_factors=_S2D_FACTORS,
    )


class TestSparseTubesTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return SparseTubesTokenizer(HIDDEN_DIM, _KERNEL_SIZES, _STRIDES, _OFFSETS)

    @pytest.fixture
    def tokenizer_interpolated(self):
        return SparseTubesTokenizer(HIDDEN_DIM, _KERNEL_SIZES, _STRIDES, _OFFSETS, interpolated_kernels=True)

    @pytest.fixture
    def tokenizer_s2d(self):
        return SparseTubesTokenizer(HIDDEN_DIM, _KERNEL_SIZES, _STRIDES, _OFFSETS, s2d_factors=_S2D_FACTORS)

    def test_output_shape(self, tokenizer, video):
        assert tokenizer(video).shape == (BATCH_SIZE, TOTAL_TOKENS, HIDDEN_DIM)

    def test_output_shape_interpolated(self, tokenizer_interpolated, video):
        assert tokenizer_interpolated(video).shape == (BATCH_SIZE, TOTAL_TOKENS, HIDDEN_DIM)

    def test_s2d_output_shape(self, tokenizer_s2d, video):
        assert tokenizer_s2d(video).shape == (BATCH_SIZE, TOTAL_TOKENS_S2D, HIDDEN_DIM)

    def test_independent_has_per_tube_weights(self, tokenizer):
        param_names = {n for n, _ in tokenizer.named_parameters()}
        assert "conv_proj_weights.0" in param_names
        assert len(tokenizer.conv_proj_weights) == len(_KERNEL_SIZES)
        assert "conv_proj_weight" not in param_names

    def test_independent_weight_shapes(self, tokenizer):
        for i, k in enumerate(_KERNEL_SIZES):
            assert tokenizer.conv_proj_weights[i].shape == (HIDDEN_DIM, 3, *k)
        # bias: ParameterList, one vector per tube, all hidden_dim when no S2D
        for i in range(len(_KERNEL_SIZES)):
            assert tokenizer.conv_proj_biases[i].shape == (HIDDEN_DIM,)

    def test_interpolated_has_shared_weight(self, tokenizer_interpolated):
        param_names = {n for n, _ in tokenizer_interpolated.named_parameters()}
        assert "conv_proj_weight" in param_names
        assert not any("conv_proj_weights" in n for n in param_names)

    def test_interpolated_weight_shape(self, tokenizer_interpolated):
        assert tokenizer_interpolated.conv_proj_weight.shape == (HIDDEN_DIM, 3, *_KERNEL_SIZES[0])
        for i in range(len(_KERNEL_SIZES)):
            assert tokenizer_interpolated.conv_proj_biases[i].shape == (HIDDEN_DIM,)

    def test_s2d_bias_shapes(self, tokenizer_s2d):
        # tube 0: temporal 2× → conv out = HIDDEN_DIM // 2
        assert tokenizer_s2d.conv_proj_biases[0].shape == (HIDDEN_DIM // 2,)
        # tube 1: spatial 2× → conv out = HIDDEN_DIM // 4
        assert tokenizer_s2d.conv_proj_biases[1].shape == (HIDDEN_DIM // 4,)
        # tubes 2, 3: no S2D
        assert tokenizer_s2d.conv_proj_biases[2].shape == (HIDDEN_DIM,)
        assert tokenizer_s2d.conv_proj_biases[3].shape == (HIDDEN_DIM,)

    def test_no_extra_buffers(self, tokenizer):
        assert list(tokenizer.named_buffers()) == []


class TestSelfAttentionPooling:
    @pytest.fixture
    def pool(self):
        return SelfAttentionPooling(HIDDEN_DIM)

    def test_output_shape(self, pool):
        x = torch.randn(BATCH_SIZE, TOTAL_TOKENS, HIDDEN_DIM)
        assert pool(x).shape == (BATCH_SIZE, HIDDEN_DIM)

    def test_uniform_tokens_return_token_value(self, pool):
        # identical tokens → uniform attention weights → weighted sum equals the token
        token = torch.randn(BATCH_SIZE, 1, HIDDEN_DIM)
        x = token.expand(-1, 10, -1)
        out = pool(x)
        assert torch.allclose(out, token.squeeze(1), atol=1e-5)

    def test_single_token_passthrough(self, pool):
        x = torch.randn(BATCH_SIZE, 1, HIDDEN_DIM)
        out = pool(x)
        assert torch.allclose(out, x.squeeze(1), atol=1e-5)


class TestEncoder:
    @pytest.fixture
    def encoder(self):
        return Encoder(NUM_LAYERS, NUM_HEADS, HIDDEN_DIM, MLP_DIM, dropout=0.0, attention_dropout=0.0)

    def test_output_shape_preserved(self, encoder):
        x = torch.randn(BATCH_SIZE, TOTAL_TOKENS, HIDDEN_DIM)
        assert encoder(x).shape == x.shape

    def test_rejects_2d_input(self, encoder):
        with pytest.raises(Exception):
            encoder(torch.randn(BATCH_SIZE, HIDDEN_DIM))


class TestTubeViT:
    def test_output_shape(self, tubevit, video):
        assert tubevit(video).shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_shape_interpolated(self, tubevit_interpolated, video):
        assert tubevit_interpolated(video).shape == (BATCH_SIZE, NUM_CLASSES)

    def test_s2d_output_shape(self, tubevit_s2d, video):
        assert tubevit_s2d(video).shape == (BATCH_SIZE, NUM_CLASSES)

    def test_s2d_pos_embedding_shape(self, tubevit_s2d):
        assert tubevit_s2d.pos_embedding.shape == (TOTAL_TOKENS_S2D, HIDDEN_DIM)

    def test_pos_embedding_is_buffer_not_parameter(self, tubevit):
        buffers = {n for n, _ in tubevit.named_buffers()}
        params = {n for n, _ in tubevit.named_parameters()}
        assert "pos_embedding" in buffers
        assert "pos_embedding" not in params

    def test_pos_embedding_not_trainable(self, tubevit):
        assert not tubevit.pos_embedding.requires_grad

    def test_pos_embedding_shape(self, tubevit):
        assert tubevit.pos_embedding.shape == (TOTAL_TOKENS, HIDDEN_DIM)

    def test_no_class_token(self, tubevit):
        param_names = {n for n, _ in tubevit.named_parameters()}
        assert not any("class_token" in n for n in param_names)

    def test_pos_embedding_nonzero(self, tubevit):
        # fixed sincos encoding must not be all-zero
        assert not torch.all(tubevit.pos_embedding == 0)


class TestTubeViTLightningModule:
    @pytest.fixture
    def module(self):
        return TubeViTLightningModule(
            num_classes=NUM_CLASSES,
            video_shape=VIDEO_SHAPE,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
            mlp_dim=MLP_DIM,
        )

    def test_hparams_saved(self, module):
        assert module.hparams.num_classes == NUM_CLASSES
        assert module.hparams.hidden_dim == HIDDEN_DIM

    def test_forward_shape(self, module, video):
        assert module(video).shape == (BATCH_SIZE, NUM_CLASSES)

    def test_example_input_array_shape(self, module):
        assert module.example_input_array.shape == (1, *VIDEO_SHAPE)

    def test_warmup_steps_hparam_saved(self):
        m = TubeViTLightningModule(
            num_classes=NUM_CLASSES,
            video_shape=VIDEO_SHAPE,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
            mlp_dim=MLP_DIM,
            warmup_steps=500,
        )
        assert m.hparams.warmup_steps == 500


class TestApplyS2d:
    def test_identity_when_no_s2d(self):
        x = torch.randn(2, 24, 4, 8, 8)
        out = _apply_s2d(x, t_factor=1, s_factor=1)
        assert out.shape == x.shape
        assert torch.equal(out, x)

    def test_temporal_s2d_shape(self):
        x = torch.randn(2, 12, 4, 8, 8)
        out = _apply_s2d(x, t_factor=2, s_factor=1)
        assert out.shape == (2, 24, 2, 8, 8)

    def test_spatial_s2d_shape(self):
        x = torch.randn(2, 6, 4, 8, 8)
        out = _apply_s2d(x, t_factor=1, s_factor=2)
        assert out.shape == (2, 24, 4, 4, 4)

    def test_combined_s2d_shape(self):
        # t=2, s=2 → fold 2 temporal + 2×2 spatial → channel ×8
        x = torch.randn(2, 3, 4, 8, 8)
        out = _apply_s2d(x, t_factor=2, s_factor=2)
        assert out.shape == (2, 24, 2, 4, 4)

    def test_temporal_s2d_values(self):
        # 4 temporal frames numbered 0-3; after temporal 2×:
        # T_new=0 → channels [0, 1],  T_new=1 → channels [2, 3]
        x = torch.arange(4, dtype=torch.float).reshape(1, 1, 4, 1, 1)
        out = _apply_s2d(x, t_factor=2, s_factor=1)
        assert out.shape == (1, 2, 2, 1, 1)
        assert out[0, 0, 0, 0, 0].item() == 0.0
        assert out[0, 1, 0, 0, 0].item() == 1.0
        assert out[0, 0, 1, 0, 0].item() == 2.0
        assert out[0, 1, 1, 0, 0].item() == 3.0

    def test_spatial_s2d_values(self):
        # 4×4 grid numbered 0-15; after spatial 2×:
        # ch0 = top-left of each 2×2 patch, ch1 = top-right, ch2 = btm-left, ch3 = btm-right
        x = torch.arange(16, dtype=torch.float).reshape(1, 1, 1, 4, 4)
        out = _apply_s2d(x, t_factor=1, s_factor=2)
        assert out.shape == (1, 4, 1, 2, 2)
        assert out[0, 0, 0].tolist() == [[0.0, 2.0], [8.0, 10.0]]
        assert out[0, 1, 0].tolist() == [[1.0, 3.0], [9.0, 11.0]]
        assert out[0, 2, 0].tolist() == [[4.0, 6.0], [12.0, 14.0]]
        assert out[0, 3, 0].tolist() == [[5.0, 7.0], [13.0, 15.0]]


class TestCosineWithWarmupLrLambda:
    TOTAL = 100

    # ── no warmup (warmup_steps=0) ──────────────────────────────────────────

    def test_no_warmup_step0_is_one(self):
        assert _cosine_with_warmup_lr_lambda(0, warmup_steps=0, total_steps=self.TOTAL) == 1.0

    def test_no_warmup_final_step_is_zero(self):
        v = _cosine_with_warmup_lr_lambda(self.TOTAL, warmup_steps=0, total_steps=self.TOTAL)
        assert abs(v) < 1e-6

    def test_no_warmup_midpoint_is_half(self):
        v = _cosine_with_warmup_lr_lambda(self.TOTAL // 2, warmup_steps=0, total_steps=self.TOTAL)
        assert abs(v - 0.5) < 1e-6

    def test_no_warmup_monotone_decreasing(self):
        vals = [_cosine_with_warmup_lr_lambda(s, 0, self.TOTAL) for s in range(self.TOTAL + 1)]
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    # ── with warmup ─────────────────────────────────────────────────────────

    def test_warmup_step0_is_zero(self):
        assert _cosine_with_warmup_lr_lambda(0, warmup_steps=10, total_steps=self.TOTAL) == 0.0

    def test_warmup_end_is_one(self):
        v = _cosine_with_warmup_lr_lambda(10, warmup_steps=10, total_steps=self.TOTAL)
        assert abs(v - 1.0) < 1e-6

    def test_warmup_midpoint_is_half(self):
        v = _cosine_with_warmup_lr_lambda(5, warmup_steps=10, total_steps=self.TOTAL)
        assert abs(v - 0.5) < 1e-6

    def test_warmup_monotone_increasing(self):
        vals = [_cosine_with_warmup_lr_lambda(s, 10, self.TOTAL) for s in range(11)]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    def test_cosine_phase_monotone_decreasing(self):
        vals = [_cosine_with_warmup_lr_lambda(s, 10, self.TOTAL) for s in range(10, self.TOTAL + 1)]
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_final_step_with_warmup_is_zero(self):
        v = _cosine_with_warmup_lr_lambda(self.TOTAL, warmup_steps=10, total_steps=self.TOTAL)
        assert abs(v) < 1e-6

    # ── clamp: never negative ────────────────────────────────────────────────

    def test_never_negative(self):
        for s in range(self.TOTAL + 5):
            assert _cosine_with_warmup_lr_lambda(s, 10, self.TOTAL) >= 0.0
