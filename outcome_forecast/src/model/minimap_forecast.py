import enum
import functools
import operator
from dataclasses import dataclass, field
from typing import Any, Sequence


import torch
from torch import nn, Tensor
from torch.nn import functional as F
from konductor.data import get_dataset_properties
from konductor.init import ModuleInitConfig
from konductor.models import MODEL_REGISTRY, ExperimentInitConfig
from konductor.models._pytorch import TorchModelConfig


class MinimapTarget(enum.Enum):
    SELF = enum.auto()
    ENEMY = enum.auto()
    BOTH = enum.auto()

    @staticmethod
    def indices(target: "MinimapTarget"):
        """Index of target(s) in minimap feature layer stack"""
        match target:
            case MinimapTarget.SELF:
                return [-4]
            case MinimapTarget.ENEMY:
                return [-1]
            case MinimapTarget.BOTH:
                return [-4, -1]

    @staticmethod
    def names(target: "MinimapTarget"):
        """Index of target(s) in minimap feature layer stack"""
        match target:
            case MinimapTarget.SELF:
                return ["self"]
            case MinimapTarget.ENEMY:
                return ["enemy"]
            case MinimapTarget.BOTH:
                return ["self", "enemy"]


def _make_mlp(in_ch: int, hidden_ch: int, out_ch: int | None = None):
    """If out_ch is None default to in_ch"""
    if out_ch is None:
        out_ch = in_ch
    return nn.Sequential(
        nn.LayerNorm(in_ch),
        nn.Linear(in_ch, hidden_ch),
        nn.GELU(),
        nn.Linear(hidden_ch, out_ch),
    )


@MODEL_REGISTRY.register_module("temporal-conv")
class TemporalConv(nn.Sequential):
    """Steps temporal dimension in half and then to one"""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        n_timesteps: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_ch, hidden_ch, (n_timesteps // 2 + 1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(hidden_ch),
            activation(),
            nn.Conv3d(hidden_ch, out_ch, (n_timesteps // 2, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_ch),
            activation(),
        )
        self.out_ch = out_ch


@MODEL_REGISTRY.register_module("temporal-conv-v2")
class TemporalConv2(nn.Sequential):
    """Convolve over adjacent frames twice before temporally downsampling"""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        n_timesteps: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(
            nn.Conv3d(in_ch, hidden_ch, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_ch),
            activation(),
            nn.Conv3d(hidden_ch, hidden_ch, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_ch),
            activation(),
            nn.Conv3d(
                hidden_ch, hidden_ch, (n_timesteps // 2 + 1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(hidden_ch),
            activation(),
            nn.Conv3d(hidden_ch, out_ch, (n_timesteps // 2, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_ch),
            activation(),
        )
        self.out_ch = out_ch


@MODEL_REGISTRY.register_module("temporal-conv-multi-out")
class TemporalConv2Multi(nn.Sequential):
    """Convolve over volume a few times"""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        n_timesteps: int,
        out_timesteps: int,
        n_layers: int = 2,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        self.out_ch = out_ch
        reduce_dim = n_timesteps - out_timesteps + 1

        def make_layer(in_ch_: int):
            return (
                nn.Conv3d(in_ch_, hidden_ch, (3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(hidden_ch),
                activation(),
            )

        layers = [*make_layer(in_ch)]
        for _ in range(n_layers - 1):
            layers.extend(make_layer(hidden_ch))

        super().__init__(
            *layers,
            nn.Conv3d(hidden_ch, out_ch, (reduce_dim, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_ch),
            activation(),
        )


@MODEL_REGISTRY.register_module("temporal-conv-decoder")
class TemporalConv2MultiDecoder(TemporalConv2Multi):
    """Additional point-wise linear layer with final output channels"""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        n_timesteps: int,
        out_timesteps: int,
        n_layers: int = 2,
    ) -> None:
        super().__init__(
            in_ch, hidden_ch, hidden_ch, n_timesteps, out_timesteps, n_layers
        )
        # Add last pixel-wise module without ReLU
        self.add_module(
            str(len(self._modules)), nn.Conv3d(hidden_ch, out_ch, (1, 1, 1))
        )


@dataclass
class BaseConfig(TorchModelConfig):
    encoder: ModuleInitConfig
    temporal: ModuleInitConfig
    decoder: ModuleInitConfig
    history_len: int = 8
    target: MinimapTarget = MinimapTarget.BOTH

    @property
    def future_len(self) -> int:
        return 1

    @property
    def is_logit_output(self):
        return True

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args
        model_cfg["encoder"]["args"]["in_ch"] = props["image_ch"]
        return super().from_config(config, idx)

    def __post_init__(self):
        if isinstance(self.encoder, dict):
            self.encoder = ModuleInitConfig(**self.encoder)
        if isinstance(self.temporal, dict):
            self.temporal = ModuleInitConfig(**self.temporal)
        if isinstance(self.decoder, dict):
            self.decoder = ModuleInitConfig(**self.decoder)
        if isinstance(self.target, str):
            self.target = MinimapTarget[self.target.upper()]


@dataclass
@MODEL_REGISTRY.register_module("conv-forecast-v1")
class ConvV1Config(BaseConfig):
    """
    Next frame convolution forecasting v1
    """

    def __post_init__(self):
        super().__post_init__()
        assert self.history_len % 2 == 0, "history_len must be even"
        self.temporal.args["n_timesteps"] = self.history_len

    def get_instance(self, *args, **kwargs) -> Any:
        """Construct modules and return conv forecaster"""
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)

        self.temporal.args["in_ch"] = encoder.out_ch[-1]
        temporal = MODEL_REGISTRY[self.temporal.type](**self.temporal.args)

        self.decoder.args["in_ch"] = temporal.out_ch + encoder.out_ch[0]
        self.decoder.args["out_ch"] = len(MinimapTarget.indices(self.target))
        decoder = MODEL_REGISTRY[self.decoder.type](**self.decoder.args)
        return ConvForecast(encoder, temporal, decoder, self.history_len)


class ConvForecast(nn.Module):
    """
    Use Siamese ConvNet to Extract features
    3dConv Along Features
    Upsample and Concatenate with Last Image
    Conv Decode TemporalFeatures+Last Image for Final Output
    """

    def __init__(
        self,
        encoder: nn.Module,
        temporal: nn.Module,
        decoder: nn.Module,
        history_len: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal_conv = temporal
        self.decoder = decoder
        self.history_len = history_len

    @property
    def future_len(self):
        return 1

    @property
    def is_logit_output(self):
        return True

    def forward_sequence(self, inputs: Tensor) -> Tensor:
        minimap_low: list[Tensor] = []
        for t in range(inputs.shape[1]):
            enc: list[Tensor] = self.encoder(inputs[:, t])
            minimap_low.append(enc[0])
        last_minimap_high = enc[-1]

        stacked_feats = torch.stack(minimap_low, dim=2)
        temporal_feats: Tensor = self.temporal_conv(stacked_feats)
        temporal_feats = F.interpolate(
            temporal_feats.squeeze(2),
            size=last_minimap_high.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        cat_features = torch.cat([temporal_feats, last_minimap_high], dim=1)
        decoded = self.decoder(cat_features)
        return decoded

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """"""
        minimaps = inputs["minimap_features"]
        ntime = minimaps.shape[1]
        preds: list[Tensor] = []
        for start_idx in range(ntime - self.history_len):
            end_idx = start_idx + self.history_len
            pred = self.forward_sequence(minimaps[:, start_idx:end_idx])
            pred = F.interpolate(
                pred, mode="bilinear", size=minimaps.shape[-2:], align_corners=True
            )
            preds.append(pred)

        out = torch.stack(preds, dim=1)
        return out


@dataclass
@MODEL_REGISTRY.register_module("conv-forecast-v2")
class ConvV2Config(BaseConfig):
    """
    Next frame convolution forecasting v2
    Adds separate high res last frame context encoder
    """

    last_frame_encoder: ModuleInitConfig = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        assert self.history_len % 2 == 0, "history_len must be even"
        self.temporal.args["n_timesteps"] = self.history_len
        if isinstance(self.last_frame_encoder, dict):
            self.last_frame_encoder = ModuleInitConfig(**self.last_frame_encoder)

    def get_instance(self, *args, **kwargs) -> Any:
        """Construct modules and return conv forecaster"""
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)

        self.temporal.args["in_ch"] = encoder.out_ch
        temporal = MODEL_REGISTRY[self.temporal.type](**self.temporal.args)

        self.last_frame_encoder.args["in_ch"] = self.encoder.args["in_ch"]
        last_frame_encoder = MODEL_REGISTRY[self.last_frame_encoder.type](
            **self.last_frame_encoder.args
        )

        self.decoder.args["in_ch"] = temporal.out_ch + last_frame_encoder.out_ch
        self.decoder.args["out_ch"] = len(MinimapTarget.indices(self.target))
        decoder = MODEL_REGISTRY[self.decoder.type](**self.decoder.args)

        return ConvForecastV2(
            encoder, temporal, decoder, last_frame_encoder, self.history_len
        )


class ConvForecastV2(nn.Module):
    """
    V2 uses another feature extractor for high resolution last frame
    features as to not add confounding factor to temporal feature inputs.
    """

    def __init__(
        self,
        encoder: nn.Module,
        temporal: nn.Module,
        decoder: nn.Module,
        last_frame_encoder: nn.Module,
        history_len: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal_conv = temporal
        self.decoder = decoder
        self.last_frame_encoder = last_frame_encoder
        self.history_len = history_len

    @property
    def future_len(self):
        return 1

    @property
    def is_logit_output(self):
        return True

    def forward_sequence(self, inputs: Tensor) -> Tensor:
        # Squeeze and unsqueeze time dimension for siamese encoder
        img_feats: Tensor = self.encoder(inputs.reshape(-1, *inputs.shape[2:]))
        img_feats = img_feats.reshape(*inputs.shape[:2], *img_feats.shape[1:])
        img_feats = img_feats.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]

        last_feats: Tensor = self.last_frame_encoder(inputs[:, -1])
        temporal_feats: Tensor = self.temporal_conv(img_feats)
        temporal_feats = F.interpolate(
            temporal_feats.squeeze(2),
            size=last_feats.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        cat_features = torch.cat([temporal_feats, last_feats], dim=1)
        decoded = self.decoder(cat_features)
        return decoded

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """"""
        minimaps = inputs["minimap_features"]
        ntime = minimaps.shape[1]
        preds: list[Tensor] = []
        for start_idx in range(ntime - self.history_len):
            end_idx = start_idx + self.history_len
            pred = self.forward_sequence(minimaps[:, start_idx:end_idx])
            pred = F.interpolate(
                pred, mode="bilinear", size=minimaps.shape[-2:], align_corners=True
            )
            preds.append(pred)

        out = torch.stack(preds, dim=1)
        return out


@dataclass
@MODEL_REGISTRY.register_module("conv-forecast-v2-multiframe")
class ConvV2MultiConfig(ConvV2Config):
    """
    Predicts three frames into the future rather than one
    """

    last_frame_encoder: ModuleInitConfig = field(kw_only=True)

    @property
    def future_len(self) -> int:
        return 3

    def __post_init__(self):
        # Subtrack two off history length as we will predict 3 frames instead
        self.history_len -= 2
        super().__post_init__()

    def get_instance(self, *args, **kwargs) -> Any:
        """Construct modules and return conv forecaster"""
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)

        self.temporal.args["in_ch"] = encoder.out_ch
        self.temporal.args["out_timesteps"] = self.future_len
        temporal = MODEL_REGISTRY[self.temporal.type](**self.temporal.args)

        self.last_frame_encoder.args["in_ch"] = self.encoder.args["in_ch"]
        last_frame_encoder = MODEL_REGISTRY[self.last_frame_encoder.type](
            **self.last_frame_encoder.args
        )

        assert temporal.out_ch == last_frame_encoder.out_ch
        self.decoder.args["in_ch"] = temporal.out_ch
        self.decoder.args["out_ch"] = len(MinimapTarget.indices(self.target))
        decoder = MODEL_REGISTRY[self.decoder.type](**self.decoder.args)

        return ConvForecastV2Multi(
            encoder, temporal, decoder, last_frame_encoder, self.history_len
        )


class ConvForecastV2Multi(ConvForecastV2):
    """ConvV2 Minimap Forecast but 3 frames output"""

    @property
    def future_len(self):
        return 3

    @property
    def is_logit_output(self):
        return True

    def forward_sequence(self, inputs: Tensor):
        # Squeeze and unsqueeze time dimension for siamese encoder
        img_feats: Tensor = self.encoder(inputs.reshape(-1, *inputs.shape[2:]))
        img_feats = img_feats.reshape(*inputs.shape[:2], *img_feats.shape[1:])
        img_feats = img_feats.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]

        last_feats: Tensor = self.last_frame_encoder(inputs[:, -1])
        # Add time dim
        last_feats = last_feats.unsqueeze(2)

        temporal_feats: Tensor = self.temporal_conv(img_feats)
        temporal_feats = F.interpolate(
            temporal_feats,
            size=[self.future_len, *last_feats.shape[-2:]],
            mode="trilinear",
            align_corners=True,
        )

        cat_features = torch.cat([last_feats, temporal_feats], dim=2)
        decoded: Tensor = self.decoder(cat_features)
        decoded = decoded.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] -> [B,T,C,H,W]
        return decoded

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """"""
        minimaps = inputs["minimap_features"]

        pred = self.forward_sequence(minimaps[:, : self.history_len])
        out: Tensor = F.interpolate(
            pred.reshape(-1, *pred.shape[-3:]),
            mode="bilinear",
            size=minimaps.shape[-2:],
            align_corners=True,
        )
        out = out.reshape(*pred.shape[:3], *minimaps.shape[-2:])
        return out


class TemporalConvV3(nn.Module):
    """
    Ultra barebones temporal model that channel-wise stacks self/enemy
    and predicts the residual from the previous frame and always stays at
    high resolution to ensure fidelity.
    """

    @property
    def future_len(self):
        return 1

    @property
    def is_logit_output(self):
        return False

    def __init__(
        self,
        history_len: int,
        hidden_ch: list[int],
        kernel_size: list[int],
        in_layers: MinimapTarget,
        out_layers: MinimapTarget,
    ) -> None:
        super().__init__()
        assert len(hidden_ch) == len(kernel_size)
        self.in_layers = in_layers
        self.out_layers = out_layers
        self.history_len = history_len

        modules = []
        in_ch = history_len * len(MinimapTarget.indices(in_layers))
        for out_ch, kernel in zip(hidden_ch, kernel_size):
            modules.append(self._make_layer(in_ch, out_ch, kernel))
            in_ch = out_ch
        modules.append(nn.Conv2d(in_ch, len(MinimapTarget.indices(out_layers)), 1))
        self.model = nn.Sequential(*modules)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, kernel_size: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, inputs: dict[str, Tensor]):
        """"""
        in_ch = MinimapTarget.indices(self.in_layers)
        minimaps = inputs["minimap_features"][:, : self.history_len, in_ch]
        minimap_stack = minimaps.reshape(
            -1, self.history_len * len(in_ch), *minimaps.shape[-2:]
        )
        residule = self.model(minimap_stack)
        residule = F.tanh(residule)

        out_ch = MinimapTarget.indices(self.out_layers)
        last_minimap = inputs["minimap_features"][:, self.history_len, out_ch]
        # Ensure prediction between 0 and 1 and unsqueeze time dimension
        prediction = torch.clamp(last_minimap + residule, 0, 1).unsqueeze(1)
        return prediction


@dataclass
@MODEL_REGISTRY.register_module("conv-forecast-v3")
class ConvForecasterV3(TorchModelConfig):
    hidden_chs: list[int]
    kernel_sizes: list[int]
    in_layers: MinimapTarget
    history_len: int = 8
    target: MinimapTarget = MinimapTarget.BOTH

    def __post_init__(self):
        if isinstance(self.target, str):
            self.target = MinimapTarget[self.target.upper()]
        if isinstance(self.in_layers, str):
            self.in_layers = MinimapTarget[self.in_layers.upper()]

    @property
    def future_len(self) -> int:
        return 1

    @property
    def is_logit_output(self):
        return False

    def get_instance(self, *args, **kwargs) -> Any:
        return TemporalConvV3(
            self.history_len,
            self.hidden_chs,
            self.kernel_sizes,
            self.in_layers,
            self.target,
        )


@MODEL_REGISTRY.register_module("cross-attn-block-v1")
class CrossAttentionBlockV1(nn.Module):
    """CrossAttention -> SelfAttention Module"""

    def __init__(self, q_ch: int, kv_ch: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(q_ch)
        self.kv_norm = nn.LayerNorm(kv_ch)
        self.c_attn = nn.MultiheadAttention(
            embed_dim=q_ch,
            kdim=kv_ch,
            vdim=kv_ch,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.middle_mlp = _make_mlp(q_ch, q_ch * 2)
        self.middle_norm = nn.LayerNorm(q_ch)
        self.s_attn = nn.MultiheadAttention(
            embed_dim=q_ch, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.out_mlp = _make_mlp(q_ch, q_ch * 2)

    def forward(self, query: Tensor, keyvalue: Tensor):
        inter: Tensor
        keyvalue_norm = self.kv_norm(keyvalue)
        inter = query + self.c_attn(self.q_norm(query), keyvalue_norm, keyvalue_norm)[0]
        inter = inter + self.middle_mlp(inter)
        middle_norm = self.middle_norm(inter)
        inter = inter + self.s_attn(middle_norm, middle_norm, middle_norm)[0]
        inter = inter + self.out_mlp(inter)
        return inter


@MODEL_REGISTRY.register_module("transformer-resampler")
class TransformerResampler(nn.Module):
    """
    Alternate between cross-attn on input data and
    self-attn over latent to decode.
    """

    def __init__(
        self, latent_dim: int, kv_dim: int, num_blocks: int, num_heads: int = 4
    ) -> None:
        super().__init__()
        self.cross_attn_blocks = nn.ModuleList(
            CrossAttentionBlockV1(latent_dim, kv_dim, num_heads)
            for _ in range(num_blocks)
        )

    def forward(self, query: Tensor, keyvalue: Tensor):
        for module in self.cross_attn_blocks:
            query = module(query, keyvalue)
        return query


class PosQueryDecoder(nn.Module):
    def __init__(
        self,
        out_shape: tuple[int, int],
        input_dim: int,
        output_dim: int,
        num_heads: int,
        query_cfg: ModuleInitConfig,
    ) -> None:
        super().__init__()

        self.out_shape = out_shape
        queries: nn.Parameter | Tensor = {
            "fixed-queries": make_fixed_queries,
            "learned-queries": make_learned_queries,
            "learned-queries-freq": make_learned_freq_queries,
        }[query_cfg.type](out_shape=out_shape, **query_cfg.args)
        if not isinstance(queries, nn.Parameter):
            self.register_buffer("queries", queries, persistent=False)
        else:
            self.queries = queries

        self.input_norm = nn.LayerNorm(input_dim)
        self.decoder = nn.MultiheadAttention(
            embed_dim=self.queries.shape[-1],
            kdim=input_dim,
            vdim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.linear = _make_mlp(
            self.queries.shape[-1], self.queries.shape[-1] * 2, output_dim
        )

    def forward(self, latent: Tensor):
        queries = self.queries[None].expand(latent.shape[0], *self.queries.shape)
        in_norm = self.input_norm(latent)
        decoded: Tensor = self.decoder(queries, in_norm, in_norm)[0]
        out: Tensor = self.linear(decoded)
        out = out.permute(0, 2, 1)  # spatial last
        out = out.reshape(latent.shape[0], -1, *self.out_shape)
        return out


def make_sinusoid_encodings(
    out_shape: Sequence[int],
    f_num: int,
    f_max: float | None = None,
    dev: torch.device | None = None,
):
    """Make fixed queries with sine/cosine encodings"""
    if f_max is None:
        f_max = float(max(out_shape))

    coords = torch.stack(
        torch.meshgrid(
            [torch.linspace(-1, 1, steps=s, device=dev) for s in out_shape],
            indexing="ij",
        ),
        dim=-1,
    )
    frequencies = torch.linspace(1.0, f_max / 2.0, f_num, device=dev)[None, None, None]
    frequency_grids = torch.pi * coords[..., None] * frequencies
    encodings = torch.cat([frequency_grids.sin(), frequency_grids.cos()], dim=-1)

    return encodings.reshape(*out_shape, -1)


def make_fixed_queries(
    out_shape: Sequence[int], f_num: int, f_max: float | None = None
):
    queries = make_sinusoid_encodings(out_shape, f_num, f_max)
    queries = queries.reshape(-1, queries.shape[-1])
    return queries


def make_learned_queries(out_shape: Sequence[int], query_dim: int):
    """Make parameters for learned queries"""
    n_queries = functools.reduce(operator.mul, out_shape)
    queries = nn.Parameter(torch.empty(n_queries, query_dim))
    with torch.no_grad():
        queries.normal_(0, 0.5).clamp_(-2, 2)
    return queries


def make_learned_freq_queries(
    out_shape: Sequence[int], f_num: int, f_max: float | None = None
):
    """Initialize parameter with sine/cosine encodings"""
    queries = make_fixed_queries(out_shape, f_num, f_max)
    return nn.Parameter(queries)


class TransformerForecasterV1(nn.Module):
    """Just do the ol' flatten, and MHA over time and space"""

    is_logit_output = True

    def __init__(
        self,
        encoder: nn.Module,
        temporal: nn.Module,
        decoder: nn.Module,
        num_latents: int,
        latent_dim: int,
        history_len: int,
        latent_minimap_shape: tuple[int, int] | None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.temporal = temporal
        self.decoder = decoder
        self.history_len = history_len
        self.latent_minimap_shape = latent_minimap_shape
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

    @torch.no_grad()
    def _init_parameters(self):
        self.latent.normal_(0, 0.5).clamp_(-2.0, 2.0)

    def flatten_input_encodings(self, inputs: Tensor):
        """Transform inputs [B,T,C,H,W] to [B,T H W,C] and add position embeddings"""
        inputs = torch.permute(inputs, [0, 1, 3, 4, 2])
        f_num = 8 if self.latent_minimap_shape is None else self.latent_minimap_shape[0]
        pos_enc = make_sinusoid_encodings(inputs.shape[1:4], f_num, dev=inputs.device)
        pos_enc = pos_enc[None].expand(inputs.shape[0], *pos_enc.shape)
        inputs_pos = torch.cat([inputs, pos_enc], dim=-1)
        inputs_pos = inputs_pos.reshape(inputs.shape[0], -1, inputs_pos.shape[-1])
        return inputs_pos

    def encode_inputs(self, inputs: Tensor):
        b_sz, t_sz = inputs.shape[:2]
        inputs = inputs.reshape(-1, *inputs.shape[2:])
        inputs_enc = self.encoder(inputs)
        if self.latent_minimap_shape is not None:
            inputs_enc = F.adaptive_avg_pool2d(inputs_enc, self.latent_minimap_shape)
        inputs_enc = inputs_enc.reshape(b_sz, t_sz, -1, *inputs_enc.shape[-2:])
        return inputs_enc

    def forward_sequence(self, inputs: Tensor):
        """Run a single sequence [B,T,C,H,W]"""
        inputs_enc = self.encode_inputs(inputs)
        inputs_enc = self.flatten_input_encodings(inputs_enc)
        latent = self.latent[None].expand(inputs_enc.shape[0], *self.latent.shape)
        temporal_feats = self.temporal(latent, inputs_enc)
        output = self.decoder(temporal_feats)
        return output

    def forward(self, inputs: dict[str, Tensor]):
        """If input sequence is longer than designated, 'convolve' over input"""
        minimaps = inputs["minimap_features"]
        ntime = minimaps.shape[1]
        preds: list[Tensor] = []
        for start_idx in range(ntime - self.history_len):
            end_idx = start_idx + self.history_len
            pred = self.forward_sequence(minimaps[:, start_idx:end_idx])
            preds.append(pred)

        out = torch.stack(preds, dim=1)
        return out


@dataclass
@MODEL_REGISTRY.register_module("transformer-forecast-v1")
class TransformerForecasterConfig(BaseConfig):
    decoder_query: ModuleInitConfig = field(kw_only=True)
    latent_minimap_shape: tuple[int, int] | None = None
    num_latents: int = field(kw_only=True)
    latent_dim: int = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.decoder_query, ModuleInitConfig):
            self.decoder_query = ModuleInitConfig(**self.decoder_query)

    def get_instance(self, *args, **kwargs) -> Any:
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)
        if hasattr(encoder, "disable_fpn"):
            assert encoder.disable_fpn
        self.temporal.args["latent_dim"] = self.latent_dim

        pos_dim = (
            8 if self.latent_minimap_shape is None else self.latent_minimap_shape[0]
        )
        pos_dim *= 6  # (T,H,W)*(sin,cos)
        self.temporal.args["kv_dim"] = encoder.out_ch + pos_dim
        temporal = MODEL_REGISTRY[self.temporal.type](**self.temporal.args)

        self.decoder.args["input_dim"] = self.latent_dim
        self.decoder.args["output_dim"] = len(MinimapTarget.indices(self.target))
        decoder = PosQueryDecoder(query_cfg=self.decoder_query, **self.decoder.args)

        return self.init_auto_filter(
            TransformerForecasterV1, encoder=encoder, temporal=temporal, decoder=decoder
        )
