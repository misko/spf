import logging
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer

from spf.model_training_and_inference.models.beamsegnet import FFNN
from spf.rf import rotate_dist, torch_pi_norm

TEMP = 10


# chat GPT


class PositionalEncoding(nn.Module):
    """
    Standard Transformer positional encoding for 1D sequences.
    If you want a simpler approach, you can skip this module
    and rely on the transformer's learned attention alone,
    but positional encoding often helps with sequence order.
    """

    def __init__(self, d_model, max_len=10000):
        super().__init__()

        # Create a (max_len, d_model) position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        # Register as a buffer so it’s not a learnable parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape = (batch_size, seq_len, d_model)
        We'll add the positional encoding up to seq_len.
        """
        seq_len = x.size(1)
        # Add (1, seq_len, d_model) to (batch, seq_len, d_model)
        x = x + self.pe[:seq_len, :]
        return x


class TransformerTimeModel(nn.Module):
    """
    Treats the input (batch, channels=68, time=256)
    as a sequence of length=256 with 68-dim features at each step.

    Steps:
      1) Transpose to (batch, seq=256, features=68)
      2) Project 68 -> d_model
      3) (Optional) add positional encoding
      4) Pass through TransformerEncoder (num_layers)
      5) Pool over seq dimension
      6) Final linear to output_dim=12
    """

    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        output_channels=12,
        use_positional_encoding=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_pe = use_positional_encoding

        # 1) Linear embedding from input_dim=68 -> d_model
        self.input_fc = nn.Linear(68, d_model)

        # 2) Positional encoding (optional)
        if self.use_pe:
            self.pos_encoder = PositionalEncoding(d_model=d_model)

        # 3) Transformer Encoder
        #    The PyTorch nn.TransformerEncoder requires an EncoderLayer + num_layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,  # crucial so our shape can be (batch, seq, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 4) Final linear -> 12
        self.fc_out = nn.Linear(d_model, output_channels)

    def forward(self, x):
        """
        x shape: (batch, 68, 256)
        return: (batch, 12)
        """
        # --- (A) Transpose to (batch, seq=256, feat=68) ---
        x = x.transpose(1, 2)  # (batch, 256, 68)

        # --- (B) Embedding: project from 68 -> d_model ---
        x = self.input_fc(x)  # (batch, 256, d_model)

        # --- (C) Positional encoding (optional) ---
        if self.use_pe:
            x = self.pos_encoder(x)

        # --- (D) Pass through Transformer Encoder ---
        # shape remains (batch, seq=256, d_model)
        x = self.transformer_encoder(x)  # shape: (batch, 256, d_model)

        # --- (E) Pool over the seq dimension to get a single vector per sample ---
        # e.g., global average:
        x = x.mean(dim=1)  # (batch, d_model)

        # --- (F) Final projection to 12 ---
        x = self.fc_out(x)  # (batch, 12)

        return x


# chatGPT
class CrossAttentionNet(nn.Module):
    def __init__(
        self, hidden_1d=16, hidden_2d=32, attn_dim=64, num_heads=4, output_channels=12
    ):
        super().__init__()

        # 1D path for the first 3 channels
        self.path1d = nn.Sequential(
            nn.Conv1d(3, hidden_1d, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_1d, hidden_1d, kernel_size=3, padding=1),
            nn.ReLU(),
            # ... more layers if needed ...
        )

        # 2D path for the remaining 65 channels
        self.path2d = nn.Sequential(
            nn.Conv2d(1, hidden_2d, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_2d, hidden_2d, kernel_size=3, padding=1),
            nn.ReLU(),
            # ... more layers if needed ...
        )

        # (Optional) linear to get key/value dimension = attn_dim
        self.to_keyvalue = nn.Linear(hidden_2d, attn_dim)
        # (Optional) linear to get query dimension = attn_dim
        self.to_query = nn.Linear(hidden_1d, attn_dim)

        # Multi-head attention module
        # For a single query vector per sample, we can do it carefully with
        # torch.nn.MultiheadAttention(attn_dim, num_heads),
        # but we need to shape the data properly.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, batch_first=True
        )

        # Final linear out
        # We'll produce a final embedding (attn_dim) and then go to 12
        self.fc_out = nn.Linear(attn_dim, output_channels)

    def forward(self, x):
        """
        x shape = (batch, 68, 256)
          first 3 = x[:, :3, :]
          next 65 = x[:, 3:, :]
        output = (batch, output_channels)
        """
        # ---------------- Path A: 1D (first 3 channels) ----------------
        x_1d = x[:, :3, :]  # (batch, 3, 256)
        x_1d = self.path1d(x_1d)  # (batch, hidden_1d, 256)
        # pool over time dimension to get a single vector:
        x_1d = F.adaptive_avg_pool1d(x_1d, 1).squeeze(-1)  # (batch, hidden_1d)

        # We'll transform it to attn_dim (query)
        q = self.to_query(x_1d)  # (batch, attn_dim)

        # ---------------- Path B: 2D (beamformer channels) ----------------
        x_2d = x[:, 3:, :]  # (batch, 65, 256)
        x_2d = x_2d.unsqueeze(1)  # (batch, 1, 65, 256)
        x_2d = self.path2d(
            x_2d
        )  # (batch, hidden_2d, H=65, W=256) [assuming no strides]

        # Flatten to sequence for attention:
        # Let's say x_2d is (batch, hidden_2d, 65, 256).
        B, C, H, W = x_2d.shape
        x_2d = x_2d.view(B, C, H * W)  # (batch, hidden_2d, 65*256)
        x_2d = x_2d.permute(0, 2, 1)  # (batch, seq_len=65*256, hidden_2d)

        # transform to attn_dim for keys/values
        kv = self.to_keyvalue(x_2d)  # (batch, seq_len, attn_dim)

        # ---------------- Cross Attention ----------------
        # MultiheadAttention in PyTorch expects (batch, seq, embed) if batch_first=True
        # But we have a single query per batch element -> shape (batch, 1, attn_dim)
        q = q.unsqueeze(1)  # (batch, 1, attn_dim)

        # q, kv => cross-attention
        # MHA returns (attn_output, attn_weights)
        attn_out, attn_weights = self.cross_attn(q, kv, kv)
        # attn_out shape: (batch, 1, attn_dim)
        # attn_weights shape: (batch, 1, seq_len)

        # We can just use attn_out's first dimension
        attn_out = attn_out.squeeze(1)  # (batch, attn_dim)

        # ---------------- Final MLP / linear ----------------
        out = self.fc_out(attn_out)  # (batch, output_channels)
        return out


# chatGPT
class Separate1D2DNet(nn.Module):
    """
    Split input into two parts:
      - First 3 channels (using 1D conv over time),
      - Remaining 65 channels (using 2D conv with "height"=65, "width"=256),
    then combine and project to final output.

    Input shape : (batch_size, 68, 256)
      - x[:,  :3, :] --> shape (batch, 3,   time=256)  -> 1D conv path
      - x[:, 3:, :] --> shape (batch, 65,  time=256)  -> 2D conv path
    Output shape: (batch_size, output_channels)
    """

    def __init__(
        self,
        hidden_channels_1d: int,
        num_layers_1d: int,
        hidden_channels_2d: int,
        num_layers_2d: int,
        output_channels: int = 12,
    ):
        super().__init__()

        # -------- 1D Conv Path for first 3 channels --------
        self.path1d = nn.ModuleList()
        in_ch_1d = 3  # The first path sees 3 channels
        for _ in range(num_layers_1d):
            self.path1d.append(
                nn.Conv1d(in_ch_1d, hidden_channels_1d, kernel_size=3, padding=1)
            )
            self.path1d.append(nn.ReLU(inplace=True))
            in_ch_1d = hidden_channels_1d

        # -------- 2D Conv Path for the remaining 65 channels --------
        self.path2d = nn.ModuleList()
        in_ch_2d = 1  # We'll treat (65,256) as a single-channel 2D “image”
        for _ in range(num_layers_2d):
            self.path2d.append(
                nn.Conv2d(in_ch_2d, hidden_channels_2d, kernel_size=3, padding=1)
            )
            self.path2d.append(nn.ReLU(inplace=True))
            in_ch_2d = hidden_channels_2d

        # Global pooling modules (we'll just use AdaptiveAvgPool)
        self.pool1d = None  # We'll use F.adaptive_avg_pool1d
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))

        # Final linear layer: input dim = hidden_channels_1d + hidden_channels_2d
        self.fc = nn.Linear(hidden_channels_1d + hidden_channels_2d, output_channels)

    def forward(self, x):
        """
        x shape: (batch, 68, 256)
        returns: (batch, output_channels)
        """
        # ---------------- Path A: 1D conv on first 3 channels ----------------
        x_1d = x[:, :3, :]  # shape (batch, 3, 256)
        for layer in self.path1d:
            x_1d = layer(x_1d)  # each Conv1d expects (batch, in_channels, time)

        # Global average pool over the time dimension: shape -> (batch, hidden_channels_1d)
        # x_1d is (batch, hidden_channels_1d, time), so we pool to (batch, hidden_channels_1d, 1)
        # then squeeze or flatten
        x_1d = F.adaptive_avg_pool1d(x_1d, 1).squeeze(
            -1
        )  # shape (batch, hidden_channels_1d)

        # ---------------- Path B: 2D conv on the last 65 channels ----------------
        x_2d = x[:, 3:, :]  # shape (batch, 65, 256)
        # turn (batch, 65, 256) into (batch, 1, 65, 256) so it looks like a single-channel 2D image
        x_2d = x_2d.unsqueeze(1)  # shape (batch, 1, 65, 256)

        for layer in self.path2d:
            x_2d = layer(x_2d)  # (batch, hidden_channels_2d, H, W)

        # Global pool to (batch, hidden_channels_2d, 1, 1) then flatten
        x_2d = self.pool2d(x_2d).view(x_2d.size(0), -1)  # (batch, hidden_channels_2d)

        # ---------------- Concatenate and project to final output ----------------
        x_cat = torch.cat(
            [x_1d, x_2d], dim=1
        )  # shape (batch, hidden_channels_1d + hidden_channels_2d)
        out = self.fc(x_cat)  # shape (batch, output_channels)

        return out


class PrepareInput(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        self.beamformer_input = global_config["beamformer_input"]
        self.beamformer_mag_input = global_config.get("beamformer_mag_input", False)
        if self.beamformer_mag_input:
            assert self.beamformer_input
        self.empirical_input = global_config["empirical_input"]
        self.phase_input = global_config["phase_input"]
        self.rx_spacing_input = global_config["rx_spacing_input"]
        self.gains_input = global_config["gains_input"]
        self.vehicle_type_input = global_config["vehicle_type_input"]
        self.inputs = 0
        self.frequency_input = global_config.get("frequency_input", False)
        self.input_dropout = model_config.get("input_dropout", 0.0)
        beamformer_dropout_p = model_config.get("beamformer_dropout", 0.0)

        if beamformer_dropout_p == 0.0:
            self.beamformer_dropout = nn.Identity()
        else:
            self.beamformer_dropout = nn.Dropout(beamformer_dropout_p)

        if self.beamformer_input:
            self.inputs += global_config["nthetas"]
        if self.beamformer_mag_input:
            self.inputs += 1
        if self.empirical_input:
            self.inputs += global_config["nthetas"]
        if self.phase_input:
            self.inputs += 3
        if self.gains_input:
            self.inputs += 2
        if self.rx_spacing_input:
            self.inputs += 1
        if self.frequency_input:
            self.inputs += 1
        if self.vehicle_type_input:
            self.inputs += 1
        assert (
            self.input_dropout == 0
            or (
                self.beamformer_input
                + self.empirical_input
                + self.phase_input
                + self.rx_spacing_input
                + self.vehicle_type_input
                + self.frequency_input
                + self.gains_input
            )
            > 1
        )

    def prepare_input(self, batch, additional_inputs=[]):
        dropout_mask = (
            torch.rand((7, *batch["y_rad"].shape), device=batch["y_rad"].device)
            < self.input_dropout
        )
        # 1 , 65
        # if mask out 65, then scale up 1 by 65?
        # [ batch, samples, dim of input ]
        # fo reach batch sample want to drop out for different samples
        # rand(batch, samples) for each input
        # if not selected just add 0
        # for those selected need to rescale
        # rescale by number of inputs or cumulative size of inputs?
        # generate random [0,1] for each input, then consider for dropout
        #
        inputs = []
        if self.beamformer_input:
            v_mean = batch["weighted_beamformer"].mean(axis=-1, keepdim=True) + 0.01
            v = batch["weighted_beamformer"] / (2 * v_mean)
            if self.beamformer_mag_input:
                v = torch.concatenate([v, v_mean], axis=-1)
            v = self.beamformer_dropout(v)
            if self.training:
                v[dropout_mask[0]] = 0
            inputs.append(v)
        if self.empirical_input:
            v = (batch["empirical"] - 1.0 / batch["empirical"].shape[-1]) * 10
            if self.training:
                v[dropout_mask[1]] = 0
            inputs.append(v)
        if self.phase_input:
            v = batch["weighted_windows_stats"]
            if self.training:
                v[dropout_mask[2]] = 0
            inputs.append(v)
        if self.rx_spacing_input:
            v = batch["rx_wavelength_spacing"][..., None] * 50
            if self.training:
                v[dropout_mask[3]] = 0
            inputs.append(v)
        if self.frequency_input:
            v = (batch["rx_lo"][..., None] + 1).log10() / 20
            if self.training:
                v[dropout_mask[4]] = 0
            inputs.append(v)
        if self.gains_input:
            v = batch["gains"] / 120
            if self.training:
                v[dropout_mask[5]] = 0
            inputs.append(v)
        if self.vehicle_type_input:
            v = batch["vehicle_type"][..., None]
            if self.training:
                v[dropout_mask[6]] = 0
            inputs.append(v)

        return torch.concatenate(
            inputs + additional_inputs,
            dim=2,
        )


def sigmoid_dist(x):
    x = torch.nn.functional.sigmoid(x)
    return x / x.sum(dim=-1, keepdim=True)


# Keep track of 10 different messages and then warn again
@lru_cache(1)
def warn_ntheta():
    logging.warning("output_ntheta is not specified, defaulting to global_config")


def check_and_load_ntheta(model_config, global_config):
    if "output_ntheta" not in model_config:
        warn_ntheta()
        model_config["output_ntheta"] = global_config["nthetas"]


def check_and_load_config(model_config, global_config):
    check_and_load_ntheta(model_config=model_config, global_config=global_config)


class SignalMatrixNet(nn.Module):
    def __init__(self, k=11, hidden_channels=32, outputs=8):
        super().__init__()
        padding = k // 2
        self.outputs = outputs
        self.conv_net = torch.nn.Sequential(
            nn.Conv1d(4, hidden_channels, k, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, k, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, k, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, k, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, k, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, k, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, k, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, outputs, k, stride=1, padding=padding),
        )

    def forward(self, x):
        return self.conv_net(x).mean(axis=2)


class SkipConnect(nn.Module):
    def __init__(self, internal_module):
        super().__init__()
        self.internal_module = internal_module

    def forward(self, x):
        out = self.internal_module(x)
        if out.shape == x.shape:
            return out + x
        return out


def create_1dconv(
    input_channels, hidden_channels, padding, k, norm_size, norm, n_layers, act, outputs
):

    layers = [
        SkipConnect(
            nn.Sequential(
                nn.Conv1d(
                    input_channels, hidden_channels, k, stride=1, padding=padding
                ),
                (nn.LayerNorm([hidden_channels, norm_size]) if norm else nn.Identity()),
                act(),
            )
        ),
    ]
    # added extra layer here
    for idx in range(n_layers - 1):
        layers += [
            SkipConnect(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        k,
                        stride=1,
                        padding=padding,
                    ),
                    (
                        nn.LayerNorm([hidden_channels, norm_size])
                        if norm
                        else nn.Identity()
                    ),
                    act(),
                )
            ),
        ]
    layers += [
        SkipConnect(
            nn.Sequential(
                nn.Conv1d(
                    hidden_channels, hidden_channels, k, stride=2, padding=padding
                ),
                (
                    nn.LayerNorm([hidden_channels, norm_size // 2])
                    if norm
                    else nn.Identity()
                ),
                act(),
            )
        ),
    ]
    for idx in range(n_layers):
        layers += [
            SkipConnect(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        k,
                        stride=1,
                        padding=padding,
                    ),
                    (
                        nn.LayerNorm([hidden_channels, norm_size // 2])
                        if norm
                        else nn.Identity()
                    ),
                    act(),
                )
            ),
        ]
    layers += [
        SkipConnect(
            nn.Sequential(
                nn.Conv1d(
                    hidden_channels, hidden_channels, k, stride=2, padding=padding
                ),
                (
                    nn.LayerNorm([hidden_channels, norm_size // 4])
                    if norm
                    else nn.Identity()
                ),
                act(),
            )
        ),
        nn.Conv1d(hidden_channels, outputs, k, stride=1, padding=padding),
    ]
    # self.conv_net = torch.nn.Sequential(*layers)
    return torch.nn.Sequential(*layers)


def create_2dconv(hidden_channels, padding, k, n_layers, act):
    # h = 68, w = 256
    layers = [
        SkipConnect(
            nn.Sequential(
                nn.Conv2d(1, hidden_channels, k, stride=1, padding=padding),
                act(),
            )
        ),
    ]
    # added extra layer here
    for idx in range(n_layers - 1):
        layers += [
            SkipConnect(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        k,
                        stride=1,
                        padding=padding,
                    ),
                    act(),
                )
            ),
        ]
    layers += [
        SkipConnect(
            nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    1,
                    k,
                    stride=1,
                    padding=padding,
                ),
                act(),
            )
        ),
    ]
    return torch.nn.Sequential(*layers)


class AllWindowsStatsNet(nn.Module):
    def __init__(
        self,
        k=11,
        hidden_channels=32,
        outputs=8,
        output_phi=False,
        norm=False,
        norm_size=256,
        n_layers=1,
        windowed_beamformer=False,
        normalize_windowed_beamformer=False,
        nthetas=0,
        act="relu",
        network_inputs=["all_windows_stats"],
        conv_type="1d",
        hidden_channels_2d=4,
        n_layers_2d=4,
        window_shrink=0.0,
        window_shuffle=0.0,
        window_dropout=0.0,
        window_fraction=0.25,
    ):
        super().__init__()
        if act == "relu":
            self.act = nn.ReLU
        elif act == "leaky":
            self.act = nn.LeakyReLU
        elif act == "selu":
            self.act = nn.SELU
        else:
            raise ValueError("invalid activation")
        padding = k // 2
        self.outputs = outputs
        self.output_phi = output_phi
        self.window_shrink = window_shrink
        self.window_shuffle = window_shuffle
        self.window_dropout = window_dropout
        self.window_fraction = window_fraction
        logging.info(
            f"ALLWINDOW CONFIG {self.window_fraction} {self.window_dropout} {self.window_shuffle}"
        )
        self.windowed_beamformer = windowed_beamformer
        self.normalize_windowed_beamformer = normalize_windowed_beamformer
        self.nthetas = nthetas
        assert (
            window_dropout == 0.0 or not norm
        ), "currently cannot do norm if dropout > 0.0"
        input_channels = 3
        if self.windowed_beamformer:
            input_channels += self.nthetas
        self.network_inputs = network_inputs

        self.conv_type = conv_type
        self.conv_net = create_1dconv(
            input_channels,
            hidden_channels,
            padding,
            k,
            norm_size,
            norm,
            n_layers,
            self.act,
            outputs,
        )
        if conv_type == "2d":
            self.conv_net_2d = create_2dconv(
                hidden_channels_2d,
                padding,
                k,
                n_layers,
                self.act,
            )
        if conv_type == "1d2d":
            self.conv_net = Separate1D2DNet(
                hidden_channels,
                n_layers,
                hidden_channels_2d,
                n_layers_2d,
                outputs,
            )
        if conv_type == "cross":
            self.conv_net = CrossAttentionNet(
                hidden_1d=hidden_channels,
                hidden_2d=hidden_channels_2d,
                attn_dim=64,
                num_heads=4,
                output_channels=outputs,
            )
        if conv_type == "transformer":
            TransformerTimeModel(
                d_model=hidden_channels,
                nhead=4,
                num_layers=n_layers,
                dim_feedforward=128,
                output_channels=12,
                use_positional_encoding=True,
            )

        if output_phi:
            self.phi_network = torch.nn.Sequential(
                nn.Linear(outputs, hidden_channels),
                self.act(),
                nn.Linear(hidden_channels, hidden_channels),
                self.act(),
                nn.Linear(hidden_channels, hidden_channels),
                self.act(),
                nn.Linear(hidden_channels, 1),
            )

    def forward(self, batch):
        inputs = []
        if "all_windows_stats" in self.network_inputs:
            normalize_by = batch["all_windows_stats"].max(dim=3, keepdims=True)[0]
            normalize_by[:, :, :2] = torch.pi

            all_windows_normalized_input = batch["all_windows_stats"] / (
                normalize_by + 1e-5
            )  # batch, snapshots, channels, time (256)

            # want B x C x L

            inputs.append(all_windows_normalized_input)

        if self.windowed_beamformer:
            if self.normalize_windowed_beamformer:
                inputs.append(
                    torch.nn.functional.normalize(
                        batch["windowed_beamformer"].transpose(2, 3), p=1, dim=2
                    )
                    - 1 / 65.0
                )
            else:
                inputs.append(batch["windowed_beamformer"].transpose(2, 3) / 500)
        input = torch.concatenate(inputs, dim=2)

        size_batch, size_snapshots, channels, windows = input.shape
        input = input.reshape(-1, channels, windows)

        if self.training:
            if self.window_shrink > 0.0 and torch.rand(1) < self.window_shrink:
                start_idx = int(torch.rand(1) * self.window_fraction * windows)
                end_idx = min(
                    start_idx + int(windows * (1 - self.window_fraction)), windows
                )
                input = input[:, :, start_idx:end_idx]

            # self.dropout = 0.2 means drop 20%
            elif self.window_dropout > 0.0:
                input = input[:, :, torch.rand(windows) > self.window_dropout]

            # assert input.isfinite().all()
            # shuffle 25% of the time
            if self.window_shuffle > 0.0 and torch.rand(1) < self.window_shuffle:
                input = input.index_select(
                    2, torch.randperm(input.shape[2], device=input.device)
                )

        if self.conv_type == "2d":
            input = self.conv_net_2d(input.unsqueeze(1)).select(1, 0)

        if self.conv_type in ["1d2d", "cross"]:
            output = self.conv_net(input)
        else:
            output = self.conv_net(input).mean(axis=2)
        r = {
            "all_windows_embedding": output.reshape(
                size_batch, size_snapshots, self.outputs
            )
        }
        if self.output_phi:
            r["output_phi"] = self.phi_network(r["all_windows_embedding"])
        return r


class SinglePointWithBeamformer(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        check_and_load_config(model_config=model_config, global_config=global_config)
        self.prepare_input = PrepareInput(model_config, global_config)

        additional_inputs = 0

        self.signal_matrix_net = None
        self.windows_stats_net = None
        if "signal_matrix_net" in model_config:
            self.signal_matrix_net = SignalMatrixNet(
                **model_config["signal_matrix_net"]
            )
            additional_inputs += self.signal_matrix_net.outputs
        if "windows_stats_net" in model_config:
            self.windows_stats_net = AllWindowsStatsNet(
                nthetas=global_config["nthetas"], **model_config["windows_stats_net"]
            )
            additional_inputs += self.windows_stats_net.outputs

        self.single_point_with_beamformer_ffnn = FFNN(
            inputs=self.prepare_input.inputs + additional_inputs,
            depth=model_config["depth"],  # 4
            hidden=model_config["hidden"],  # 128
            outputs=model_config["output_ntheta"],
            block=model_config["block"],  # True
            norm=model_config["norm"],  # False, [batch,layer]
            act=nn.LeakyReLU,
            dropout=model_config.get("dropout", model_config.get("dropout", 0.0)),
            # act=nn.SELU,
            bn=model_config["bn"],  # False , bool
        )

    def forward(self, batch):
        return_dict = {}
        # first dim odd / even is the radios
        additional_inputs = []
        if self.windows_stats_net:
            all_windows_output = self.windows_stats_net(batch)

            if "output_phi" in all_windows_output:
                return_dict["output_phi"] = all_windows_output["output_phi"]

            additional_inputs.append(all_windows_output["all_windows_embedding"])

            # breakpoint()
            # a = 1
        if self.signal_matrix_net:
            normalize_by = batch["abs_signal_and_phase_diff"].max(dim=3, keepdims=True)[
                0
            ]

            normalize_by[:, :, 2] = torch.pi
            input = batch["abs_signal_and_phase_diff"] / normalize_by

            batch["rx_wavelength_spacing"][:, :, None, None].expand(
                batch["rx_lo"].shape + (1, batch["abs_signal_and_phase_diff"].shape[-1])
            )

            input_with_spacing = torch.concatenate(
                [
                    input,
                    batch["rx_wavelength_spacing"][:, :, None, None].expand(
                        batch["rx_lo"].shape
                        + (1, batch["abs_signal_and_phase_diff"].shape[-1])
                    ),
                ],
                dim=2,
            )
            additional_inputs.append(
                self.signal_matrix_net(input_with_spacing.select(1, 0))[:, None]
            )
        return_dict["single"] = torch.nn.functional.normalize(
            self.single_point_with_beamformer_ffnn(
                self.prepare_input.prepare_input(batch, additional_inputs),
            ).abs(),
            dim=2,
            p=1,
        )
        return return_dict


class PairedSinglePointWithBeamformer(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        check_and_load_config(model_config=model_config, global_config=global_config)
        self.single_radio_net = SinglePointWithBeamformer(
            model_config["single"], global_config
        )
        self.prepare_input = PrepareInput(model_config["single"], global_config)

        self.detach = model_config.get("detach", True)
        self.paired_single_point_with_beamformer_ffnn = FFNN(
            inputs=(model_config["single"]["output_ntheta"] + self.prepare_input.inputs)
            * 2,
            depth=model_config["depth"],  # 4
            hidden=model_config["hidden"],  # 128
            outputs=model_config["output_ntheta"],
            block=model_config["block"],  # True
            norm=model_config["norm"],  # False, [batch,layer]
            act=nn.LeakyReLU,
            bn=model_config["bn"],  # False , bool
            dropout=model_config.get("dropout", 0.0),
        )

    def forward(self, batch):
        single_radio_estimates = self.single_radio_net(batch)["single"]

        # rotate to align with respect to craft theta
        rotation_offets = batch["rx_theta_in_pis"] * torch.pi
        single_radio_estimates_rotated = rotate_dist(
            single_radio_estimates[:, 0],
            torch_pi_norm(rotation_offets),
        ).unsqueeze(1)

        # these inputs are invariant
        additional_inputs = self.prepare_input.prepare_input(batch)

        joined_input = torch.concatenate(
            [single_radio_estimates_rotated, additional_inputs], dim=2
        )

        joined_input_detached_or_not = detach_or_not(joined_input, self.detach)

        x = self.paired_single_point_with_beamformer_ffnn(
            torch.concatenate(
                [joined_input_detached_or_not[::2], joined_input_detached_or_not[1::2]],
                dim=2,
            )
        )

        idxs = torch.arange(x.shape[0]).reshape(-1, 1).repeat(1, 2).reshape(-1)
        # first dim odd / even is the radios
        return {
            "single": single_radio_estimates,
            # "paired": torch.nn.functional.softmax(x[idxs], dim=2),
            # "paired": x[idxs],
            "paired": torch.nn.functional.normalize(x[idxs].abs(), p=1, dim=2),
        }


class NormP1Dim2(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x.abs(), p=1, dim=2)


def detach_or_not(x, detach):
    if detach:
        return x.detach()
    return x


class PairedMultiPointWithBeamformer(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        check_and_load_config(model_config=model_config, global_config=global_config)
        self.multi_radio_net = PairedSinglePointWithBeamformer(
            model_config["paired"], global_config
        )
        self.detach = model_config.get("detach", True)
        self.transformer_config = model_config["transformer"]
        self.d_model = self.transformer_config["d_model"]
        self.skip_connection = self.transformer_config.get("skip_connection", True)
        self.use_xy = model_config.get("use_xy", False)
        self.output_ntheta = model_config["output_ntheta"]

        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.transformer_config["n_heads"],
            dim_feedforward=self.transformer_config["d_hid"],
            dropout=self.transformer_config.get("dropout", 0.0),
            activation="gelu",
            batch_first=True,  # batch, sequence, feature
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            self.transformer_config["n_layers"],
            LayerNorm(self.d_model),
        )
        input_size = model_config["paired"]["output_ntheta"] + 1
        if self.use_xy:
            input_size += 2
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(
                input_size, self.d_model
            )  # 5 output beam_former R1+R2, time
        )
        # self.output_net = torch.nn.Sequential(
        #     torch.nn.Linear(self.d_model, 65),  # 5 output beam_former R1+R2, time
        #     NormP1Dim2(),
        # )
        self.output_net = torch.nn.Linear(
            self.d_model, self.output_ntheta + 1
        )  # output discrete and actual
        self.norm = NormP1Dim2()
        self.input_dropout = torch.nn.Dropout1d(model_config.get("input_dropout", 0.0))

    def forward(self, batch):

        output = self.multi_radio_net(batch)

        if self.training and (torch.rand(1) > 0.5).item():
            # if training can "double" data by flipping time backwards
            normalized_time = (
                batch["system_timestamp"][:, [-1]] - batch["system_timestamp"]
            )
            normalized_time /= normalized_time[:, [0]] + 1e-3
        else:
            normalized_time = (
                batch["system_timestamp"] - batch["system_timestamp"][:, [0]]
            )
            normalized_time /= normalized_time[:, [-1]] + 1e-3
        # normalized_time *= 0.0

        if self.use_xy:
            normalized_positions = (
                batch["rx_pos_xy"] - batch["rx_pos_xy"][..., [0], :]
            )  # / 200
            normalized_positions = normalized_positions / (
                normalized_positions.max(axis=2, keepdim=True)[0].max(
                    axis=1, keepdim=True
                )[0]
                + 1e-3
            )
            assert normalized_positions.isfinite().all()
        output_paired = detach_or_not(output["paired"], self.detach)

        if self.use_xy:
            input_with_time = torch.concatenate(
                [
                    self.input_dropout(output_paired),
                    normalized_time.unsqueeze(2),
                    normalized_positions,
                ],
                axis=2,
            )
        else:
            input_with_time = torch.concatenate(
                [
                    self.input_dropout(output_paired),
                    normalized_time.unsqueeze(2),
                ],
                axis=2,
            )
        full_output = self.output_net(
            self.transformer_encoder(self.input_net(input_with_time))
        )
        discrete_output = full_output[..., : self.output_ntheta]
        direct_output = full_output[..., [-1]]

        if self.skip_connection:
            output["multipaired"] = self.norm(
                output_paired + discrete_output  # skip connection for paired
            )
        else:
            output["multipaired"] = self.norm(
                discrete_output  # skip connection for paired
            )

        output["multipaired_direct"] = direct_output
        return output


class TrajPairedMultiPointWithBeamformer(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        check_and_load_config(model_config=model_config, global_config=global_config)
        self.multi_radio_net = PairedSinglePointWithBeamformer(
            model_config["paired"], global_config
        )
        self.detach = model_config.get("detach", True)
        self.transformer_config = model_config["transformer"]
        self.d_model = self.transformer_config["d_model"]
        self.skip_connection = self.transformer_config.get("skip_connection", False)
        self.use_xy = model_config.get("use_xy", False)
        self.pred_xy = model_config.get("pred_xy", False)
        self.output_ntheta = model_config["output_ntheta"]

        self.latent = model_config["latent"]  # 8

        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.transformer_config["n_heads"],
            dim_feedforward=self.transformer_config["d_hid"],
            dropout=self.transformer_config.get("dropout", 0.0),
            activation="gelu",
            batch_first=True,  # batch, sequence, feature
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            self.transformer_config["n_layers"],
            LayerNorm(self.d_model),
        )
        input_size = model_config["paired"]["output_ntheta"] + 1
        if self.use_xy:
            input_size += 2
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(
                input_size, self.d_model
            )  # 5 output beam_former R1+R2, time
        )
        # self.output_net = torch.nn.Sequential(
        #     torch.nn.Linear(self.d_model, 65),  # 5 output beam_former R1+R2, time
        #     NormP1Dim2(),
        # )
        self.output_net = torch.nn.Linear(
            self.d_model, self.latent
        )  # output discrete and actual
        self.norm = NormP1Dim2()
        self.input_dropout = torch.nn.Dropout1d(model_config.get("input_dropout", 0.0))

        traj_net_input_dim = self.latent + 1
        if self.use_xy:
            traj_net_input_dim += 2
        self.traj_net = FFNN(
            inputs=traj_net_input_dim,
            depth=model_config["traj_layers"],
            hidden=model_config["traj_hidden"],
            outputs=self.output_ntheta + 1 + 2,  # predict tx pos
            block=True,
            bn=True,
            norm="layer",
            act=torch.nn.LeakyReLU,
            dropout=0.0,
        )

    def forward(self, batch):
        output = self.multi_radio_net(batch)

        batch_size, snapshots_per_example = batch["system_timestamp"].shape

        normalized_time = batch["system_timestamp"] - batch["system_timestamp"][:, [0]]
        normalized_time /= normalized_time[:, [-1]] + 1e-3
        # normalized_time *= 0.0

        if self.use_xy:
            normalized_positions = (
                batch["rx_pos_xy"] - batch["rx_pos_xy"][..., [0], :]
            )  # / 200
            # normalized_positions = normalized_positions / (
            #     normalized_positions.max(axis=2, keepdim=True)[0].max(
            #         axis=1, keepdim=True
            #     )[0]
            #     + 1e-3
            # )
            assert normalized_positions.isfinite().all()
        output_paired = detach_or_not(output["paired"], self.detach)

        if self.use_xy:
            input_with_time = torch.concatenate(
                [
                    self.input_dropout(output_paired),
                    normalized_time.unsqueeze(2),
                    normalized_positions,
                ],
                axis=2,
            )
        else:
            input_with_time = torch.concatenate(
                [
                    self.input_dropout(output_paired),
                    normalized_time.unsqueeze(2),
                ],
                axis=2,
            )
        transformer_output = self.output_net(
            self.transformer_encoder(self.input_net(input_with_time))
        )

        trajectories = transformer_output[:, [0]].expand(
            batch_size, snapshots_per_example, self.latent
        )

        if self.use_xy:
            trajectories_with_time = torch.concatenate(
                [trajectories, normalized_time.unsqueeze(2), normalized_positions],
                dim=2,
            )
        else:
            trajectories_with_time = torch.concatenate(
                [trajectories, normalized_time.unsqueeze(2)], dim=2
            )

        traj_output = self.traj_net(trajectories_with_time)

        discrete_output = traj_output[..., : self.output_ntheta]
        tx_preds = traj_output[..., self.output_ntheta : self.output_ntheta + 2]
        direct_output = traj_output[..., [-1]]

        output["multipaired"] = self.norm(discrete_output)  # skip connection for paired

        output["multipaired_direct"] = direct_output
        if self.pred_xy:
            output["multipaired_tx_pos"] = tx_preds
        return output


class SinglePointPassThrough(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        # weighted_beamformer(batch)
        return {"single": batch["empirical"] + self.w * 0.0}
