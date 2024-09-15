from functools import cache

import numpy as np
import torch
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer

from spf.model_training_and_inference.models.beamsegnet import (
    BeamNetDirect,
    BeamNetDiscrete,
    SimpleNet,
)
from spf.rf import torch_pi_norm, torch_pi_norm_pi, torch_reduce_theta_to_positive_y
from spf.utils import PositionalEncoding

DIST_NORM = 4000


# @torch.no_grad
# @torch.compile
def random_loss(target: torch.Tensor, y_rad_reduced: torch.Tensor):
    random_target = (torch.rand(target.shape, device=target.device) - 0.5) * 2 * np.pi
    beamnet_mse_random = (
        torch_pi_norm(
            y_rad_reduced
            - (torch.rand(y_rad_reduced.shape, device=target.device) - 0.5)
            * 2
            * np.pi
            / 2,
            max_angle=torch.pi / 2,
        )
        ** 2
    ).mean()
    transformer_random_loss = (torch_pi_norm_pi(target - random_target) ** 2).mean()
    return beamnet_mse_random, transformer_random_loss


@cache
def get_time_split(batch_size, snapshots_per_sessions, device, dtype):
    return (
        torch.linspace(-1, 0, snapshots_per_sessions, device=device, dtype=dtype)
        .reshape(1, -1, 1)
        .expand(batch_size // 2, snapshots_per_sessions, 1)
    )


class DebugFunkyNet(torch.nn.Module):
    def __init__(
        self,
        input_dim=10,
        d_model=2048,
        d_hid=512,
        dropout=0.1,
        n_heads=16,
        n_layers=24,
        output_dim=1,
        token_dropout=0.0,
        only_beamnet=False,
    ):
        super(DebugFunkyNet, self).__init__()
        self.beam_m = torch.nn.Identity()
        self.l = torch.nn.Linear(3, 1).to(torch.float32)

    def forward(
        self,
        x,
        seg_mask,
        rx_spacing,
        y_rad,
        windowed_beam_former,
        rx_pos,
        timestamps,
    ):
        return {
            "transformer_output": None,
            "pred_theta": None,
            "fake_out": self.l(x[:, 0, 0, :3]),
        }

    def loss(
        self,
        output: torch.Tensor,
        y_rad: torch.Tensor,
        craft_y_rad: torch.Tensor,
        seg_mask: torch.Tensor,
        tx_pos_full: torch.Tensor,
    ):
        loss = output["fake_out"].mean()
        return {
            "loss": loss,
            "transformer_mse_loss": loss,
            "beamnet_loss": loss,
            "beamnet_mse_loss": loss,
            "beamnet_mse_random_loss": loss,
            "transformer_mse_random_loss": loss,
        }


class FunkyNet(torch.nn.Module):
    def __init__(
        self,
        args,
        d_model=2048,
        d_hid=512,
        dropout=0.1,
        n_heads=16,
        n_layers=24,
        output_dim=1 + 2,  # 1 theta, 2 delta xy pos
        token_dropout=0.5,
        latent=0,
        beamformer_input=False,
        include_input=True,
        only_beamnet=False,
        positional_encoding=True,
    ):
        super(FunkyNet, self).__init__()

        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_hid,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            n_layers,
            LayerNorm(d_model),
        )
        self.beamformer_input = beamformer_input
        self.output_dim = output_dim
        self.include_input = include_input
        self.only_beamnet = only_beamnet

        if args.beam_type == "simple":

            self.beam_m = SimpleNet(
                nthetas=65,
                depth=args.beam_net_depth,
                hidden=args.beam_net_hidden,
                symmetry=False,
                act=torch.nn.LeakyReLU,
                other=True,
                bn=args.beam_norm,
                no_sigmoid=True,
                block=True,
                inputs=3 + 1,  # + 1,  # 3 basic + 1 rx_spacing
                norm=args.beam_norm_type,
                positional_encoding=False,
                latent=latent,
                max_angle=np.pi / 2,
                linear_sigmas=True,
                correction=True,
                min_sigma=0.0001,
            )  # .to(torch_device)
        elif args.beam_type == "direct":
            if beamformer_input:
                self.beam_m = BeamNetDirect(
                    nthetas=65,
                    depth=args.beam_net_depth,
                    hidden=args.beam_net_hidden,
                    symmetry=False,
                    act=torch.nn.LeakyReLU,
                    other=True,
                    bn=args.beam_norm,  # True
                    no_sigmoid=True,
                    block=True,
                    rx_spacing_track=-1,
                    pd_track=-1,
                    mag_track=-1,
                    stddev_track=-1,
                    inputs=65,
                    latent=latent,
                    norm=args.beam_norm.type,
                    max_angle=torch.pi / 2,
                    linear_sigmas=True,
                    correction=True,
                    min_sigma=0.0001,
                )

                if self.include_input:
                    input_dim = (65 + 2) * 2
                    # (65,65,2,2) # 65 for R0 signal, 65 for R1 signal, 2 for pos0, 2 for pos1
                    self.input_net = torch.nn.Sequential(
                        torch.nn.Linear(
                            input_dim + (5 + latent) * 2 + 1, d_model
                        )  # 5 output beam_former R1+R2, time
                    )
                else:
                    input_dim = 2 * 2
                    # (2,2) # 2 for pos0, 2 for pos1
                    self.input_net = torch.nn.Sequential(
                        torch.nn.Linear(
                            input_dim + (5 + latent) * 2 + 1, d_model
                        )  # 5 output beam_former R1+R2, time
                    )
            else:
                self.beam_m = BeamNetDirect(
                    nthetas=65,
                    depth=args.beam_net_depth,
                    hidden=args.beam_net_hidden,
                    symmetry=False,
                    act=torch.nn.LeakyReLU,
                    other=True,
                    bn=args.beam_norm,
                    no_sigmoid=True,
                    block=True,
                    inputs=3 + 1,  # 3 basic + 1 rx_spacing
                    norm=args.beam_norm_type,
                    positional_encoding=False,
                    latent=latent,
                    max_angle=np.pi / 2,
                    linear_sigmas=True,
                    correction=True,
                    min_sigma=0.0001,
                )  # .to(torch_device)

                if self.include_input:
                    input_dim = (
                        4 + 4 + 2 + 2
                    )  # (4,4,2,2) # 4 for R0 signal, 4 for R1 signal, 2 for pos0, 2 for pos1
                    self.input_net = torch.nn.Sequential(
                        torch.nn.Linear(
                            input_dim + (5 + latent) * 2 + 1, d_model
                        )  # 5 output beam_former R1+R2, time
                    )
                else:
                    input_dim = 4  # (2,2) #  2 for pos0, 2 for pos1
                    self.input_net = torch.nn.Sequential(
                        torch.nn.Linear(
                            input_dim + (5 + latent) * 2 + 1, d_model
                        )  # 5 output beam_former R1+R2, time
                    )
        else:
            nthetas = 65
            self.beam_m = BeamNetDiscrete(
                nthetas=nthetas,
                depth=args.beam_net_depth,
                hidden=args.beam_net_hidden,
                symmetry=False,
                act=torch.nn.LeakyReLU,
                bn=args.beam_norm,
                block=True,
                inputs=3 + 1,  # 3 basic + 1 rx_spacing
                norm=args.beam_norm_type,
                positional_encoding=args.positional,
                latent=latent,
                max_angle=np.pi / 2,
            )  # .to(torch_device)

            if self.include_input:
                input_dim = (
                    4 + 4 + 2 + 2
                )  # (4,4,2,2) # 3 for R0 signal, 3 for R1 signal, 2 for pos0, 2 for pos1
                self.input_net = torch.nn.Sequential(
                    torch.nn.Linear(
                        input_dim + (nthetas + latent) * 2 + 1, d_model
                    )  # 5 output beam_former R1+R2, time
                )
            else:
                input_dim = 4  # (2,2) #  2 for pos0, 2 for pos1
                self.input_net = torch.nn.Sequential(
                    torch.nn.Linear(
                        input_dim + (nthetas + latent) * 2 + 1, d_model
                    )  # 5 output beam_former R1+R2, time
                )

        self.paired_drop_in_gt = 1.00
        self.token_dropout = token_dropout

        self.positional_encoding = positional_encoding
        self.pe = PositionalEncoding(d_model=d_model, dropout=0.0)

    # @torch.compile
    def forward(
        self,
        x,
        seg_mask,
        rx_spacing,
        y_rad,
        windowed_beam_former,
        rx_pos,
        timestamps,
    ):
        rx_pos = rx_pos.detach().clone() / DIST_NORM

        batch_size, snapshots_per_sessions = y_rad.shape
        # weighted_input = torch.mul(x, seg_mask).sum(axis=3) / (
        #     seg_mask.sum(axis=3) + 0.001
        # )
        if self.beamformer_input:
            windowed_beam_former_scaled = windowed_beam_former / (
                seg_mask.sum(axis=3, keepdim=True) + 0.1
            )
            weighted_input = (
                windowed_beam_former_scaled * seg_mask[:, 0][..., None]
            ).mean(axis=2)

        else:
            weighted_input = (
                torch.mul(x, seg_mask) / (seg_mask.sum(axis=3, keepdim=True) + 0.001)
            ).sum(axis=3)
            # weighted_input ~ (batch_size*2,1,3)
        # add rx_spacing (batch_size*2,1,4)
        weighted_input = torch.concatenate(
            [weighted_input, rx_spacing.unsqueeze(2)], dim=2
        )
        pred_theta = self.beam_m(
            weighted_input.reshape(batch_size * snapshots_per_sessions, -1)
        )
        if self.only_beamnet:
            return {"pred_theta": pred_theta}
        # if not pred_theta.isfinite().all():
        #    breakpoint()
        #    a = 1
        detached_pred_theta = torch.hstack(
            [
                pred_theta[:, : self.beam_m.outputs - self.beam_m.latent].detach(),
                pred_theta[:, self.beam_m.outputs - self.beam_m.latent :],
            ]
        )
        # breakpoint()
        # TODO inject correct means sometimes!
        # if self.train() randomly inject
        # if self.eval() never inject!
        if y_rad is not None and self.training and self.paired_drop_in_gt > 0.0:
            y_rad_reduced = torch_reduce_theta_to_positive_y(y_rad).reshape(-1, 1)
            mask = torch.rand(detached_pred_theta.shape[0]) < self.paired_drop_in_gt
            # detached_pred_theta[mask, 0] = y_rad_reduced[mask, 0]
            detached_pred_theta[mask, 1:3] = 0
        detached_pred_theta = detached_pred_theta.reshape(
            batch_size, snapshots_per_sessions, -1
        )

        weighted_input_by_example = weighted_input.reshape(
            batch_size, snapshots_per_sessions, weighted_input.shape[-1]
        )
        # breakpoint()
        rx_pos_by_example = rx_pos.reshape(batch_size, snapshots_per_sessions, 2)

        rel_pos0 = (
            rx_pos_by_example[::2] - rx_pos_by_example[::2][:, [-1]]
        )  # batch,snapshots_per_session,2
        rel_pos1 = (
            rx_pos_by_example[1::2] - rx_pos_by_example[::2][:, [-1]]
        )  # batch,snapshots_per_session,2

        if self.include_input:
            input = torch.concatenate(
                [
                    weighted_input_by_example[::2],
                    weighted_input_by_example[1::2],
                    rel_pos0,
                    rel_pos1,
                    detached_pred_theta[::2],
                    detached_pred_theta[1::2],
                    get_time_split(
                        batch_size,
                        snapshots_per_sessions,
                        weighted_input_by_example.device,
                        dtype=weighted_input_by_example.dtype,
                    ),
                ],
                axis=2,
            )
        else:
            input = torch.concatenate(
                [
                    rel_pos0,
                    rel_pos1,
                    detached_pred_theta[::2],  # batch,snapshots_per_session,5
                    detached_pred_theta[1::2],  # batch,snapshots_per_session,5
                    get_time_split(
                        batch_size,
                        snapshots_per_sessions,
                        weighted_input_by_example.device,
                        dtype=weighted_input_by_example.dtype,
                    ),
                ],
                axis=2,
            )

        embedded_input = self.input_net(input)
        if self.positional_encoding:
            embedded_input = self.pe(embedded_input)
        # drop out 1/4 of the sequence, except the last (element we predict on)
        if self.training and self.token_dropout > 0.0:
            assert 1 == 0, "need to fix all target loss"
            src_key_padding_mask = (
                torch.rand(batch_size // 2, snapshots_per_sessions, device=input.device)
                < self.token_dropout  # True here means skip
            )
            src_key_padding_mask[:, -1] = False  # True is not allowed to attend
            transformer_output = self.transformer_encoder(
                embedded_input, src_key_padding_mask=src_key_padding_mask
            )[:, :, : self.output_dim]
            # breakpoint()
            # a = 1
        else:
            transformer_output = self.transformer_encoder(embedded_input)[
                :, :, : self.output_dim
            ]

        return {
            "transformer_output": transformer_output,
            "pred_theta": pred_theta,
            "rx_pos_full": rx_pos_by_example,
        }

    # @torch.compile
    def loss(
        self,
        output: torch.Tensor,
        y_rad: torch.Tensor,
        craft_y_rad: torch.Tensor,
        seg_mask: torch.Tensor,
        tx_pos_full: torch.Tensor,
    ):
        single_target = craft_y_rad[::2, [-1]]
        all_target = craft_y_rad[::2]

        if not self.only_beamnet:
            single_transformer_loss = (
                torch_pi_norm_pi(
                    single_target - output["transformer_output"][:, -1, [0]]
                )
                ** 2
            ).mean()
            all_transformer_loss = (
                torch_pi_norm_pi(all_target - output["transformer_output"][:, :, 0])
                ** 2
            ).mean()
            tx_pos = (tx_pos_full[::2] + tx_pos_full[1::2]) / 2
            rx_pos = (output["rx_pos_full"][::2] + output["rx_pos_full"][1::2]) / 2
            delta = (tx_pos - rx_pos) / DIST_NORM
            pos_transformer_mse_loss = (
                (delta - output["transformer_output"][:, :, 1:3]) ** 2
            ).mean()
        else:
            single_transformer_loss = torch.tensor(0.0)
            all_transformer_loss = torch.tensor(0.0)
            pos_transformer_mse_loss = torch.tensor(0.0)

        y_rad_reduced = torch_reduce_theta_to_positive_y(y_rad).reshape(-1, 1)
        # x to beamformer loss (indirectly including segmentation)
        beamnet_loss = -self.beam_m.loglikelihood(
            output["pred_theta"], y_rad_reduced
        ).mean()

        beamnet_mse = self.beam_m.mse(output["pred_theta"], y_rad_reduced)

        # loss = all_transformer_loss + beamnet_loss + pos_transformer_mse_loss

        # hack for simple net TODO
        loss = all_transformer_loss + beamnet_mse  # + pos_transformer_mse_loss
        # beamnet_loss = beamnet_mse

        # loss = pos_transformer_mse_loss

        beamnet_mse_random, transformer_random_loss = random_loss(
            single_target, y_rad_reduced
        )
        return {
            "loss": loss,
            "transformer_mse_loss": single_transformer_loss,
            "all_transformer_mse_loss": all_transformer_loss,
            "beamnet_loss": beamnet_loss,
            "beamnet_mse_loss": beamnet_mse,
            "beamnet_mse_random_loss": beamnet_mse_random,
            "transformer_mse_random_loss": transformer_random_loss,
            "pos_transformer_mse_loss": pos_transformer_mse_loss,
            "mm_pos_transformer_mse_loss": pos_transformer_mse_loss
            * DIST_NORM
            * DIST_NORM,
        }
