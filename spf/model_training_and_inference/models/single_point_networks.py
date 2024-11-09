import logging

import torch
from torch import nn
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer

from spf.model_training_and_inference.models.beamsegnet import FFNN

TEMP = 10


class PrepareInput(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        self.beamformer_input = global_config["beamformer_input"]
        self.empirical_input = global_config["empirical_input"]
        self.phase_input = global_config["phase_input"]
        self.rx_spacing_input = global_config["rx_spacing_input"]
        self.inputs = 0
        self.input_dropout = model_config.get("input_dropout", 0.0)
        if self.beamformer_input:
            self.inputs += global_config["nthetas"]
        if self.empirical_input:
            self.inputs += global_config["nthetas"]
        if self.phase_input:
            self.inputs += 3
        if self.rx_spacing_input:
            self.inputs += 1

    def prepare_input(self, batch):
        dropout_mask = (
            torch.rand((4, *batch["y_rad"].shape), device=batch["y_rad"].device)
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
            v = batch["weighted_beamformer"] / (
                batch["weighted_beamformer"].max(axis=-1, keepdim=True)[0] + 0.1
            )
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
        return torch.concatenate(inputs, axis=2)


def sigmoid_dist(x):
    x = torch.nn.functional.sigmoid(x)
    return x / x.sum(dim=-1, keepdim=True)


def check_and_load_ntheta(model_config, global_config):
    if "output_ntheta" not in model_config:
        logging.warning("output_ntheta is not specified, defaulting to global_config")
        model_config["output_ntheta"] = global_config["nthetas"]


def check_and_load_config(model_config, global_config):
    check_and_load_ntheta(model_config=model_config, global_config=global_config)


class SinglePointWithBeamformer(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        check_and_load_config(model_config=model_config, global_config=global_config)
        self.prepare_input = PrepareInput(model_config, global_config)
        self.single_point_with_beamformer_ffnn = FFNN(
            inputs=self.prepare_input.inputs,
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
        # first dim odd / even is the radios
        return {
            "single": torch.nn.functional.normalize(
                self.single_point_with_beamformer_ffnn(
                    self.prepare_input.prepare_input(batch)
                ).abs(),
                dim=2,
                p=1,
            )
        }


class PairedSinglePointWithBeamformer(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        check_and_load_config(model_config=model_config, global_config=global_config)
        self.single_radio_net = SinglePointWithBeamformer(
            model_config["single"], global_config
        )
        self.detach = model_config.get("detach", True)
        self.paired_single_point_with_beamformer_ffnn = FFNN(
            inputs=model_config["single"]["output_ntheta"] * 2,
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

        single_radio_estimates_input = detach_or_not(
            single_radio_estimates, self.detach
        )

        x = self.paired_single_point_with_beamformer_ffnn(
            torch.concatenate(
                [single_radio_estimates_input[::2], single_radio_estimates_input[1::2]],
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
