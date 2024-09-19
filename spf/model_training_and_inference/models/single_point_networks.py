import torch
from torch import nn

from spf.model_training_and_inference.models.beamsegnet import FFNN

TEMP = 10


class PrepareInput:
    def __init__(self, model_config, global_config):
        self.beamformer_input = global_config["beamformer_input"]
        self.empirical_input = global_config["empirical_input"]
        self.phase_input = global_config["phase_input"]
        self.rx_spacing_input = global_config["rx_spacing_input"]
        self.inputs = 0
        if self.beamformer_input:
            self.inputs += global_config["nthetas"]
        if self.empirical_input:
            self.inputs += global_config["nthetas"]
        if self.phase_input:
            self.inputs += 3
        if self.rx_spacing_input:
            self.inputs += 1

    def prepare_input(self, batch):
        inputs = []
        if self.beamformer_input:
            inputs.append(
                # (batch["weighted_beamformer"] / 256) - 0.5
                batch["weighted_beamformer"]
                / (batch["weighted_beamformer"].max(axis=-1, keepdim=True)[0] + 0.1)
            )  # 256 just for random scaling down of beamformer not related to shape
        if self.empirical_input:
            inputs.append(
                (batch["empirical"] - 1.0 / batch["empirical"].shape[-1]) * 10
            )
        if self.phase_input:
            inputs.append(batch["weighted_windows_stats"])
        if self.rx_spacing_input:
            inputs.append(batch["rx_spacing"][..., None] * 50)
        return torch.concatenate(inputs, axis=2)


def sigmoid_dist(x):
    x = torch.nn.functional.sigmoid(x)
    return x / x.sum(dim=-1, keepdim=True)


class SinglePointWithBeamformer(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        self.prepare_input = PrepareInput(model_config, global_config)
        self.net = FFNN(
            inputs=self.prepare_input.inputs,
            depth=model_config["depth"],  # 4
            hidden=model_config["hidden"],  # 128
            outputs=global_config["nthetas"],
            block=model_config["block"],  # True
            norm=model_config["norm"],  # False, [batch,layer]
            act=nn.LeakyReLU,
            # act=nn.SELU,
            bn=model_config["bn"],  # False , bool
        )
        self.paired = False

    def forward(self, batch):
        # first dim odd / even is the radios
        return {
            # "single": torch.nn.functional.softmax(
            #     self.net(self.prepare_input.prepare_input(batch)) / TEMP,
            #     dim=2,
            # )
            # "single": sigmoid_dist(self.net(self.prepare_input.prepare_input(batch)))
            # "single": self.net(self.prepare_input.prepare_input(batch))
            # / 50
            "single": torch.nn.functional.normalize(
                self.net(self.prepare_input.prepare_input(batch)).abs(), dim=2, p=1
            )
        }


class PairedSinglePointWithBeamformer(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        self.single_radio_net = SinglePointWithBeamformer(model_config, global_config)

        self.net = FFNN(
            inputs=global_config["nthetas"] * 2,
            depth=model_config["depth"],  # 4
            hidden=model_config["hidden"],  # 128
            outputs=global_config["nthetas"],
            block=model_config["block"],  # True
            norm=model_config["norm"],  # False, [batch,layer]
            act=nn.LeakyReLU,
            bn=model_config["bn"],  # False , bool
        )
        self.paired = True

    def forward(self, batch):
        single_radio_estimates = self.single_radio_net(batch)["single"]
        x = self.net(
            torch.concatenate(
                [single_radio_estimates[::2], single_radio_estimates[1::2]], dim=2
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


class SinglePointPassThrough(nn.Module):
    def __init__(self, model_config, global_config):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.paired = False

    def forward(self, batch):
        # weighted_beamformer(batch)
        return {"single": batch["empirical"] + self.w * 0.0}
