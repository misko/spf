from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from spf.dataset.spf_dataset import v5_thetas_to_targets
from spf.rf import torch_circular_mean


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, act=nn.LeakyReLU, bn=False):
        super(ConvNet, self).__init__()
        self.ks = 17
        self.act = act
        self.net = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=hidden,
                            kernel_size=self.ks,
                            padding=self.ks // 2,
                            # bias=False,
                        ),
                    ),
                    (
                        "norm1",
                        nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity(),
                    ),
                    ("relu1", self.act()),
                    (
                        "conv2",
                        nn.Conv1d(
                            in_channels=hidden,
                            out_channels=hidden,
                            kernel_size=self.ks,
                            padding=self.ks // 2,
                            # bias=False,
                        ),
                    ),
                    (
                        "norm2",
                        nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity(),
                    ),
                    ("relu2", self.act()),
                    (
                        "conv3",
                        nn.Conv1d(
                            in_channels=hidden,
                            out_channels=hidden,
                            kernel_size=self.ks,
                            padding=self.ks // 2,
                            # bias=False,
                        ),
                    ),
                    (
                        "norm3",
                        nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity(),
                    ),
                    ("relu3", self.act()),
                    (
                        "conv4",
                        nn.Conv1d(
                            in_channels=hidden,
                            out_channels=hidden,
                            kernel_size=self.ks,
                            padding=self.ks // 2,
                            # bias=False,
                        ),
                    ),
                    (
                        "norm4",
                        nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity(),
                    ),
                    ("relu4", self.act()),
                    (
                        "conv5",
                        nn.Conv1d(
                            in_channels=hidden,
                            out_channels=out_channels,
                            kernel_size=self.ks,
                            padding=self.ks // 2,
                            # bias=False,
                        ),
                    ),
                    # ("norm5", nn.BatchNorm1d(num_features=hidden)),
                    # ("relu5", self.act()),
                ]
            )
        )

    def forward(self, x):
        return self.net(x)


# largely copied from https://raw.githubusercontent.com/mateuszbuda/brain-segmentation-pytorch/master/unet.py
class UNet1D(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        hidden=4,
        act=nn.LeakyReLU,
        step=8,
        bn=False,
    ):
        super(UNet1D, self).__init__()
        features = hidden
        self.encoder1 = UNet1D._block(
            in_channels, features, name="enc1", act=act, bn=bn
        )
        self.pool1 = nn.MaxPool1d(kernel_size=step, stride=step)
        self.encoder2 = UNet1D._block(
            features, features * 2, name="enc2", act=act, bn=bn
        )
        self.pool2 = nn.MaxPool1d(kernel_size=step, stride=step)
        self.encoder3 = UNet1D._block(
            features * 2, features * 4, name="enc3", act=act, bn=bn
        )
        self.pool3 = nn.MaxPool1d(kernel_size=step, stride=step)
        self.encoder4 = UNet1D._block(
            features * 4, features * 8, name="enc4", act=act, bn=bn
        )
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1D._block(
            features * 8, features * 16, name="bottleneck", act=act, bn=bn
        )

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet1D._block(
            (features * 8) * 2, features * 8, name="dec4", act=act, bn=bn
        )
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=step, stride=step
        )
        self.decoder3 = UNet1D._block(
            (features * 4) * 2, features * 4, name="dec3", act=act, bn=bn
        )
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=step, stride=step
        )
        self.decoder2 = UNet1D._block(
            (features * 2) * 2, features * 2, name="dec2", act=act, bn=bn
        )
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=step, stride=step
        )
        self.decoder1 = UNet1D._block(
            features * 2, features, name="dec1", act=act, bn=bn
        )

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name, act, bn):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=15,
                            padding=7,
                            bias=True,
                        ),
                    ),
                    (
                        name + "norm1",
                        nn.BatchNorm1d(num_features=features) if bn else nn.Identity(),
                    ),
                    # (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "relu1", act()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=15,
                            padding=7,
                            bias=True,
                        ),
                    ),
                    (
                        name + "norm2",
                        nn.BatchNorm1d(num_features=features) if bn else nn.Identity(),
                    ),
                    # (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "relu2", act()),
                ]
            )
        )


class BeamNetDiscrete(nn.Module):
    def __init__(
        self,
        nthetas,
        hidden,
        symmetry,
        mag_track=2,
        stddev_track=1,
        pd_track=0,
        act=nn.LeakyReLU,
        bn=False,
    ):
        super(BeamNetDiscrete, self).__init__()
        self.nthetas = nthetas
        self.hidden = hidden
        self.pd_track = pd_track
        self.act = act
        self.symmetry = symmetry
        self.mag_track = mag_track
        self.stddev_track = stddev_track
        self.beam_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(3, hidden)),
                    ("relu1", self.act()),
                    (
                        "batchnorm1",
                        nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity(),
                    ),
                    ("linear2", nn.Linear(hidden, hidden)),
                    ("relu2", self.act()),
                    (
                        "batchnorm2",
                        nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity(),
                    ),
                    ("linear3", nn.Linear(hidden, hidden)),
                    ("relu3", self.act()),
                    (
                        "batchnorm3",
                        nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity(),
                    ),
                    ("linear4", nn.Linear(hidden, self.nthetas)),
                    # ("relu4", self.act()),
                    ("softmax", nn.Softmax(dim=1)),  # output a probability distribution
                ]
            )
        )

    def forward(self, x):
        # split into pd>=0 and pd<0
        pd_pos_mask = x[:, self.pd_track] >= 0
        n_pos = pd_pos_mask.sum().item()

        _x = x.new(x.shape)
        # copy over the pd>0
        if self.symmetry:
            _x[:n_pos] = x[pd_pos_mask]
            # flip the magnitudes and phase diffs so all now have phase diff >0
            _x[n_pos:, self.mag_track] = x[~pd_pos_mask, self.mag_track]
            _x[n_pos:, self.stddev_track] = x[~pd_pos_mask, self.stddev_track]
            _x[n_pos:, self.pd_track] = -x[~pd_pos_mask, self.pd_track]
        else:
            _x = x

        _y = self.beam_net(_x)

        y = _y.new(_y.shape)

        if self.symmetry:
            # copy over results for pd>0
            y[pd_pos_mask] = _y[:n_pos]
            # flip distributions over theta for ones that had pd<0
            y[~pd_pos_mask] = _y[n_pos:].flip(1)
        else:
            y = _y
        return y

    def likelihood(self, x, y):
        r = torch.einsum("bk,bk->b", x, self.render_discrete_y(y))[:, None]
        assert r.shape == (x.shape[0], 1)
        return r

    def loglikelihood(self, x, y):
        return torch.log(self.likelihood(x, y))

    # this is discrete its already rendered
    def render_discrete_x(self, x):
        return x

    # this is discrete its already rendered
    def render_discrete_y(self, y):
        return v5_thetas_to_targets(y, self.nthetas)


def normal_dist_d(sigma, d):
    assert sigma.ndim == 1
    d = d / sigma
    return (1 / (sigma * np.sqrt(2 * np.pi))) * torch.exp(-0.5 * d**2)


def normal_dist(x, y, sigma, d=None):
    assert x.ndim == 1
    assert y.ndim == 1
    assert sigma.ndim == 1
    return normal_dist_d(sigma, (x - y))


class BasicBlock(nn.Module):
    def __init__(self, hidden, act, bn=True):
        super(BasicBlock, self).__init__()
        self.linear1 = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity()
        self.act = act()

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.linear1(x)))
        out = self.act(self.bn2(self.linear2(out)))
        return self.act(out + identity)


class PairedBeamNet(nn.Module):
    pass


class BeamNetDirect(nn.Module):
    def __init__(
        self,
        nthetas,
        hidden,
        symmetry,
        depth=3,
        rx_spacing_track=3,
        mag_track=2,
        stddev_track=1,
        pd_track=0,
        other=True,
        bn=False,
        no_sigmoid=False,
        act=nn.LeakyReLU,
        block=False,
        latent=2,
        inputs=3,
    ):
        super(BeamNetDirect, self).__init__()
        self.latent = latent
        self.outputs = 1 + 2 + 2 + latent  # u, o1, o2, k1, k2
        self.hidden = hidden
        self.pd_track = pd_track
        self.act = act
        self.mag_track = mag_track
        self.stddev_track = stddev_track
        self.rx_spacing_track = rx_spacing_track
        self.symmetry = symmetry
        self.block = block
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.nthetas = nthetas
        self.other = other
        self.no_sigmoid = no_sigmoid
        self.inputs = inputs
        net_layout = [
            nn.Linear(self.inputs, hidden),
            self.act(),
            nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity(),
        ]
        for _ in range(depth):
            if self.block:
                net_layout += [BasicBlock(hidden, self.act, bn=bn)]
            else:
                net_layout += [
                    nn.Linear(hidden, hidden),
                    self.act(),
                    nn.BatchNorm1d(num_features=hidden) if bn else nn.Identity(),
                ]
        net_layout += [nn.Linear(hidden, self.outputs)]
        self.beam_net = nn.Sequential(*net_layout)

    def fixify(self, _y, sign):
        _y_sig = self.sigmoid(_y)  # in [0,1]
        if self.no_sigmoid:
            mean_values = sign * _y[:, [0]]
        else:
            _y_sig_centered = (_y_sig[:, [0]] - 0.5) * 4  # in [-2,2]
            mean_values = sign * _y_sig_centered * torch.pi / 2
        return torch.hstack(
            [
                mean_values,  # mu
                _y_sig[:, [1, 2]] * 1.0 + 0.1,  # sigmas
                self.softmax(_y[:, [3, 4]]),
                _y[:, 5:],
            ]
        )

    def forward(self, x):
        # split into pd>=0 and pd<0
        pd_pos_mask = x[:, self.pd_track] >= 0

        if self.symmetry:
            x = x.abs()

        # try to normalize
        x[:, self.pd_track] = x[:, self.pd_track] / (torch.pi / 2)
        x[:, self.mag_track] = x[:, self.mag_track] / 200
        if (x.shape[1] - 1) >= self.rx_spacing_track:
            x[:, self.rx_spacing_track] = x[:, self.rx_spacing_track] / 1000

        y = self.beam_net(x)

        if self.symmetry:
            # copy over results for pd>0
            y[pd_pos_mask] = self.fixify(y[pd_pos_mask], sign=1)
            y[~pd_pos_mask] = self.fixify(y[~pd_pos_mask], sign=-1)
        else:
            y = self.fixify(y, sign=1)
        return y  # [theta_u, sigma1, sigma2, k1, k2]

    def likelihood(self, x, y, sigma_eps=0.00001):
        assert y.ndim == 2 and y.shape[1] == 1
        assert x.ndim == 2 and x.shape[1] >= 5
        ### EXTREMELY IMPORTANT!!! x[:,[0]] NOT x[:,0]
        # mu_likelihood = torch.exp(-((x[:, [0]] - y) ** 2))  #
        mu_likelihood = x[:, 3] * normal_dist(
            x=x[:, 0], y=y[:, 0], sigma=x[:, 1].clamp(min=sigma_eps)
        )
        mu_likelihood += x[:, 3] * normal_dist_d(
            d=(x[:, 0] - torch.pi / 2).abs() + (y[:, 0] - torch.pi / 2).abs(),
            sigma=x[:, 1].clamp(min=sigma_eps),
        )
        mu_likelihood += x[:, 3] * normal_dist_d(
            d=(x[:, 0] + torch.pi / 2).abs() + (y[:, 0] + torch.pi / 2).abs(),
            sigma=x[:, 1].clamp(min=sigma_eps),
        )
        other_likelihood = x[:, 4] * normal_dist(
            x=-x[:, 0].sign() * torch.pi / 2,
            y=y[:, 0],
            sigma=x[:, 2].clamp(min=sigma_eps),
        )
        # mu_likelihood = torch.exp(-((x[:, [0]] - y) ** 2) / (x[:, [1]] + sigma_eps))
        # other_likelihood = 0 * torch.exp(
        #     -((-x[:, 0].sign() * torch.pi / 2 - y) ** 2) / 1
        # )
        l = mu_likelihood
        if self.other:
            l += other_likelihood
        l = l.reshape(-1, 1)
        assert l.shape == (x.shape[0], 1)
        return l

    def loglikelihood(self, x, y, log_eps=0.000000001):
        return torch.log(self.likelihood(x, y) + log_eps)

    # this is discrete its already rendered
    def render_discrete_x(self, x):

        thetas = torch.linspace(
            -torch.pi / 2,
            torch.pi / 2,
            self.nthetas,
            device=x.device,
        ).reshape(1, -1)
        _thetas = thetas.expand(x.shape[0], -1).reshape(-1, 1)
        _x = (
            x.clone()
            .detach()[:, None]
            .expand(-1, self.nthetas, -1)
            .reshape(-1, x.shape[1])
        )  # reshape to eval for all thetas
        likelihoods = self.likelihood(_x, _thetas).reshape(x.shape[0], self.nthetas)
        likelihoods = likelihoods / likelihoods.sum(axis=1, keepdim=True)

        return likelihoods

    # this is discrete its already rendered
    def render_discrete_y(self, y):
        return v5_thetas_to_targets(y, self.nthetas, sigma=0.5)


class BeamNSegNet(nn.Module):
    def __init__(
        self,
        beamnet,
        segnet,
        average_before=True,
        circular_mean=False,
        skip_segmentation=False,
        segmentation_lambda=10.0,
        paired_lambda=1.0,
        independent=True,
        n_radios=1,
        paired_net=None,
        rx_spacing=False,
    ):
        super(BeamNSegNet, self).__init__()
        self.beamnet = beamnet
        self.segnet = segnet
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.average_before = average_before
        self.circular_mean = circular_mean
        self.skip_segmentation = skip_segmentation
        self.segmentation_lambda = segmentation_lambda
        self.independent = independent
        self.n_radios = n_radios
        self.paired_net = paired_net
        self.paired_lambda = paired_lambda
        self.rx_spacing = rx_spacing

    def loss(self, output, y_rad, seg_mask):
        # x to beamformer loss (indirectly including segmentation)
        beamformer_loss = -self.beamnet.loglikelihood(
            output["pred_theta"], y_rad
        ).mean()

        paired_beamformer_loss = 0
        if self.n_radios > 1:
            paired_beamformer_loss = -self.paired_net.loglikelihood(
                output["paired_pred_theta"], y_rad[::2]
            ).mean()

        # segmentation loss
        segmentation_loss = ((output["segmentation"] - seg_mask) ** 2).mean()

        if self.skip_segmentation:
            loss = beamformer_loss
        else:
            loss = (
                beamformer_loss
                + self.segmentation_lambda * segmentation_loss
                + paired_beamformer_loss * self.paired_lambda
            )

        assert loss.isfinite().all()
        results = {
            "beamformer_loss": beamformer_loss,
            "segmentation_loss": segmentation_loss,
            "loss": loss,
        }
        if self.n_radios > 1:
            results["paired_beamformer_loss"] = paired_beamformer_loss
        return results

    def forward(self, x, gt_seg_mask, rx_spacing):
        mask_weights = self.segnet(x)
        pred_seg_mask = self.sigmoid(mask_weights)

        if self.independent:
            seg_mask = gt_seg_mask
        else:
            seg_mask = pred_seg_mask

        assert seg_mask.ndim == 3 and seg_mask.shape[1] == 1

        # taking the average after beam_former
        if not self.average_before:
            batch_size, input_channels, session_size = x.shape
            beam_former_input = x.transpose(1, 2).reshape(
                batch_size * session_size, input_channels
            )
            assert beam_former_input.isfinite().all()
            beamnet_output = self.beamnet(beam_former_input)
            assert (
                beamnet_output.ndim == 2
                and beamnet_output.shape[0] == batch_size * session_size
            )
            assert beamnet_output.isfinite().all()

            beam_former = beamnet_output.reshape(
                batch_size, session_size, beamnet_output.shape[1]
            ).transpose(1, 2)

            pred_theta = torch.mul(beam_former, seg_mask).sum(axis=2) / (
                seg_mask.sum(axis=2) + 0.001
            )
        else:
            weighted_input = torch.mul(x, seg_mask).sum(axis=2) / (
                seg_mask.sum(axis=2) + 0.001
            )
            if self.circular_mean:
                weighted_input[:, self.beamnet.pd_track] = torch_circular_mean(
                    x[:, self.beamnet.pd_track], weights=seg_mask[:, 0], trim=0
                )[0]
            if self.rx_spacing:
                pred_theta = self.beamnet(torch.hstack([weighted_input, rx_spacing]))
            else:
                pred_theta = self.beamnet(weighted_input)
        # p_mask_weights = self.softmax(mask_weights)
        assert pred_theta.isfinite().all()
        assert seg_mask.ndim == 3 and seg_mask.shape[1] == 1
        results = {
            # "pred_theta": torch.mul(beam_former, p_mask_weights).sum(axis=2),
            # "beam_former": beam_former,
            "pred_theta": pred_theta,
            "segmentation": pred_seg_mask,
        }

        if self.n_radios > 1:
            # do a paired prediction
            paired_input = pred_theta.reshape(-1, pred_theta.shape[-1] * self.n_radios)
            results["paired_pred_theta"] = self.paired_net(paired_input)
        return results
