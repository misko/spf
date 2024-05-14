from collections import OrderedDict

import torch
import torch.nn as nn

from spf.dataset.spf_dataset import v5_thetas_to_targets


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden):
        super(ConvNet, self).__init__()
        print("BUILD CONV NET!!!")
        self.ks = 17
        self.act = nn.LeakyReLU
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
                    # ("norm1", nn.BatchNorm1d(num_features=hidden)),
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
                    # ("norm2", nn.BatchNorm1d(num_features=hidden)),
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
                    # ("norm3", nn.BatchNorm1d(num_features=hidden)),
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
                    # ("norm4", nn.BatchNorm1d(num_features=hidden)),
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

    def __init__(self, in_channels=3, out_channels=1, init_features=4):
        super(UNet1D, self).__init__()

        features = init_features
        self.encoder1 = UNet1D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.encoder2 = UNet1D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.encoder3 = UNet1D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.encoder4 = UNet1D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet1D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=8, stride=8
        )
        self.decoder3 = UNet1D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=8, stride=8
        )
        self.decoder2 = UNet1D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=8, stride=8
        )
        self.decoder1 = UNet1D._block(features * 2, features, name="dec1")

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
    def _block(in_channels, features, name):
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
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    # (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "relu1", nn.LeakyReLU()),
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
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    # (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "relu2", nn.LeakyReLU()),
                ]
            )
        )


class BeamNetDiscrete(nn.Module):
    def __init__(self, nthetas, hidden, magA_track=0, magB_track=1, pd_track=2):
        super(BeamNetDiscrete, self).__init__()
        self.nthetas = nthetas
        self.hidden = hidden
        self.pd_track = pd_track
        self.act = nn.SELU
        self.magA_track = magA_track
        self.magB_track = magB_track
        self.beam_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(3, hidden)),
                    ("relu1", self.act()),
                    # ("batchnorm1", nn.BatchNorm1d(num_features=hidden)),
                    ("linear2", nn.Linear(hidden, hidden)),
                    ("relu2", self.act()),
                    # ("batchnorm2", nn.BatchNorm1d(num_features=hidden)),
                    ("linear3", nn.Linear(hidden, hidden)),
                    ("relu3", self.act()),
                    # ("batchnorm3", nn.BatchNorm1d(num_features=hidden)),
                    ("linear4", nn.Linear(hidden, self.nthetas)),
                    ("relu4", self.act()),
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
        _x[:n_pos] = x[pd_pos_mask]
        # flip the magnitudes and phase diffs so all now have phase diff >0
        _x[n_pos:, self.magA_track] = x[~pd_pos_mask, self.magB_track]
        _x[n_pos:, self.magB_track] = x[~pd_pos_mask, self.magA_track]
        _x[n_pos:, self.pd_track] = -x[~pd_pos_mask, self.pd_track]

        _y = self.beam_net(_x)

        y = _y.new(_y.shape)
        # copy over results for pd>0
        y[pd_pos_mask] = _y[:n_pos]
        # flip distributions over theta for ones that had pd<0
        y[~pd_pos_mask] = _y[n_pos:].flip(1)

        return y

    def likelihood(self, x, y):
        return torch.einsum("bk,bk->b", x, self.render_discrete_y(y))

    def loglikelihood(self, x, y):
        return torch.log(self.likelihood(x, y))

    # this is discrete its already rendered
    def render_discrete_x(self, x):
        return x

    # this is discrete its already rendered
    def render_discrete_y(self, y):
        return v5_thetas_to_targets(y, self.nthetas)


class BeamNetDirect(nn.Module):
    def __init__(
        self, nthetas, hidden, symmetry, depth=3, magA_track=0, magB_track=1, pd_track=2
    ):
        super(BeamNetDirect, self).__init__()
        self.outputs = 1 + 2 + 2  # u, o1, o2, k1, k2
        self.hidden = hidden
        self.pd_track = pd_track
        self.act = nn.LeakyReLU
        self.magA_track = magA_track
        self.magB_track = magB_track
        self.symmetry = symmetry
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.flip_mu = torch.Tensor([[-1, 1, 1, 1, 1]])
        self.nthetas = nthetas
        net_layout = [
            nn.Linear(3, hidden),
            self.act(),
            nn.BatchNorm1d(num_features=hidden),
        ]
        for layer in range(depth):
            net_layout += [
                nn.Linear(hidden, hidden),
                self.act(),
                nn.BatchNorm1d(num_features=hidden),
            ]
        net_layout += [nn.Linear(hidden, self.outputs)]
        self.beam_net = nn.Sequential(*net_layout)

    def fixify(self, _y, sign):
        _y_sig = self.sigmoid(_y) * 2  # in [0,1]
        return torch.hstack(
            [
                sign * (_y_sig[:, [0]] - 0.5) * 2 * torch.pi / 2,  # in
                _y_sig[:, [1, 2]] * 1 + 0.01,  # sigmas
                self.softmax(_y[:, [3, 4]]),
            ]
        )

    def forward(self, x):
        # split into pd>=0 and pd<0
        pd_pos_mask = x[:, self.pd_track] >= 0
        n_pos = pd_pos_mask.sum().item()

        _x = x.new(x.shape)

        if self.symmetry:
            # copy over the pd>0
            _x[:n_pos] = x[pd_pos_mask]
            # flip the magnitudes and phase diffs so all now have phase diff >0
            _x[n_pos:, self.magA_track] = x[~pd_pos_mask, self.magB_track]
            _x[n_pos:, self.magB_track] = x[~pd_pos_mask, self.magA_track]
            _x[n_pos:, self.pd_track] = -x[~pd_pos_mask, self.pd_track]
        else:
            _x = x

        _y = self.beam_net(_x)

        y = _y.new(_y.shape)
        if self.symmetry:
            # copy over results for pd>0
            y[pd_pos_mask] = self.fixify(_y[:n_pos], sign=1)
            y[~pd_pos_mask] = self.fixify(_y[n_pos:], sign=-1)
        else:
            y = self.fixify(_y, sign=1)
        return y  # [theta_u, sigma1, sigma2, k1, k2]

    def likelihood(self, x, y):
        ### EXTREMELY IMPORTANT!!! x[:,[0]] NOT x[:,0]
        # mu_likelihood = torch.exp(-((x[:, [0]] - y) ** 2))  #
        mu_likelihood = x[:, [3]] * torch.exp(-((x[:, [0]] - y) ** 2) / x[:, [1]])
        other_likelihood = x[:, [4]] * torch.exp(
            -((-x[:, [0]].sign() * torch.pi / 2 - y) ** 2) / x[:, [2]]
        )
        # mu_likelihood = torch.exp(-((x[:, 0] - y) ** 2) / 1)
        # other_likelihood = 0 * torch.exp(
        #     -((-x[:, 0].sign() * torch.pi / 2 - y) ** 2) / 1
        # )
        return mu_likelihood + other_likelihood

    def loglikelihood(self, x, y):
        return torch.log(self.likelihood(x, y))

    # this is discrete its already rendered
    def render_discrete_x(self, x):
        mu_discrete = x[:, [3]] * v5_thetas_to_targets(
            x[:, 0], self.nthetas, sigma=x[:, [1]]
        )
        other_discrete = x[:, [4]] * v5_thetas_to_targets(
            -x[:, [0]].sign() * torch.pi / 2, self.nthetas, sigma=x[:, [2]]
        )
        return mu_discrete + other_discrete

    # this is discrete its already rendered
    def render_discrete_y(self, y):
        return v5_thetas_to_targets(y, self.nthetas)


class BeamNSegNet(nn.Module):
    def __init__(
        self,
        beamnet,
        segnet,
    ):
        super(BeamNSegNet, self).__init__()
        self.beamnet = beamnet
        self.segnet = segnet
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        mask_weights = self.segnet(x)
        assert mask_weights.ndim == 3 and mask_weights.shape[1] == 1

        batch_size, input_channels, session_size = x.shape
        beam_former_input = x.transpose(1, 2).reshape(
            batch_size * session_size, input_channels
        )

        beamnet_output = self.beamnet(beam_former_input)
        assert (
            beamnet_output.ndim == 2
            and beamnet_output.shape[0] == batch_size * session_size
        )

        beam_former = beamnet_output.reshape(
            batch_size, session_size, beamnet_output.shape[1]
        ).transpose(1, 2)
        p_mask_weights = self.softmax(mask_weights)

        return {
            "pred_theta": torch.mul(beam_former, p_mask_weights).sum(axis=2),
            "segmentation": p_mask_weights,
        }
