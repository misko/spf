import torch

from spf.filters.filters import (
    ParticleFilter,
    add_noise,
    dual_radio_mse_theta_metrics,
    theta_phi_to_bins,
)
from spf.rf import rotate_dist, torch_pi_norm_pi


def cached_model_inference_to_absolute_north(ds, cached_model_inference):
    _rx_heading = torch.concatenate(
        [
            torch_pi_norm_pi(
                ds.cached_keys[ridx]["rx_heading_in_pis"][:, None] * torch.pi
            )
            for ridx in range(2)
        ],
        dim=1,
    ).reshape(-1, 1)
    _cached_model_inference = cached_model_inference.reshape(-1, 65)
    cached_model_inference_rotated = rotate_dist(
        _cached_model_inference,
        rotations=_rx_heading,
    ).reshape(cached_model_inference.shape)
    return cached_model_inference_rotated


class PFSingleThetaDualRadioNN(ParticleFilter):
    def __init__(
        self,
        nn_ds,
    ):
        self.ds = nn_ds
        self.absolute = nn_ds.absolute
        self.generator = torch.Generator()
        self.generator.manual_seed(0)

    def observation(self, idx):
        return self.ds.get_inference_for_idx(idx)[0]["paired"][0]

    def fix_particles(self):
        self.particles[:, 0] = torch_pi_norm_pi(self.particles[:, 0])

    def predict(self, our_state, dt, noise_std):
        if noise_std is None:
            noise_std = torch.tensor([[0.1, 0.001]])
        self.particles[:, 0] += dt * self.particles[:, 1]
        add_noise(self.particles, noise_std=noise_std, generator=self.generator)

    def update(self, z):
        # z is not the raw observation, but the processed model output
        theta_bin = theta_phi_to_bins(self.particles[:, 0], nbins=z.shape[0])
        prob_theta_given_observation = torch.take(z, theta_bin)

        self.weights *= prob_theta_given_observation
        self.weights += 1.0e-30  # avoid round-off to zero
        self.weights /= torch.sum(self.weights)  # normalize

    def metrics(self, trajectory):
        return dual_radio_mse_theta_metrics(
            trajectory,
            (
                self.ds.craft_ground_truth_thetas
                if not self.absolute
                else self.ds.absolute_thetas.mean(axis=0)
            ),
        )

    def trajectory(self, **kwargs):
        trajectory = super().trajectory(**kwargs)
        for x in trajectory:
            x["craft_theta"] = x["mu"][0]
            x["P_theta"] = x["var"][0]
        return trajectory
