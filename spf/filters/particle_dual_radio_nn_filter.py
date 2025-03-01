import torch

from spf.dataset.spf_dataset import v5_collate_keys_fast
from spf.filters.filters import (
    ParticleFilter,
    add_noise,
    dual_radio_mse_theta_metrics,
    theta_phi_to_bins,
)
from spf.model_training_and_inference.models.single_point_networks_inference import (
    convert_datasets_config_to_inference,
    get_inference_on_ds,
    load_model_and_config_from_config_fn_and_checkpoint,
)
from spf.rf import torch_pi_norm_pi
from spf.scripts.train_single_point import (
    global_config_to_keys_used,
    load_config_from_fn,
)


class PFSingleThetaDualRadioNN(ParticleFilter):
    def __init__(
        self, ds, checkpoint_fn, config_fn, inference_cache=None, device="cpu"
    ):
        self.ds = ds

        self.generator = torch.Generator()
        self.generator.manual_seed(0)

        checkpoint_config = load_config_from_fn(config_fn)
        # assert (
        #     self.ds.empirical_data_fn
        #     == checkpoint_config["datasets"]["empirical_data_fn"]
        # ), f"Empirical pkl path from ds does not match that from checkpoint {self.ds.empirical_data_fn} vs {checkpoint_config['datasets']['empirical_data_fn']}"

        if not self.ds.temp_file:
            # cache model results
            self.cached_model_inference = torch.as_tensor(
                get_inference_on_ds(
                    ds_fn=ds.zarr_fn,
                    config_fn=config_fn,
                    checkpoint_fn=checkpoint_fn,
                    device=device,
                    inference_cache=inference_cache,
                    batch_size=64,
                    workers=0,
                    precompute_cache=ds.precompute_cache,
                    segmentation_version=ds.segmentation_version,
                )["paired"]
            )
        else:
            # load the model and such
            self.model, self.model_config = (
                load_model_and_config_from_config_fn_and_checkpoint(
                    config_fn=config_fn, checkpoint_fn=checkpoint_fn, device=device
                )
            )
            self.model.eval()

            self.model_datasets_config = convert_datasets_config_to_inference(
                self.model_config["datasets"],
                ds_fn=ds.zarr_fn,
                precompute_cache=self.ds.precompute_cache,
            )

            self.model_optim_config = {"device": device, "dtype": torch.float32}

            self.model_keys_to_get = global_config_to_keys_used(
                global_config=self.model_config["global"]
            )

    def model_inference_at_observation_idx(self, idx):
        if not self.ds.temp_file:
            return self.cached_model_inference[idx]

        z = v5_collate_keys_fast(self.model_keys_to_get, [self.ds[idx]]).to(
            self.model_optim_config["device"]
        )
        with torch.no_grad():
            return self.model(z)["paired"].cpu()

    def observation(self, idx):
        # even though the model outputs one paired dist for each reciever
        # they should be identical
        return self.model_inference_at_observation_idx(idx)[0, 0]

    def fix_particles(self):
        self.particles[:, 0] = torch_pi_norm_pi(self.particles[:, 0])

    def predict(self, our_state, dt, noise_std):
        if noise_std is None:
            noise_std = torch.tensor([[0.1, 0.001]])
        self.particles[:, 0] += dt * self.particles[:, 1]
        add_noise(self.particles, noise_std=noise_std, generator=self.generator)

    def update(self, z):
        #
        # z is not the raw observation, but the processed model output
        theta_bin = theta_phi_to_bins(self.particles[:, 0], nbins=z.shape[0])
        prob_theta_given_observation = torch.take(z, theta_bin)

        self.weights *= prob_theta_given_observation
        self.weights += 1.0e-30  # avoid round-off to zero
        self.weights /= torch.sum(self.weights)  # normalize

    def metrics(self, trajectory):
        return dual_radio_mse_theta_metrics(
            trajectory, self.ds.craft_ground_truth_thetas
        )

    def trajectory(self, **kwargs):
        trajectory = super().trajectory(**kwargs)
        for x in trajectory:
            x["craft_theta"] = x["mu"][0]
            x["P_theta"] = x["var"][0]
        return trajectory
