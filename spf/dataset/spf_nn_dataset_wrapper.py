from functools import lru_cache

import torch
from torch.utils.data import Dataset

from spf.dataset.spf_dataset import v5_collate_keys_fast
from spf.filters.particle_dual_radio_nn_filter import (
    cached_model_inference_to_absolute_north,
)
from spf.model_training_and_inference.models.single_point_networks_inference import (
    get_nn_inference_on_ds_and_cache,
    load_model_and_config_from_config_fn_and_checkpoint,
)
from spf.rf import rotate_dist, torch_pi_norm_pi
from spf.scripts.train_utils import global_config_to_keys_used


# attach nn inference attributes to dataset entries
class v5spfdataset_nn_wrapper(Dataset):
    def __init__(
        self,
        ds,
        checkpoint_config_fn,
        checkpoint_fn,
        inference_cache,
        device="cpu",
        v4=None,
        absolute=False,
    ):
        self.ds = ds
        assert self.ds.paired
        self.checkpoint_config_fn = checkpoint_config_fn
        self.checkpoint_fn = checkpoint_fn
        self.inference_cache = inference_cache
        self.absolute = absolute

        if v4 is None:
            v4 = self.ds.v4

        if not ds.realtime:
            self.cached_model_inference = {
                k: torch.as_tensor(v)
                for k, v in get_nn_inference_on_ds_and_cache(
                    ds_fn=ds.zarr_fn,
                    config_fn=self.checkpoint_config_fn,
                    checkpoint_fn=self.checkpoint_fn,
                    device=device,
                    inference_cache=inference_cache,
                    batch_size=64,
                    workers=0,
                    precompute_cache=ds.precompute_cache,
                    crash_if_not_cached=False,
                    segmentation_version=ds.segmentation_version,
                    v4=v4,
                ).items()
            }
            if self.absolute:
                self.cached_model_inference = {
                    k: cached_model_inference_to_absolute_north(ds, v)
                    for k, v in self.cached_model_inference.items()
                }
        else:
            self.model, self.model_config = (
                load_model_and_config_from_config_fn_and_checkpoint(
                    self.checkpoint_config_fn, self.checkpoint_fn, device=device
                )
            )
            self.model.eval()
            self.keys_to_get = global_config_to_keys_used(
                global_config=self.model_config["global"]
            )

    def to_absolute_north(self, sample):
        for ridx in range(2):
            ntheta = sample[ridx]["paired"].shape[-1]
            paired_nn_inference = sample[ridx]["paired"].reshape(-1, ntheta)
            paired_nn_inference_rotated = rotate_dist(
                paired_nn_inference,
                rotations=torch_pi_norm_pi(
                    sample[ridx]["rx_heading_in_pis"][:, None] * torch.pi
                ),
            ).reshape(paired_nn_inference.shape)
            sample[ridx]["paired"] = paired_nn_inference_rotated
        return sample

    @lru_cache
    def get_inference_for_idx(self, idx):
        if not self.ds.realtime:
            return [
                {k: v[idx][ridx] for k, v in self.cached_model_inference.items()}
                for ridx in range(2)
            ]
        return self.get_and_annotate_entry_at_idx(idx)

    @lru_cache
    def get_and_annotate_entry_at_idx(self, idx):
        sample = self.ds[idx]
        if not self.ds.realtime:
            for ridx in range(2):
                sample[ridx].update(
                    {k: v[idx][ridx] for k, v in self.cached_model_inference.items()}
                )
            return sample
        else:
            single_example = v5_collate_keys_fast(self.keys_to_get, [sample]).to(
                self.model_config["optim"]["device"]
            )
            with torch.no_grad():
                nn_inference = self.model(single_example)
            for ridx in range(2):
                sample[ridx].update({k: v[ridx] for k, v in nn_inference.items()})
            if self.absolute:
                sample = self.to_absolute_north(sample)
            return sample

    def __iter__(self):
        self.serving_idx = 0
        return self

    def __next__(self):
        sample = self.get_and_annotate_entry_at_idx(self.serving_idx)
        self.serving_idx += 1
        return sample

    @lru_cache
    def __getitem__(self, idx):
        return self.get_and_annotate_entry_at_idx(idx)

    def __len__(self):
        return len(self.ds)

    @property
    def mean_phase(self):
        return self.ds.mean_phase

    @property
    def ground_truth_phis(self):
        return self.ds.ground_truth_phis

    @property
    def craft_ground_truth_thetas(self):
        return self.ds.craft_ground_truth_thetas

    @property
    def absolute_thetas(self):
        return self.ds.absolute_thetas
