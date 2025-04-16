import yaml

from spf.utils import SEGMENTATION_VERSION


def load_config_from_fn(fn):
    with open(fn, "r") as f:
        return load_defaults(yaml.safe_load(f))
    return None


def get_key_or_set_default(d, key, default):
    if "/" in key:
        this_part = key.split("/")[0]
        if this_part not in d:
            d[this_part] = {}
        return get_key_or_set_default(
            d[key.split("/")[0]], "/".join(key.split("/")[1:]), default
        )
    if key not in d:
        d[key] = default


def load_defaults(config):
    get_key_or_set_default(config, "global/signal_matrix_input", False)
    get_key_or_set_default(config, "global/gains_input", False)
    get_key_or_set_default(config, "global/vehicle_type_input", False)
    get_key_or_set_default(config, "optim/output", None)
    get_key_or_set_default(config, "datasets/flip", False)
    get_key_or_set_default(config, "logger", {})
    get_key_or_set_default(config, "datasets/random_snapshot_size", False)
    get_key_or_set_default(config, "optim/save_on", "")
    get_key_or_set_default(config, "optim/scheduler", "step")
    get_key_or_set_default(config, "global/signal_matrix_input", False)
    get_key_or_set_default(config, "global/empirical_input", False)
    get_key_or_set_default(config, "global/windowed_beamformer_input", False)
    get_key_or_set_default(config, "global/sdr_device_type_input", False)
    get_key_or_set_default(config, "datasets/train_on_val", False)
    get_key_or_set_default(config, "datasets/empirical_data_fn", None)
    get_key_or_set_default(config, "datasets/empirical_symmetry", None)
    get_key_or_set_default(
        config, "datasets/segmentation_version", SEGMENTATION_VERSION
    )
    get_key_or_set_default(config, "datasets/random_adjacent_stride", False)
    get_key_or_set_default(config, "datasets/val_subsample_fraction", 1.0)
    get_key_or_set_default(config, "optim/scheduler_step", 1)
    get_key_or_set_default(config, "model/load_single", False)
    get_key_or_set_default(config, "model/load_paired", False)

    get_key_or_set_default(
        config,
        "datasets/val_snapshots_adjacent_stride",
        config["datasets"]["snapshots_adjacent_stride"],
    )
    get_key_or_set_default(
        config,
        "model/output_ntheta",
        config["global"]["nthetas"],
    )

    # config['global'][("signal_matrix_input", False)
    return config


def global_config_to_keys_used(global_config):
    keys_to_get = [
        "all_windows_stats",
        # "rx_pos_xy",
        # "tx_pos_xy",
        # "downsampled_segmentation_mask",
        "rx_lo",
        "rx_wavelength_spacing",
        "y_rad",
        "y_phi",
        "craft_y_rad",
        "y_phi",
        "system_timestamp",
        "y_rad_binned",
        "craft_y_rad_binned",
        "weighted_windows_stats",
        "rx_pos_xy",
        "tx_pos_xy",
        "rx_theta_in_pis",
        "rx_heading_in_pis",
    ]
    if global_config is not None:
        if global_config["signal_matrix_input"]:
            keys_to_get += ["abs_signal_and_phase_diff"]
        if global_config["empirical_input"]:
            keys_to_get += ["empirical"]
        if global_config["windowed_beamformer_input"]:
            keys_to_get += ["windowed_beamformer"]
        if global_config["beamformer_input"]:
            # keys_to_get += ["windowed_beamformer"]
            keys_to_get += ["weighted_beamformer"]
        if global_config["gains_input"]:
            keys_to_get += ["gains"]
        if global_config["vehicle_type_input"]:
            keys_to_get += ["vehicle_type"]
        if global_config["sdr_device_type_input"]:
            keys_to_get += ["sdr_device_type"]
    return keys_to_get
