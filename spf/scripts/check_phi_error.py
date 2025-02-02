import argparse
import os
import sys

from spf.rf import torch_pi_norm

if __name__ == "__main__":

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--segmentation-version",
            type=float,
            required=True,
        )

        parser.add_argument("--precompute-cache", type=str, required=True)
        parser.add_argument(
            "-d",
            "--dataset",
            type=str,
            help="dataset prefix",
            required=True,
        )

        return parser

    parser = get_parser()
    args = parser.parse_args()

    from spf.dataset.spf_dataset import v5spfdataset

    if True:
        ds = v5spfdataset(
            args.dataset,
            nthetas=65,
            ignore_qc=True,
            precompute_cache=args.precompute_cache,
            gpu=True,
            snapshots_per_session=1,
            n_parallel=8,
            paired=True,
            segmentation_version=3.5,
        )
    # except Exception as e:
    #    print(f"Failed to load {os.path.basename(args.dataset)} {str(e)}")
    #   sys.exit(1)

    for rx_idx in range(2):
        err = torch_pi_norm(ds.mean_phase[f"r{rx_idx}"] - ds.ground_truth_phis[rx_idx])
        rx_theta_in_pis = ds.z.receivers[f"r{rx_idx}"].rx_theta_in_pis[:].mean().item()
        print(
            f"{os.path.basename(args.dataset)} r{rx_idx} {err.nanmean()} / {err[err.isfinite()].var()} theta {rx_theta_in_pis} freq {ds.carrier_frequencies[0]} {ds.carrier_frequencies[0]} {ds.yaml_config["receivers"][rx_idx]["antenna-spacing-m"]}"
        )
