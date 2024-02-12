import argparse
import glob
import os

import numpy as np

from spf.dataset.spf_dataset import SessionsDatasetRealV2
from spf.grbl.grbl_interactive import get_default_dynamics, get_default_gm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--directory", help="directory that has data files", required=True
    )
    parser.add_argument("-s", "--serial", help="serial port")
    args = parser.parse_args()

    list_of_files = glob.glob(f"{args.directory}/*.npy")
    latest_file = max(list_of_files, key=os.path.getctime)
    print("Restoring from...", latest_file)

    ds = SessionsDatasetRealV2(
        root_dir=args.directory,
        snapshots_in_session=-1,
        check_files=False,
        step_size=1,
        filenames=[latest_file],
    )

    bfo = ds[0]["beam_former_outputs_at_t"].copy()

    if (bfo == 0).min(axis=1).max():
        first_invalid_idx = np.argmax((bfo == 0).min(axis=1))
        assert first_invalid_idx != 0
        last_valid_idx = first_invalid_idx - 1

        # get receivers
        rx0 = ds[0]["receiver_positions_at_t"][last_valid_idx].mean(axis=0)
        rx1 = ds[1]["receiver_positions_at_t"][last_valid_idx].mean(axis=0)
        assert np.isclose(rx0, rx1).all()

        # get the emitter
        tx0 = ds[0]["source_positions_at_t"][-1, 0]
        tx1 = ds[1]["source_positions_at_t"][-1, 0]
        assert np.isclose(tx0, tx1, rtol=0.1, atol=0.1).all()

        print(f"Last valid idx: {last_valid_idx}, rx-pos:{rx0} tx-pos:{tx0}")

        dynamics = get_default_dynamics()

        gm = get_default_gm(args.serial)
        gm.set_current_position(
            motor_channel=ds.get_yaml_config()["emitter"]["motor_channel"],
            steps=dynamics.to_steps(tx0),
        )

        gm.set_current_position(
            motor_channel=ds.get_yaml_config()["receivers"][0]["motor_channel"],
            steps=dynamics.to_steps(dynamics.to_steps(rx0)),
        )
