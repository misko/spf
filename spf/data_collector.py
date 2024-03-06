import logging
import sys
import time

import numpy as np

from spf.sdrpluto.sdr_controller import (
    EmitterConfig,
    ReceiverConfig,
    get_pplus,
    setup_rx,
    setup_rxtx,
)
from spf.wall_array_v2 import v2_column_names


class DataCollector:
    def __init__(self, yaml_config, filename_npy, tag=""):
        self.yaml_config = yaml_config
        self.filename_npy = filename_npy
        self.record_matrix = None

    def radios_to_online(self):
        # record matrix
        column_names = v2_column_names(nthetas=self.yaml_config["n-thetas"])
        if not self.yaml_config["dry-run"]:
            self.record_matrix = np.memmap(
                self.filename_npy,
                dtype="float32",
                mode="w+",
                shape=(
                    2,  # TODO should be nreceivers
                    self.yaml_config["n-records-per-receiver"],
                    len(column_names),
                ),  # t,tx,ty,rx,ry,rtheta,rspacing /  avg1,avg2 /  sds
            )

        # lets open all the radios
        radio_uris = []
        if self.yaml_config["emitter"]["type"] == "sdr":
            radio_uris.append(self.yaml_config["emitter"]["receiver-uri"])
        for receiver in self.yaml_config["receivers"]:
            radio_uris.append(receiver["receiver-uri"])
        for radio_uri in radio_uris:
            get_pplus(uri=radio_uri)

        time.sleep(0.1)

        target_yaml_config = self.yaml_config["emitter"]
        if target_yaml_config["type"] == "sdr":  # this  wont be the case for mavlink
            # setup the emitter
            target_rx_config = ReceiverConfig(
                lo=target_yaml_config["f-carrier"],
                rf_bandwidth=target_yaml_config["bandwidth"],
                sample_rate=target_yaml_config["f-sampling"],
                gains=[target_yaml_config["rx-gain"], target_yaml_config["rx-gain"]],
                gain_control_modes=[
                    target_yaml_config["rx-gain-mode"],
                    target_yaml_config["rx-gain-mode"],
                ],
                enabled_channels=[0, 1],
                buffer_size=target_yaml_config["buffer-size"],
                intermediate=target_yaml_config["f-intermediate"],
                uri=target_yaml_config["receiver-uri"],
            )
            target_tx_config = EmitterConfig(
                lo=target_yaml_config["f-carrier"],
                rf_bandwidth=target_yaml_config["bandwidth"],
                sample_rate=target_yaml_config["f-sampling"],
                intermediate=target_yaml_config["f-intermediate"],
                gains=[target_yaml_config["tx-gain"], -80],
                enabled_channels=[0],
                cyclic=True,
                uri=target_yaml_config["emitter-uri"],
                motor_channel=target_yaml_config["motor_channel"],
            )

            pplus_rx, _ = setup_rxtx(
                rx_config=target_rx_config, tx_config=target_tx_config, leave_tx_on=True
            )
            pplus_rx.close_rx()

        # get radios online
        receiver_pplus = {}
        for receiver in self.yaml_config["receivers"]:
            rx_config = ReceiverConfig(
                lo=receiver["f-carrier"],
                rf_bandwidth=receiver["bandwidth"],
                sample_rate=receiver["f-sampling"],
                gains=[receiver["rx-gain"], receiver["rx-gain"]],
                gain_control_modes=[
                    receiver["rx-gain-mode"],
                    receiver["rx-gain-mode"],
                ],
                enabled_channels=[0, 1],
                buffer_size=receiver["buffer-size"],
                intermediate=receiver["f-intermediate"],
                uri=receiver["receiver-uri"],
                rx_spacing=receiver["antenna-spacing-m"],
                rx_theta_in_pis=receiver["theta-in-pis"],
                motor_channel=receiver["motor_channel"],
                rx_buffers=receiver["rx-buffers"],
            )
            assert "emitter-uri" not in receiver
            assert (
                "skip_phase_calibration" not in self.yaml_config
                or self.yaml_config["skip_phase_calibration"]
            )
            # there is no emitter to setup, its already blasting
            pplus_rx = setup_rx(rx_config=rx_config)

            if pplus_rx is None:
                logging.info("Failed to bring RXTX online, shuttingdown")
                sys.exit(1)
            else:
                logging.debug("RX online!")
                receiver_pplus[pplus_rx.uri] = pplus_rx
                assert pplus_rx.rx_config.rx_pos is not None

    def start(self):
        pass

    def is_collecting(self):
        pass
