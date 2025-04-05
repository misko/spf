import os

import spf
from spf.dataset.spf_dataset import v5spfdataset
from spf.filters.particle_dualradio_filter import plot_single_theta_dual_radio
from spf.filters.run_filters_on_data import run_PF_single_theta_dual_radio

# this contains the empirical distribution mapping observed phase difference
# to the data distribution of theta (ground truth angle between receiver and transmitter)
empirical_data_fn = os.path.join(
    os.path.dirname(spf.__file__), "../empirical_dists/full.pkl"
)

# When temp_file=True: Used for accessing **in-progress recordings** where data
# is actively being written. Files are expected to have a temporary extension (default `.tmp`).
# When temp_file=False: Used for accessing **completed recordings** where all data
# is finalized and no new data will be added.

temp_file = True
ds_fn = "./test_circle"

ds = v5spfdataset(
    ds_fn,
    precompute_cache="./",
    nthetas=65,
    skip_fields=set(["signal_matrix"]),
    empirical_data_fn=empirical_data_fn,
    paired=True,
    ignore_qc=True,
    gpu=False,  # use GPU in segmentation (pre-processing) # turn off if your GPU cant go VROOM
    temp_file=temp_file,
    temp_file_suffix="",
    n_parallel=0,
    skip_detrend=True,
)
args = {
    "ds": ds,
    "N": 1024 * 4,
    "theta_err": 0.01,
    "theta_dot_err": 0.01,
}
result = run_PF_single_theta_dual_radio(**args)
# plot_single_theta_dual_radio(ds)
