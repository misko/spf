import pickle
from spf.dataset.spf_dataset import v5inferencedataset, data_single_radio_to_raw
from spf.dataset.fake_dataset import create_fake_dataset, fake_yaml_v4
from spf.dataset.spf_dataset import v5spfdataset

n = 50
tmpdirname = "/tmp/test1"
fn = tmpdirname + f"/perfect_circle_n{n}_noise0p0"
v5inf = v5inferencedataset(
    yaml_fn=fn + ".yaml",
    nthetas=65,
    gpu=False,
    n_parallel=8,
    paired=True,
    model_config_fn="",
    vehicle_type="wallarray",
    skip_segmentation=True,
    skip_detrend=True
)
precomputed_ds = pickle.load(open("/tmp/test1/ds.pkl", "rb"))
for idx in range(20):
    d = precomputed_ds[idx]
    for ridx in range(2):
        print(idx, ridx)
        v5inf.write_to_idx(idx, ridx, data_single_radio_to_raw(d[ridx]))
