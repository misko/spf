from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


zarr_fn = "/mnt/md2/2d_wallarray_v2_data/feb/wallarrayv3_2025_02_10_16_10_44_nRX2_bounce_spacing0p07.zarr"
z = zarr_open_from_lmdb_store(zarr_fn, readahead=False, map_size=2**32)
print(z.tree())
