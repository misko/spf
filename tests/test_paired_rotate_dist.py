import torch

from spf.dataset.spf_dataset import v5spfdataset
from spf.rf import rotate_dist


# test if we get a simulated paired signal at different array offsets
# and when we bring the array beamformers back to craft theta
# and multiply they correspond to craft theta ground truth
def test_paired_rotate_dist(perfect_circle_n50_0p01):
    dirname, ds_fn = perfect_circle_n50_0p01

    ds = v5spfdataset(
        ds_fn,
        nthetas=65,
        ignore_qc=True,
        precompute_cache=dirname,
        segment_if_not_exist=True,
        paired=True,
    )

    craft_ground_truth_thetas = torch.vstack(
        [ds[idx][0]["craft_ground_truth_theta"] for idx in range(len(ds))]
    )

    rs = []
    for rx_idx in range(2):
        r = torch.as_tensor(
            ds.precomputed_zarr[f"r{rx_idx}"].windowed_beamformer[:].mean(axis=1)
        )
        r = rotate_dist(
            r,
            torch.as_tensor(ds.z.receivers[f"r{rx_idx}"].rx_theta_in_pis[:]) * torch.pi,
        ).T
        r /= r.sum(axis=0)
        rs.append(r)

    gt = torch.scatter(
        torch.zeros(50, 65),
        1,
        (((craft_ground_truth_thetas - torch.pi) // (2 * torch.pi / 65)) % 65).to(
            torch.long
        ),
        1,
    )

    gt_masked = gt * rs[0].T * rs[1].T
    target_max = (rs[0] * rs[1]).max()

    print("GT_MASKED", gt_masked.sum(axis=1))
    print("MAX", target_max)
    print(gt_masked.sum(axis=1).mean().item(), "vs", target_max.item())
    assert gt_masked.sum(axis=1).mean().isclose(target_max, rtol=1e-1).all()
