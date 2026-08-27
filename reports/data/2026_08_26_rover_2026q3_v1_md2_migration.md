# Rover 2026 Q3 data migration to MD2

**Date:** 2026-08-26

**Status:** Complete and verified

**Release name:** `rover_2026q3_v1`

## Summary

The 48 merged rover datasets collected from 2026-07-31 through 2026-08-07 and their segmentation 3.7 artifacts were copied from QNAP to MD2. The copies were staged, verified against QNAP, and then promoted into versioned final directories.

The original QNAP datasets remain unchanged. Raw captures were intentionally not copied to MD2.

## Final layout

| Purpose | Final path | Contents |
| --- | --- | --- |
| Merged rover datasets | `/mnt/md2/rovers/2026q3_v1/merged` | 48 Zarr stores and 48 YAML sidecars |
| Integrated training cache | `/mnt/md2/cache/precompute_cache_3p7` | 2,454 PKL/YARR pairs: 2,406 historical plus 48 rover |
| Verified rover fallback | `/mnt/md2/cache/precompute_cache_3p7_rover_2026q3_v1` | Isolated copy of the 48 rover pairs retained temporarily |

The rover artifacts use the same segmentation 3.7 and 65-angle contract as the historical cache. A fresh basename audit found zero collisions, so the 48 rover pairs were copied directly into the historical cache with overwrite protection. A checksum comparison then verified that every integrated rover artifact is identical to the isolated verified copy.

## Source data retained on QNAP

| Purpose | Source path | Migration policy |
| --- | --- | --- |
| Raw captures | `/mnt/qnap01/mouse9911/rovers_2026/raw` | Retained on QNAP; not copied to MD2 |
| Merged datasets | `/mnt/qnap01/mouse9911/rovers_2026/merged` | Retained after the verified MD2 copy |
| Segmentation artifacts | `/mnt/qnap01/mouse9911/rovers_2026/precompute` | Retained after the verified MD2 copy |

No QNAP file was deleted, renamed, or modified during this migration.

## Storage impact

The promoted MD2 directories occupy:

- Merged datasets: approximately 389 GiB allocated.
- Integrated rover segmentation contribution: approximately 7.6 GiB allocated.
- Isolated rover fallback: approximately 7.6 GiB allocated until a separate retention decision is made.

After migration, `/mnt/md2` reported approximately 782 GiB available and 99% utilization. Further large copies on MD2 should therefore be avoided unless capacity is reviewed first.

## Procedure and validation

1. A preflight verified source counts, target-path availability, MD2 inode availability, and a projected reserve of at least 500 GiB after copying.
2. Segmentation artifacts were copied into an isolated MD2 staging directory using `rsync -aH --sparse --partial`.
3. A checksum-mode dry run verified the staged segmentation tree against QNAP.
4. Merged Zarr/YAML data were copied into an isolated MD2 staging directory using resumable `rsync` with `--partial --append-verify`.
5. The merged source and staging trees were compared using:

   ```text
   rsync -aH --sparse --dry-run --checksum --itemize-changes SOURCE/ DESTINATION/
   ```

   The verification finished with exit status 0, a zero-byte difference log, and a zero-byte error log.
6. Verified staging directories were promoted by same-filesystem atomic rename.
7. A temporary combined historical-plus-rover symlink view was assembled and validated while the physical caches remained isolated.
8. A second collision audit found zero overlapping artifact names. The 48 rover pairs were then copied directly into `/mnt/md2/cache/precompute_cache_3p7` with `--ignore-existing`, so no historical artifact could be overwritten.
9. Checksum-mode `rsync` verified that the integrated rover artifacts are identical to the isolated verified cache.
10. The now-redundant combined view was audited to contain only 4,908 valid symlinks, then those symlinks and the empty view directory were removed. No target data were deleted.
11. Final loader-style basename resolution confirmed that every one of the 48 merged rover datasets resolves both its PKL and YARR artifact through the standard historical cache.

Final validation passed these invariants:

- 48 merged `.zarr` directories.
- 48 matching `.yaml` sidecars.
- 2,454 integrated segmentation `.pkl` files.
- 2,454 integrated segmentation `.yarr` stores.
- 48 isolated fallback PKL/YARR pairs retained outside the training cache.
- Zero integrated-cache basename collisions or overwrites.
- Zero checksum differences between the QNAP and MD2 merged trees.
- Zero checksum differences between the isolated and integrated rover segmentation artifacts.

## Training use

Training configurations that need both the historical segmentation corpus and the 2026 rover segmentation should use:

```text
/mnt/md2/cache/precompute_cache_3p7
```

New training and evaluation manifests should reference the promoted merged datasets under:

```text
/mnt/md2/rovers/2026q3_v1/merged
```

The immutable historical train and validation manifests should remain unchanged. Any augmented manifest should be a new TXT file formed from the historical TXT plus the selected rover paths.

## Remaining administrative items

- QNAP remains the retained source/archive copy; no retention or deletion decision has been made.
- The isolated 48-pair rover cache is retained as a verified fallback; removing it would reclaim approximately 7.6 GiB and requires a separate retention decision.
- Transfer and checksum status logs remain under `/mnt/md2/rovers/.2026q3_v1.staging/.copy_state` and have not been removed.
- Training manifests and model configurations were not changed as part of this migration.
- Before launching training, verify that the selected manifest and configuration point to the final MD2 paths above rather than an earlier local staging path.
