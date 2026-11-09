# Rover 3.1 seven-antenna mounts

These are production-resolution binary STL exports of the parametric
seven-antenna mount. Dimensions are millimetres and each mesh is already in its
print orientation with the base at z = 0.

`PCD` means the pitch-circle diameter through the six outer antenna axes. The
OpenSCAD `antenna_spacing` parameter is the radius, not this diameter.

| STL | PCD | Radius / adjacent outer-axis spacing | Outer rotation | Centre cable exit | Mount bounds |
| --- | ---: | ---: | ---: | ---: | --- |
| [`antenna_mount_hex7_pcd47p5_rot0.stl`](./antenna_mount_hex7_pcd47p5_rot0.stl) | 47.5 | 23.75 | 0 deg | -30 deg | 107 x 107 x 31.325 |
| [`antenna_mount_hex7_pcd47p5_rot30.stl`](./antenna_mount_hex7_pcd47p5_rot30.stl) | 47.5 | 23.75 | 30 deg | -60 deg | 107 x 107 x 31.325 |
| [`antenna_mount_hex7_pcd51_rot0.stl`](./antenna_mount_hex7_pcd51_rot0.stl) | 51 | 25.5 | 0 deg | -30 deg | 110.5 x 107 x 31.325 |
| [`antenna_mount_hex7_pcd51_rot30.stl`](./antenna_mount_hex7_pcd51_rot30.stl) | 51 | 25.5 | 30 deg | -60 deg | 107 x 110.5 x 31.325 |

The [`flat E-Z LOK base`](./array_mount_flat_base_ezlok_m3.stl) is a
112 x 112 x 4 mm plate for using any of these mounts away from a Rover. Its
four 4.40 mm straight-through insert bores reproduce the supplied Rover 3.1
top-front panel's known-working `260-M3-BR` fit and mounting axes.

All four variants retain the recovered reference antenna bores, tapers, flex
slots, underside raceways, and exact mounting-hole axes. Only the radius of the
outer six and their 0/30-degree rotation change. The centre cable exit is routed
through the nearest outer-antenna gap to avoid the collision present with a
fixed -45-degree route.

With a conservative 11 mm L-shaped antenna keepout, minimum antenna clearance
is 0.875 mm for the 47.5 mm PCD and 1.75 mm for the 51 mm PCD. The screenshot's
95 mm height and 30 mm elbow reach are known, but its diameters were estimated.
Measure the real lower pivot and distal connector before a production run; the
estimated 10.5 mm connector is wider than the recovered 9.75 mm raceway.

## Source and regeneration

- [Parametric OpenSCAD source](../../../../3D_printing_design_files/antenna_mount.scad)
- [Dimensions and mesh-validation report](../../../../3D_printing_design_files/antenna_mount.md)
- [Binary-STL export helper](../../../../3D_printing_design_files/export_antenna_mount.sh)
- [Flat-base source and dimensional notes](../../../../3D_printing_design_files/array_mount_flat_base.md)

From `data_collection/3D_printing_design_files/`:

```bash
./export_antenna_mount.sh --variant hex_7 --spacing 23.75 --rotation 0
./export_antenna_mount.sh --variant hex_7 --spacing 23.75 --rotation 30
./export_antenna_mount.sh --variant hex_7 --spacing 25.5 --rotation 0
./export_antenna_mount.sh --variant hex_7 --spacing 25.5 --rotation 30
```

The repository stores these binary meshes with Git LFS. Run `git lfs pull`
after cloning if the working tree contains pointer files instead of STL data.

## Validation

All four published meshes were checked as single watertight components with
consistent winding. Cross-sections preserve the fixed mounting axes and place
all seven antenna axes within 0.000001 mm of their declared coordinates.

| STL | Faces | Volume (mm^3) | SHA-256 |
| --- | ---: | ---: | --- |
| `antenna_mount_hex7_pcd47p5_rot0.stl` | 280,098 | 86,351.047788 | `270e27a6202de8c0a582cc55104eec26c6ccc047f453fe20bbf88c65723841dc` |
| `antenna_mount_hex7_pcd47p5_rot30.stl` | 280,098 | 86,351.047658 | `3ab3f7ab072ccbbe3242cc6f36075192580e67637219e2b8c1d4002d72715d63` |
| `antenna_mount_hex7_pcd51_rot0.stl` | 288,018 | 89,468.535091 | `76d8bd43f8e85dff5625437a20d1be3b2809f14ab58d71fff7651838d5f0569a` |
| `antenna_mount_hex7_pcd51_rot30.stl` | 288,018 | 89,468.535341 | `634cac3d926d68a16f399a551e4f3430c89e08d1523adb8930d55ef5ecccbfbb` |

The flat base is also a single watertight component with consistent winding:

| STL | Faces | Bounds (mm) | Volume (mm^3) | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `array_mount_flat_base_ezlok_m3.stl` | 2,908 | 112 x 112 x 4 | 49,901.853744 | `a02983fb31b6493b1fabe536a1a9d5f249bd471cf44194745af4e1bbc3243f44` |
