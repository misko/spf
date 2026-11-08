# Flat base for the Rover 3.1 antenna-array mount

[`array_mount_flat_base.scad`](array_mount_flat_base.scad) generates a simple
flat plate that accepts the same antenna-array mount used on the Rover 3.1 top
panel. All dimensions are millimetres.

## Default production geometry

- Base: 112 x 112 x 4 mm with 3 mm plan-view corner radii.
- Insert bores: four straight 4.40 mm through-holes.
- Array mounting axes: `(48.75, -12.15)`, `(12.15, -48.75)`,
  `(-48.75, 12.15)`, and `(-12.15, 48.75)`.
- Nearest bore-edge to plate-edge clearance: 5.05 mm.
- Print orientation: the large flat face is on the bed at z = 0.

The four axes were checked against the production seven-antenna STL. The
4.40 mm bore and 4.00 mm deck thickness were measured directly from the
supplied `rover v3.1 - top_front (1).stl`; its matching bores are unchamfered,
straight cylinders from z = 79 to z = 83. This intentionally reproduces the
known-working printed fit rather than changing to a nominal catalog hole.

## How the array mounts to the Rover top

In the supplied top-front STL coordinate system, the array is centred at
`(0, -66.25)` on the upper deck. The top provides three complete four-insert
patterns, allowing the same array mount to be installed at 0, 45, or 90
degrees. For the mount's native 0-degree orientation, its four axes translate
to:

| Mount axis | Matching Rover-top insert axis |
| --- | --- |
| `(48.75, -12.15)` | `(48.75, -78.40)` |
| `(12.15, -48.75)` | `(12.15, -115.00)` |
| `(-48.75, 12.15)` | `(-48.75, -54.10)` |
| `(-12.15, 48.75)` | `(-12.15, -17.50)` |

The flat base keeps the mount-native pattern centred at `(0, 0)`; no rotation
or offset is needed when stacking the array STL on the exported plate.

The specified insert is the brass E-Z LOK E-Z Press flanged insert,
`260-M3-BR`, with an M3-0.5 internal thread. Its manufacturer drawing lists a
5/32 inch (3.962 mm) starting hole and notes that required hole size varies by
material and tooling. The 4.40 mm Rover bore is therefore an application-
specific allowance and should be retained only when using the same printer,
material, and insert process. Print a small fit coupon first if any of those
change.

The insert body may project slightly below this 4 mm through-hole plate, as it
does on the Rover top panel. Increase `base_thickness` if a flush underside is
more important than reproducing the Rover panel thickness.

## Export

From this directory:

```bash
openscad --hardwarnings --export-format binstl \
  -o ../rover/rover_v3.1/3D_printed_parts/antenna_mount/array_mount_flat_base_ezlok_m3.stl \
  array_mount_flat_base.scad
```

The checked-in STL uses the defaults above. Change the parameters at the top
of the OpenSCAD file to resize the plate while preserving the mounting axes
and insert fit.
