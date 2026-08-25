# Parametric antenna mount

[`antenna_mount.scad`](antenna_mount.scad) is an analytic reconstruction of the
supplied five-antenna STL, plus a seven-antenna layout with six sockets on a
regular hexagon and one in the centre. All dimensions are millimetres.

The antenna fit and mounting-hole pattern are fixed dimensions. Changing the
antenna spacing or rotating the outer six cannot move or resize the four mount
holes.

## Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `variant` | `"reference_5"` | Original five-socket layout or the new `"hex_7"` layout. |
| `antenna_spacing` | `23.75` | Radius from the centre socket to each outer socket. For the regular hexagon this is also adjacent outer-socket spacing. |
| `outer_rotation` | `0` | Hex layout rotation. Only `0` or `30` degrees is accepted. |
| `center_raceway_angle` | `-45` | Source centre-exit direction and the preferred direction used when choosing a hex gap. |
| `auto_route_hex_center` | `true` | Moves the hex centre exit to the closest safe gap between outer antennas. |
| `render_quality` | `"production"` | `"draft"` for quick checks or `"production"` for the source mesh's circle resolution. |

The solid-geometry minimum is 21.75 mm. Smaller values overlap the centre
socket's root blend and an outer socket, so OpenSCAD stops with an assertion.
The conservative 11 mm antenna keepouts require at least 22 mm; the 23.75 mm
default clears both checks. Larger values can increase the overall footprint
while the mounting pattern remains fixed.

## Export

The helper writes binary STL files and accepts the spacing and 0/30-degree
rotation as command-line parameters:

```bash
./export_antenna_mount.sh --variant reference_5
./export_antenna_mount.sh --variant hex_7 --rotation 0 --spacing 23.75
./export_antenna_mount.sh --variant hex_7 --rotation 30 --spacing 23.75
```

Use `--quality draft` for a faster preview and `-o path/to/output.stl` to choose
the output path. With no `-o`, files are written to the ignored `generated/`
directory beside the script.

The same values can be overridden directly with OpenSCAD's `-D` arguments:

```bash
openscad --export-format binstl \
  -o antenna_mount_hex_7_rot30.stl \
  -D 'variant="hex_7"' \
  -D 'antenna_spacing=23.75' \
  -D 'outer_rotation=30' \
  antenna_mount.scad
```

The checked-in Rover 3.1 production exports and their checksums are listed in
the [Rover 3.1 antenna-mount manifest](../rover/rover_v3.1/3D_printed_parts/antenna_mount/README.md).

## Antenna previews and clearance

Set `show_antennas=true` to add seven visual antenna mock-ups to a PNG render.
These objects are off by default and the STL export helper never enables them.
The mock-up uses the supplied image's 95 mm vertical height and 30 mm elbow
reach. Diameters inferred from the image remain editable preview parameters:

- 19.5 mm lower joint at 5.75 mm diameter;
- upper whip tapering from 9.7 to 8.3 mm;
- 8.5 mm horizontal barrel with a 10.5 mm distal connector;
- optional 11 mm conservative keepout around the complete L shape.

Only the labeled 95 and 30 mm dimensions are authoritative. Replace the
diameter parameters with caliper measurements when available.

Example oblique render:

```bash
xvfb-run -a openscad --preview \
  --imgsize=900,900 --projection=o --autocenter --viewall \
  --camera=0,0,0,62,0,35,220 \
  -o antenna_mount_hex_7_rot0_with_antennas.png \
  -D 'variant="hex_7"' \
  -D 'outer_rotation=0' \
  -D 'render_quality="draft"' \
  -D 'show_antennas=true' \
  antenna_mount.scad
```

Use `show_antenna_keepouts=true` to display the blue 11 mm envelopes, and
`show_preview_mount=false` for an unobstructed top clearance view.

The antenna check caught and corrected a collision in the first hex draft. A
fixed -45-degree centre elbow passed only 6.147 mm from an outer antenna axis
in both requested rotations. The hex layout now routes the centre elbow through
the nearest gap automatically:

| Outer-six rotation | Centre elbow direction |
| ---: | ---: |
| 0 degrees | -30 degrees |
| 30 degrees | -60 degrees |

At the default 23.75 mm ring radius, the closest centreline spacing is 11.875
mm. The conservative 11 mm L-shaped keepouts therefore have 0.875 mm minimum
clearance; adjacent outer keepouts have 12.75 mm clearance. For a changed ring
radius `R` and keepout diameter `D`, the critical centre-elbow clearance is
`R/2 - D`. Keep `R >= 2D` for nonnegative conservative clearance.

## Recovered source geometry

The attached STL is unitless; its dimensions were interpreted as millimetres,
consistent with the surrounding OpenSCAD designs and the mating holder.

- Source bounds: 107 x 107 x 31.325 mm.
- Base top: z = 11.325 mm; socket top: z = 31.325 mm.
- Original socket axes: `(0, 0)`, `(+/-23.75, 0)`, and `(0, +/-23.75)`.
- Socket outside diameter: 19.75 mm.
- Antenna bore: 9.75 mm diameter through z = 20, tapering to 8.75 mm at z = 30, then straight to the top.
- Four flex petals per socket: two 0.4 mm diagonal slots at 45 and 135 degrees.
- Fixed mounting axes: `(48.75, -12.15)`, `(12.15, -48.75)`, `(-48.75, 12.15)`, and `(-12.15, 48.75)`.
- Mounting holes: 3.5 mm through diameter with an R1 underside mouth opening to 5.5 mm.
- Each underside antenna raceway has a 9.75 mm width and radius, with its roof axis at z = 5.325 mm.

The mount axes form a 51.760216 x 86.125606 mm rectangle rotated 45 degrees.
They are constants rather than functions of antenna spacing, correcting the
coupling present in the earlier model.

The seven-socket variant retains the original cross and mounting tabs, then
adds a rounded hexagonal carrier and six radial cable-raceway tails. Socket
bores, tapers, slots, top rounds, and raceway clearances are shared with the
reference reconstruction. Its centre raceway is rotated to a gap as described
above; the original five-socket route remains exactly -45 degrees.

## Mesh validation

The production reference export was compared with the supplied STL:

| Check | Supplied STL | Reconstructed STL |
| --- | ---: | ---: |
| Bounds | 107 x 107 x 31.325 mm | 107 x 107 x 31.325 mm |
| Volume | 53,513.486 mm^3 | 53,504.961 mm^3 |
| Volume difference | - | -0.0159% |
| Connected bodies | 1 | 1 |
| Watertight / consistent winding | Yes / yes | Yes / yes |

Cross-sections confirm the antenna and mounting axes at their exact recovered
coordinates. Both rerouted seven-socket rotations are also single watertight
bodies; their measured socket-axis error at the default spacing is below
0.000004 mm.

OpenSCAD cannot directly loft a non-convex offset profile without convexifying
it. The base's R1 bottom and R2 top edge rolls are therefore divided at the
source mesh's 36 quarter-circle angles. In production mode the largest vertical
step is 0.056 mm, below a normal FDM layer height. This is a dimensionally
matched reconstruction, not a bit-for-bit copy of the original triangulation.

The supplied reference file has SHA-256
`a1e74e1611c6b9027d5c63d88bc9293ca1ad833619e40cb52d8556bd1cd1030f`.
Because printer and material shrinkage vary, print a socket or short fit coupon
before committing to a full production batch.
