/*
 * Flat base for the Rover 3.1 antenna-array mount.
 *
 * The four mounting axes come from the production seven-antenna STL.  The
 * 4.40 mm insert bores and 4.00 mm plate thickness are copied from the
 * supplied Rover 3.1 top-front STL so the existing E-Z LOK 260-M3-BR fit is
 * preserved.  All dimensions are millimetres.
 */

/* [Base] */
base_length = 112;
base_width = 112;
base_thickness = 4;
corner_radius = 3;

/* [Insert fit] */
insert_hole_diameter = 4.4;

/* [Render quality] */
circle_fragments = 144;
$fn = circle_fragments;

mount_centers = [
    [ 48.75, -12.15],
    [ 12.15, -48.75],
    [-48.75,  12.15],
    [-12.15,  48.75]
];

boolean_epsilon = 0.05;

assert(base_length > 0 && base_width > 0 && base_thickness > 0,
       "Base dimensions must be positive");
assert(corner_radius >= 0 &&
       2 * corner_radius < min(base_length, base_width),
       "corner_radius is too large for the base");
assert(insert_hole_diameter > 3,
       "Insert hole must clear an M3 fastener");

function nearest_edge_clearance(point) = min(
    base_length / 2 - abs(point[0]),
    base_width / 2 - abs(point[1])
) - insert_hole_diameter / 2;

for (point = mount_centers)
    assert(nearest_edge_clearance(point) >= 3,
           str("Mount hole at ", point,
               " is too close to the selected base edge"));

module rounded_rectangle_2d(length, width, radius) {
    if (radius == 0)
        square([length, width], center = true);
    else
        offset(r = radius)
            square([length - 2 * radius, width - 2 * radius], center = true);
}

difference() {
    linear_extrude(height = base_thickness)
        rounded_rectangle_2d(base_length, base_width, corner_radius);

    for (point = mount_centers)
        translate([point[0], point[1], -boolean_epsilon])
            cylinder(h = base_thickness + 2 * boolean_epsilon,
                     d = insert_hole_diameter);
}
