/*
 * Parametric antenna mount reconstructed from the supplied reference STL.
 *
 * All dimensions are millimetres.  The default `reference_5` configuration
 * recreates the five-socket part.  `hex_7` places six sockets on a regular
 * hexagon and one at the centre.  The mounting-hole axes and antenna fit are
 * intentionally independent of `antenna_spacing`.
 *
 * Examples:
 *   openscad -o antenna_mount_reference_5.stl antenna_mount.scad
 *   openscad -o antenna_mount_hex_7_rot0.stl \
 *       -D 'variant="hex_7"' -D 'outer_rotation=0' antenna_mount.scad
 *   openscad -o antenna_mount_hex_7_rot30.stl \
 *       -D 'variant="hex_7"' -D 'outer_rotation=30' antenna_mount.scad
 */

/* [Layout] */
variant = "reference_5";       // [reference_5,hex_7]
antenna_spacing = 23.75;       // centre-to-centre radius of the outer sockets
outer_rotation = 0;            // [0,30] hex_7 only
center_raceway_angle = -45;    // source direction; hex_7 snaps it to a safe gap
auto_route_hex_center = true;  // keep the centre elbow between outer antennas

/* [Render quality] */
render_quality = "production"; // [draft,production]

/* [Antenna preview (excluded by default)] */
show_antennas = false;
show_preview_mount = true;
show_antenna_keepouts = false;
antenna_preview_height = 95;
antenna_preview_reach = 30;
antenna_preview_joint_height = 19.5;
antenna_preview_joint_diameter = 5.75;
antenna_preview_whip_base_diameter = 9.7;
antenna_preview_whip_tip_diameter = 8.3;
antenna_preview_barrel_diameter = 8.5;
antenna_preview_connector_diameter = 10.5;
antenna_preview_connector_length = 10;
antenna_keepout_diameter = 11;

// The reference STL uses 144 segments on full circles and 36 on R1/R2
// quarter-rounds.  Draft mode is intended only for quick previews/tests.
circle_fragments = render_quality == "production" ? 144 : 48;
fillet_fragments = render_quality == "production" ? 36 : 12;
$fn = circle_fragments;

// Recovered source dimensions.
base_height = 11.325;
part_height = 31.325;
base_half_extent = 53.5;
base_arm_half_width = 9.875;
base_plan_fillet = 2;
base_bottom_fillet = 1;
base_top_fillet = 2;

holder_outer_radius = 9.875;
holder_top_radius = 7.875;
holder_top_fillet = 2;
holder_top_fillet_start = 29.325;
center_root_radius = 11.875;
center_root_fillet = 2;

antenna_lower_radius = 4.875;
antenna_upper_radius = 4.375;
antenna_taper_start = 20;
antenna_taper_end = 30;
raceway_axis_z = 5.325;
raceway_roof_z = raceway_axis_z + antenna_lower_radius; // 10.2
relief_slot_width = 0.4;

mount_hole_radius = 1.75;
mount_mouth_radius = 2.75;
mount_tab_size = 9.5;
mount_centers = [
    [ 48.75, -12.15],
    [ 12.15, -48.75],
    [-48.75,  12.15],
    [-12.15,  48.75]
];

// The four source raceways exit at +/-53.5.  Keeping the same free length on
// hex layouts gives the right-angle antenna body the same clearance.
raceway_free_length = base_half_extent - 23.75; // 29.75
cut_length = 2 * (base_half_extent + antenna_spacing + raceway_free_length);
slice_epsilon = 0.01;
boolean_epsilon = 0.05;

assert(variant == "reference_5" || variant == "hex_7",
       "variant must be reference_5 or hex_7");
assert(antenna_spacing >= center_root_radius + holder_outer_radius,
       "antenna_spacing makes the centre root overlap an outer socket");
assert(variant != "hex_7" || abs(outer_rotation) < 0.000001 ||
       abs(outer_rotation - 30) < 0.000001,
       "hex_7 outer_rotation must be 0 or 30 degrees");

function polar(radius, angle) = [radius * cos(angle), radius * sin(angle)];
function nearest_hex_gap(angle, rotation) =
    rotation + 30 + 60 * round((angle - (rotation + 30)) / 60);

reference_outer_centers = [
    [ antenna_spacing, 0],
    [0,  antenna_spacing],
    [-antenna_spacing, 0],
    [0, -antenna_spacing]
];
reference_outer_angles = [0, 90, 180, 270];

hex_outer_angles = [for (i = [0:5]) outer_rotation + i * 60];
hex_outer_centers = [for (angle = hex_outer_angles) polar(antenna_spacing, angle)];
effective_center_raceway_angle =
    variant == "hex_7" && auto_route_hex_center
        ? nearest_hex_gap(center_raceway_angle, outer_rotation)
        : center_raceway_angle;

outer_centers = variant == "reference_5" ? reference_outer_centers : hex_outer_centers;
outer_raceway_angles = variant == "reference_5" ? reference_outer_angles : hex_outer_angles;
holder_centers = concat([[0, 0]], outer_centers);
raceway_angles = concat([effective_center_raceway_angle], outer_raceway_angles);

if (variant == "hex_7")
    echo(str("hex_7 centre raceway angle: ", effective_center_raceway_angle,
             " degrees"));

if (variant == "hex_7" && auto_route_hex_center && show_antennas)
    echo(str("hex_7 conservative antenna keepout clearance: ",
             antenna_spacing * sin(30) - antenna_keepout_diameter, " mm"));

// Apply a true R-radius fillet to both convex and re-entrant corners without
// changing the nominal footprint.  The alternating radial/delta offsets are
// the 2-D equivalent of filleting every vertical profile edge.
module fillet_all_plan_corners(radius) {
    offset(r = -radius, $fn = fillet_fragments)
        offset(delta = radius)
            offset(r = radius, $fn = fillet_fragments)
                offset(delta = -radius)
                    children();
}

module reference_base_raw_plan() {
    union() {
        square([2 * base_half_extent, 2 * base_arm_half_width], center = true);
        square([2 * base_arm_half_width, 2 * base_half_extent], center = true);

        // Full 9.5 mm squares are used here; the cross arms bury their inner
        // portions and leave the exact four source mounting tabs.
        for (point = mount_centers)
            translate(point)
                square([mount_tab_size, mount_tab_size], center = true);
    }
}

module outward_tail_raw_plan(center, angle) {
    translate(center)
        rotate(angle)
            union() {
                circle(r = holder_outer_radius);
                translate([0, -holder_outer_radius])
                    square([raceway_free_length, 2 * holder_outer_radius]);
            }
}

module hex_carrier_raw_plan() {
    // A rounded hexagonal web supports all seven sockets.  Radial U-channel
    // tails retain the source raceway wall thickness and support-free exits.
    hull()
        for (center = hex_outer_centers)
            translate(center)
                circle(r = holder_outer_radius);

    for (i = [0:len(hex_outer_centers) - 1])
        outward_tail_raw_plan(hex_outer_centers[i], hex_outer_angles[i]);
}

module base_raw_plan() {
    union() {
        // Retaining this cross in both variants keeps the exact mounting tabs
        // and their surrounding structure independent of antenna spacing.
        reference_base_raw_plan();

        if (variant == "hex_7")
            hex_carrier_raw_plan();
    }
}

module base_outer_plan() {
    fillet_all_plan_corners(base_plan_fillet)
        base_raw_plan();
}

function quarter_angle(i) = 90 * i / fillet_fragments;
function bottom_fillet_z(i) =
    base_bottom_fillet * (1 - cos(quarter_angle(i)));
function bottom_fillet_offset(i) =
    base_bottom_fillet * (1 - sin(quarter_angle(i)));
function top_fillet_z(i) =
    base_height - base_top_fillet + base_top_fillet * sin(quarter_angle(i));
function top_fillet_offset(i) =
    base_top_fillet * (1 - cos(quarter_angle(i)));

module rounded_base_outer() {
    // OpenSCAD's hull() convexifies a non-convex cross section, so the edge
    // rolls are represented as fine analytic terraces instead.  Production
    // mode uses the same 36 subdivisions as the reference quarter-rounds;
    // their maximum z step is 0.056 mm (well below a normal print layer).
    union() {
        for (i = [0:fillet_fragments - 1]) {
            z0 = bottom_fillet_z(i);
            z1 = bottom_fillet_z(i + 1);
            translate([0, 0, z0])
                linear_extrude(height = z1 - z0 + slice_epsilon)
                    offset(delta = -bottom_fillet_offset(i))
                        base_outer_plan();
        }

        translate([0, 0, base_bottom_fillet])
            linear_extrude(height = base_height - base_bottom_fillet - base_top_fillet)
                base_outer_plan();

        for (i = [0:fillet_fragments - 1]) {
            z0 = top_fillet_z(i);
            z1 = top_fillet_z(i + 1);
            translate([0, 0, z0])
                linear_extrude(height = z1 - z0)
                    offset(delta = -top_fillet_offset(i + 1))
                        base_outer_plan();
        }
    }
}

function holder_top_arc() = [
    for (i = [0:fillet_fragments])
        let(angle = quarter_angle(i))
            [holder_top_radius + holder_top_fillet * cos(angle),
             holder_top_fillet_start + holder_top_fillet * sin(angle)]
];

module holder_outer_body(center) {
    // Starting at z=9.325 buries the cylinder in the unrounded arm.  As the
    // base top fillet retreats, it exposes the same straight R9.875 patches
    // seen around the four cardinal sockets in the reference mesh.
    translate(center)
        rotate_extrude($fn = circle_fragments)
            polygon(points = concat(
                [[0, base_height - base_top_fillet],
                 [holder_outer_radius, base_height - base_top_fillet],
                 [holder_outer_radius, holder_top_fillet_start]],
                holder_top_arc(),
                [[0, part_height]]
            ));
}

function center_root_arc() = [
    for (i = [0:fillet_fragments])
        let(angle = quarter_angle(i))
            [center_root_radius - center_root_fillet * sin(angle),
             base_height + center_root_fillet * (1 - cos(angle))]
];

module center_root_blend() {
    rotate_extrude($fn = circle_fragments)
        polygon(points = concat(
            [[0, base_height], [center_root_radius, base_height]],
            center_root_arc(),
            [[0, base_height + center_root_fillet]]
        ));
}

module antenna_bore(center) {
    translate([center[0], center[1], -boolean_epsilon])
        cylinder(h = antenna_taper_start + boolean_epsilon,
                 r = antenna_lower_radius);

    translate([center[0], center[1], antenna_taper_start - boolean_epsilon])
        cylinder(h = antenna_taper_end - antenna_taper_start + 2 * boolean_epsilon,
                 r1 = antenna_lower_radius, r2 = antenna_upper_radius);

    translate([center[0], center[1], antenna_taper_end - boolean_epsilon])
        cylinder(h = part_height - antenna_taper_end + 2 * boolean_epsilon,
                 r = antenna_upper_radius);
}

module raceway_plan(center, angle) {
    translate(center)
        rotate(angle)
            union() {
                circle(r = antenna_lower_radius);
                translate([0, -antenna_lower_radius])
                    square([cut_length, 2 * antenna_lower_radius]);
            }
}

module raceway_body(center, angle) {
    // Vertical-sided lower half.
    translate([0, 0, -boolean_epsilon])
        linear_extrude(height = raceway_axis_z + boolean_epsilon)
            raceway_plan(center, angle);

    // Semicircular, support-free roof.  The lower half overlaps the prism.
    translate([center[0], center[1], raceway_axis_z])
        rotate([0, 0, angle])
            rotate([0, 90, 0])
                cylinder(h = cut_length, r = antenna_lower_radius);
}

module raceway_mouth_slice(center, angle, z, expansion) {
    translate([0, 0, z - slice_epsilon / 2])
        linear_extrude(height = slice_epsilon)
            offset(delta = expansion)
                raceway_plan(center, angle);
}

module raceway_bottom_roundover(center, angle) {
    for (i = [0:fillet_fragments - 1]) {
        z0 = bottom_fillet_z(i);
        z1 = bottom_fillet_z(i + 1);
        hull() {
            raceway_mouth_slice(center, angle, z0, bottom_fillet_offset(i));
            raceway_mouth_slice(center, angle, z1, bottom_fillet_offset(i + 1));
        }
    }
}

module holder_relief_slots(center) {
    for (angle = [45, 135])
        translate([center[0], center[1], base_height])
            rotate([0, 0, angle])
                translate([-center_root_radius - 1, -relief_slot_width / 2, 0])
                    cube([2 * (center_root_radius + 1), relief_slot_width,
                          part_height - base_height + boolean_epsilon]);
}

function mount_roundover_arc() = [
    for (i = [0:fillet_fragments])
        [mount_hole_radius + bottom_fillet_offset(i), bottom_fillet_z(i)]
];

module mounting_hole(center) {
    translate(center)
        rotate_extrude($fn = circle_fragments)
            polygon(points = concat(
                [[0, -boolean_epsilon], [mount_mouth_radius, -boolean_epsilon]],
                mount_roundover_arc(),
                [[mount_hole_radius, base_height + boolean_epsilon],
                 [0, base_height + boolean_epsilon]]
            ));
}

module antenna_mount_geometry() {
    difference() {
        union() {
            rounded_base_outer();

            for (center = holder_centers)
                holder_outer_body(center);

            // The source has this R2 flare only on its centre socket.
            center_root_blend();
        }

        union() {
            for (i = [0:len(holder_centers) - 1]) {
                antenna_bore(holder_centers[i]);
                raceway_body(holder_centers[i], raceway_angles[i]);
                raceway_bottom_roundover(holder_centers[i], raceway_angles[i]);
                holder_relief_slots(holder_centers[i]);
            }

            for (center = mount_centers)
                mounting_hole(center);
        }
    }
}

module horizontal_cylinder(length, radius, fragments = circle_fragments) {
    rotate([0, 90, 0])
        cylinder(h = length, r = radius, $fn = fragments);
}

module upper_hemisphere(radius) {
    intersection() {
        sphere(r = radius, $fn = circle_fragments);
        translate([-radius, -radius, 0])
            cube([2 * radius, 2 * radius, radius + boolean_epsilon]);
    }
}

module antenna_preview_local() {
    vertical_bottom = raceway_axis_z - antenna_preview_connector_diameter / 2;
    joint_top = vertical_bottom + antenna_preview_joint_height;
    antenna_top = vertical_bottom + antenna_preview_height;
    tip_radius = antenna_preview_whip_tip_diameter / 2;
    whip_cone_top = antenna_top - tip_radius;
    connector_body_length = antenna_preview_reach - antenna_preview_connector_length;

    color([0.035, 0.04, 0.05]) {
        translate([0, 0, vertical_bottom])
            cylinder(h = antenna_preview_joint_height,
                     r = antenna_preview_joint_diameter / 2);

        translate([0, 0, joint_top])
            cylinder(h = whip_cone_top - joint_top,
                     r1 = antenna_preview_whip_base_diameter / 2,
                     r2 = antenna_preview_whip_tip_diameter / 2);

        translate([0, 0, whip_cone_top])
            upper_hemisphere(tip_radius);

        translate([0, 0, raceway_axis_z])
            horizontal_cylinder(connector_body_length,
                                antenna_preview_barrel_diameter / 2);
    }

    // A low-facet distal sleeve suggests the knurled coupling in the photo.
    color([0.09, 0.095, 0.105])
        translate([connector_body_length, 0, raceway_axis_z])
            horizontal_cylinder(antenna_preview_connector_length,
                                antenna_preview_connector_diameter / 2, 32);
}

module antenna_keepout_local() {
    vertical_bottom = raceway_axis_z - antenna_keepout_diameter / 2;
    preview_vertical_bottom =
        raceway_axis_z - antenna_preview_connector_diameter / 2;
    vertical_height =
        antenna_preview_height + preview_vertical_bottom - vertical_bottom;

    union() {
        translate([0, 0, vertical_bottom])
            cylinder(h = vertical_height,
                     r = antenna_keepout_diameter / 2);

        translate([0, 0, raceway_axis_z])
            horizontal_cylinder(antenna_preview_reach,
                                antenna_keepout_diameter / 2);
    }
}

module antenna_at(index, keepout = false) {
    translate([holder_centers[index][0], holder_centers[index][1], 0])
        rotate([0, 0, raceway_angles[index]])
            if (keepout)
                antenna_keepout_local();
            else
                antenna_preview_local();
}

module all_antenna_previews() {
    for (i = [0:len(holder_centers) - 1])
        antenna_at(i);
}

module all_antenna_keepouts() {
    for (i = [0:len(holder_centers) - 1])
        antenna_at(i, true);
}

if (show_antennas) {
    if (show_preview_mount)
        color([0.72, 0.76, 0.82, 0.48])
            antenna_mount_geometry();

    all_antenna_previews();

    if (show_antenna_keepouts)
        color([0.1, 0.55, 1.0, 0.28])
            all_antenna_keepouts();
} else {
    antenna_mount_geometry();
}
