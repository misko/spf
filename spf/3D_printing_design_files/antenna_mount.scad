wave_length=115/2.7; // 2.6ghz
antenna_width=10-0.2;
antenna_stalk_width=10-0.25;
antenna_base_length=23;

base_width=wave_length+2*antenna_base_length;
height=antenna_width+8;

ziptie_width=3.5;
ziptie_depth=2.5;

dry_screw_r=5.5/2; //dry wall
screw_r=3.5/2; //m3
backing=2;
edges=2;

module antenna_cutout() {
    difference() 
    {
        union() {
            translate([-antenna_width/2-edges-ziptie_depth,-antenna_base_length,0]) cube([
                                antenna_width+ziptie_depth*2+edges*2,
                                antenna_base_length+antenna_width/2+backing,
                                antenna_width+backing]);
            
            cylinder(height,antenna_width/2+backing,antenna_width/2+backing);
             translate([-(antenna_width+ziptie_depth*2+edges*2)/2,0,0]) cube([antenna_width+ziptie_depth*2+edges*2,
            wave_length-antenna_width-2*backing,3]);
        }
        translate([0,0,antenna_width/2]) antenna();
    }
}

module antenna() {
    //antenna cutout below
    translate([antenna_width/2,antenna_width/2,-antenna_width]) rotate([0,0,180]) cube([antenna_width,antenna_base_length+antenna_width/2,antenna_width]);
    
    //the base antenna
    cylinder(height*4,antenna_stalk_width/2,antenna_stalk_width/2);
    translate([0,antenna_width/2,0]) rotate([90,0,0]) {
        cylinder(antenna_base_length+antenna_width/2,antenna_width/2,antenna_width/2);
        }
    
    //a ziptie
    rotate([0,0,180]) translate([antenna_width/2,antenna_width/2+2,-antenna_width]) {
        
        cube([ziptie_depth,ziptie_width,height*2]);
        translate([-antenna_width-ziptie_depth,0,0]) cube([ziptie_depth,ziptie_width,height*2]);
        translate([0,10,0]) {
            cube([ziptie_depth,ziptie_width,height*2]);
            translate([-antenna_width-ziptie_depth,0,0]) cube([ziptie_depth,ziptie_width,height*2]);
        }
    }
}

module screw_hold() {
    difference() {
        cylinder(antenna_width+backing,screw_r+backing,screw_r+backing);
        cylinder(antenna_width+backing,screw_r,screw_r);
    }
}


tx_theta=45;

difference() {
    union() {
        for ( i = [0,90,180,270] ) {
            rotate([0,0,i]) translate([0,-wave_length/2,0]) antenna_cutout();
        }
        /*for ( i = [45,180+45] ) {
            rotate([0,0,i]) translate([0,-wave_length/2+10-20,0]) antenna_cutout();
        }*/
    }

    rotate([0,0,tx_theta]) translate([-antenna_width/2-edges-ziptie_depth,-antenna_base_length,0]) cube([
                        antenna_width+ziptie_depth*2+edges*2,
                        antenna_base_length+antenna_width/2+backing,
                        antenna_width+backing]);
}

rotate([0,0,tx_theta]) antenna_cutout();
    
k1=antenna_width/2+ziptie_depth+(dry_screw_r+backing);
k2=wave_length/2+antenna_base_length-dry_screw_r*2;
translate([k1,-k2,0]) screw_hold();
translate([-k1,k2,0]) screw_hold();
translate([k2,-k1,0]) screw_hold();
translate([-k2,k1,0]) screw_hold();

