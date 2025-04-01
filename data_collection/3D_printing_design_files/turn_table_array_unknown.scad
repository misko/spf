$fn=$preview ? 32 : 256;

wave_length=125/2;

dipole_antenna_width=10;
dipole_antenna_stalk_width=10-0.25;
dipole_antenna_base_length=23;

monopole_antenna_width=9.0;
monopole_antenna_stalk_width=9.0+0.1;
monopole_antenna_base_length=20;

plate_height=5;
plate_radius=100;

base_width=wave_length+2*dipole_antenna_base_length;
height=dipole_antenna_width+8;

ziptie_width=3.5;
ziptie_depth=2.5;

backing=2;
edges=2;

insert_width=10+ziptie_depth*2+edges*2; //width of insert
insert_height=dipole_antenna_width+backing;
insert_length=dipole_antenna_base_length+dipole_antenna_width/2+backing;
antenna_brace_width=dipole_antenna_width/2+backing;

module antenna_cutout() {
    difference() 
    {
        union() {
            //base block where antenna mounts
            translate([insert_width/2,
                        insert_width/2,
                        0]) rotate([0,0,180]) 
                cube([
                    insert_width,
                    insert_length,
                    insert_height]
                );
            
            //antenna brace
            cylinder(height,antenna_brace_width,antenna_brace_width);
            
            //connector thin rectangle
            translate(
                [-insert_width/2,0,0]
            ) {
                cube(
                    [insert_width,
                    wave_length/2, //distance to center of antenna in block above
                    3]
                );
            }
        }
        children();
        
    }
}

module add_zipties() {
        for (j=[0:1]) {
            translate([0,-j*2*ziptie_width-2,0]) {
                for (i=[-1:2:1]) {
                    translate([-ziptie_depth/2-i*(insert_width/2-ziptie_depth*2),
                            -(insert_width/4+ziptie_width),
                            0]) 
                        cube([ziptie_depth,ziptie_width,2*height],center=false);
                }
            }
        }
        children();
}

module dipole_antenna() {
    translate([0,0,dipole_antenna_width/2]) {
        //antenna cutout below
        translate([dipole_antenna_width/2,dipole_antenna_width/2,-dipole_antenna_width]) rotate([0,0,180]) cube([dipole_antenna_width,insert_length,dipole_antenna_width]);
        
        //the base antenna
        cylinder(height*4,dipole_antenna_stalk_width/2,dipole_antenna_stalk_width/2);
        translate([0,dipole_antenna_width/2,0]) rotate([90,0,0]) {
            cylinder(insert_length,dipole_antenna_width/2,dipole_antenna_width/2);
        }
    }
    
}

module monopole_antenna() {
    translate([0,0,monopole_antenna_width/2]) {
        //antenna cutout below
        translate([monopole_antenna_width/2,monopole_antenna_width/2,-monopole_antenna_width]) rotate([0,0,180]) cube([monopole_antenna_width,insert_length,monopole_antenna_width]);
        
        //the base antenna
        cylinder(height*2,monopole_antenna_stalk_width/2,monopole_antenna_stalk_width/2+0.3);
        translate([0,monopole_antenna_width/2,0]) rotate([90,0,0]) {
            cylinder(insert_length,monopole_antenna_width/2,monopole_antenna_width/2);
            }
        }
    
}


module antenna_array() {
     for (i=[0:3] ) {
           rotate([0,0,90*i]) translate([0,-wave_length/2,0]) children();
     }
}

module top_plate() {
    translate([0,0,-plate_height]) {
        difference() {
            cylinder(plate_height,plate_radius-1.0,plate_radius);
            cylinder(plate_height+1,plate_radius-20,plate_radius-20);
        }
        difference() {
            union() {
                for ( i = [0 : 4] ){
                    rotate([0,0,90*i]) 
                        translate([-(insert_width+6)/2,0,0]) 
                            cube([insert_width+6,
                        plate_radius-4,plate_height*1.5]);
                }
            }
            for ( i = [0 : 4] ){
                union() {
                    rotate([0,0,90*i]) 
                translate([-(insert_width+0.3)/2,0,plate_height]) 
                    cube([insert_width+0.3,
                        plate_radius-4,plate_height]);
                }
            }
            translate([0,0,plate_height*1.5]) cube([40,40,plate_height],center=true);
        }
    }
}

module bottom_plate()  {
    difference() {
        translate([0,0,-plate_height*2]) cylinder(plate_height*2,plate_radius+3,plate_radius+3);
    union() {
        translate([0,0,-plate_height]) cylinder(plate_height,plate_radius+0.15,plate_radius+0.15);
        translate([0,0,-plate_height*2]) cylinder(plate_height*2,plate_radius-20,plate_radius-20);
        for (i = [0:4] ) {
            rotate([0,0,i*90]) translate([plate_radius-10,0,-plate_height*2]) cylinder(plate_height,4/2,8/2);
        }
    }
}
}

//top_plate();
//bottom_plate();
antenna_array() antenna_cutout() add_zipties() monopole_antenna();
//antenna_array() antenna_cutout() add_zipties() dipole_antenna();
//cube([wave_length,wave_length,wave_length],center=true);
//antenna_cutout() add_zipties() dipole_antenna();

        