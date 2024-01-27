
sdr_width=103;
sdr_height=71.5;
sdr_depth=26;

base=2; // w and d
base_h=5;

offset=10;

strap_w=30;
strap_h=2;
stack_width=sdr_width/4-offset/2;
tol=0.05;
    
hook_out=5;
hook_in=2.5;

array_width=148;
array_height=110;
array_depth=30;

array_screw_hole_width1=1.75; //r screw
array_screw_hole_width2=2; //r screw

poe_screw_hole_width1=1.3; //r screw
poe_screw_hole_width2=1.7; //r screw

array_hole_out=18;
array_hole_in=7;
    
    
module hollow_cylinder(out,in,height) {
    difference() {
        //cylinder(height,out,out);
        translate([-out,-out,0]) cube([out*2,out*2,height]);
        cylinder(height,in,in);
    }
}
module sdr_box() {
    cube([sdr_width+base*2, sdr_height+base_h*2, sdr_depth+base]);
    
    translate([sdr_width/2-stack_width/2+base,-strap_h/2+base_h/2,base]) cube([stack_width,strap_h,sdr_depth+offset/4]);
    
    translate([sdr_width/2-stack_width/2+base,base_h+base_h/2-strap_h/2+sdr_height,base]) cube([stack_width,strap_h,sdr_depth+offset/4]);
}
    
module sdr_cutout() {
            //base
            translate([base,base_h,base]) difference() {
                union() {
                    cube([sdr_width, sdr_height, sdr_depth]);
                    translate([-base*2,offset/2,0]) cube([sdr_width+base*4, sdr_height-offset, sdr_depth]);
                }
            }

            //velcro straps
            for (i = [1,3]) {
                    translate([i*(sdr_width+base*2)/4-strap_w/2,base_h/2-strap_h/2,-offset/2]) cube([strap_w,strap_h,sdr_depth+base+offset]);
                    translate([i*(sdr_width+base*2)/4-strap_w/2,base_h+base_h/2-strap_h/2+sdr_height,-offset/2]) cube([strap_w,strap_h,sdr_depth+base+offset]);
                    
            }

        
        translate([sdr_width/2-stack_width/2+base-tol,-strap_h/2+base_h/2-tol,0]) cube([stack_width+tol*2,strap_h+tol*2,offset]);
            
        translate([sdr_width/2-stack_width/2+base-tol,base_h+base_h/2-strap_h/2+sdr_height-tol,0]) cube([stack_width+tol*2,strap_h+tol*2,offset]);
}


module ziptie_points(height,depth) {
    translate([hook_out,-hook_out]) hollow_cylinder(hook_out,hook_in,height=depth);
    translate([sdr_width+base*2-hook_out,-hook_out]) hollow_cylinder(hook_out,hook_in,height=depth);
    //height=sdr_height+base_h*2;
    translate([hook_out*3,hook_out+height]) hollow_cylinder(hook_out,hook_in,height=depth);
    translate([sdr_width+base*2-3*hook_out,hook_out+height]) hollow_cylinder(hook_out,hook_in,height=depth);
}

module sdr_unit() {
    difference() {
        sdr_box();
        sdr_cutout();
    }
    ziptie_points(height=sdr_height+base_h*2, depth=sdr_depth+base);
}



module holes(xp,yp,r1,r2) {
    for (y = [xp/2,-xp/2]) {
        for (x=[yp/2, -yp/2]) {
            translate([x,y,-1]) cylinder(h=array_depth+2,r1=r1,r2=r2);
        }
    }
}

module array_mount() {
    difference() {
        union() {
            cube([array_width,array_height,array_depth]); 
            translate([(array_width-sdr_width)/2,0,0]) ziptie_points(height=array_height,depth=array_depth);
            translate([array_width/2,array_height+10]) hollow_cylinder(array_hole_out,array_hole_in,array_depth);
        }
        translate([array_width/2,array_height/2+10]) holes(51.6,86.2,array_screw_hole_width1,array_screw_hole_width2); //antenna mount
        translate([array_width/2,array_height/2]) holes(61.6,134.7,poe_screw_hole_width1,poe_screw_hole_width2); //poe
        
        translate([0,10,array_depth/2]) cube([array_width,array_height*2,array_depth/2]);
    }

}

//sdr_unit();

        
array_mount();



