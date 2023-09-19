import serial
import time
import sys
import numpy as np
import matplotlib.pyplot as plt


class GRBLManager:
    # a is the left motor on the wall
    # b is the right motor on the wall
    # origin is very close to b
    #origin_a=np.array([2880,-300]) # a1 / y 
    #origin_b=np.array([-300,-365]) # a2 / x
    #origin_a=np.array([2910,-350]) # a1 / y 
    origin_a=np.array([2910+100,-350]) # a1 / y 
    origin_b=np.array([-300,-370]) # a2 / x

    limit_topleft=np.array([2880-300,4000-200]) 
    limit_bottomright=np.array([0,1500-200]) 
    x_range=2880-300
    y_max=2500

    def from_steps(self,a_motor_steps,b_motor_steps):
        r1=np.linalg.norm(self.origin_a)-a_motor_steps
        r2=np.linalg.norm(self.origin_b)-b_motor_steps
        d=np.linalg.norm(self.origin_a-self.origin_b)

        x=(d*d-r2*r2+r1*r1)/(2*d) # on different axis
        x_dir=(self.origin_b-self.origin_a)/d

        y=np.sqrt(r1*r1-x*x)
        y_dir=np.array([-x_dir[1],x_dir[0]]) #x_dir.T # orthogonal to x 

        xy_g=self.origin_a-y_dir*y+x_dir*x
        return xy_g

    def to_steps(self,p):
        x_frac=p[0]/self.x_range
        y_limit=min(self.y_max,(self.limit_bottomright*(1-x_frac)+self.limit_topleft*x_frac)[1])
        
        p[0]=min(self.x_range,max(0,p[0]))
        p[1]=min(y_limit,max(0,p[1]))

        a_motor_steps=np.linalg.norm(self.origin_a)-np.linalg.norm(self.origin_a-p)

        b_motor_steps=np.linalg.norm(self.origin_b)-np.linalg.norm(self.origin_b-p)#-np.linalg.norm(a2)
        return a_motor_steps,b_motor_steps

    def spiral(self):
        center=np.array([1500,900])
        spiral_radius=1200
        t_max=6*2*np.pi
        v=spiral_radius/t_max
        w=1
        for t in np.linspace(0,t_max,256*16*2):
            x=(v*t)*np.cos(w*t)
            y=(v*t)*np.sin(w*t)
            p=np.array([x,y])+center
            a_motor_steps,b_motor_steps=to_steps(p)
            cmd="G0 X%0.2f Y%0.2f" % (b_motor_steps,a_motor_steps)
            #print("SENDING",x,y,cmd)
            s.write((cmd + '\n').encode()) # Send g-code block to grbl
            s.readline().strip()

    def calibrate(self):
        for x in np.linspace(1500,500,5):
            for y in np.linspace(30,1200,5):
                x,y=to_steps(np.array([x,y]))
                cmd="G0 X%0.2f Y%0.2f" % (x,y)
                s.write((cmd + '\n').encode()) # Send g-code block to grbl

    def __init__(self,serial_fn):
        # Open grbl serial port ==> CHANGE THIS BELOW TO MATCH YOUR USB LOCATION
        self.s = serial.Serial(serial_fn,115200,timeout=0.3,write_timeout=0.3) # GRBL operates at 115200 baud. Leave that part alone.
        self.s.write("?".encode())
        grbl_out=self.s.readline() # get the response
        print("GRBL ONLINE",grbl_out)
        self.position={'time':time.time(),'xy':np.zeros(2)}
        self.update_status()
        time.sleep(0.05)
        self.collect=True

    def push_reset(self):
        self.s.write(b'\x18')
        return self.s.readline().decode().strip()

    def update_status(self,skip_write=False):
        if not skip_write:
            time.sleep(0.01)
            self.s.write("?".encode())
        time.sleep(0.01)

        start_time=time.time()
        response=self.s.readline().decode().strip()
        time.sleep(0.01)
        #print("STATUS",response)
        #<Idle|MPos:-3589.880,79.560,0.000,0.000|FS:0,0>
        try:
            motor_position_str=response.split("|")[1]
        except Exception as e:
            print("FAILED TO PARSE",response,"|e|",e,time.time()-start_time)
            return self.update_status(skip_write=not skip_write)
        b_motor_steps,a_motor_steps,_,_=map(float,motor_position_str[len('MPos:'):].split(','))

        xy=self.from_steps(a_motor_steps,b_motor_steps)
        is_moving=(self.position['xy']!=xy).any()
        self.position={
            'a_motor_steps':a_motor_steps,
            'b_motor_steps':b_motor_steps,
            'xy':xy,
            'is_moving':is_moving,
            'time':time.time()
            }
        return self.position

    def wait_while_moving(self):
        while True:
            old_pos=self.update_status()
            time.sleep(0.05)
            new_pos=self.update_status()
            if old_pos['a_motor_steps']==new_pos['a_motor_steps'] and old_pos['b_motor_steps']==new_pos['b_motor_steps']:
                return 
            time.sleep(0.01)

    def binary_search_edge(self, left, right, xy, direction, epsilon):
        if (right-left)<epsilon:
            return left
        l=(right+left)/2
        p=l*direction+xy
        steps=self.to_steps(np.copy(p))
        actual=self.from_steps(*steps)
        deviation=np.linalg.norm(p-actual)
        if deviation>0.0001:
            #go back left
            return self.binary_search_edge(left,l,xy, direction, epsilon)
        return self.binary_search_edge(l,right,xy,direction,epsilon)

    def bounce(self,bounces,direction=None):
        if direction is None:
            theta=np.random.uniform(2*np.pi)
            direction=np.array([np.sin(theta),np.cos(theta)])
        for _ in range(bounces):
            to_points,new_direction=self.single_bounce(direction)
            #print("MOVE")
            for point in to_points:
                self.move_to(point)
                #print("MOVE")
                self.update_status()
                while np.linalg.norm(self.position['xy']-point)>200:
                    self.update_status()
            if (new_direction!=direction).any(): # we are changing direction
                self.wait_while_moving()
                direction=new_direction
        return direction

    def single_bounce(self,direction,xy=None,step_size=30):
        #find current position
        #pick a random direction
        #take full field step 
        #if hit a wall 
        #direction=np.array([1,0])
        #direction=direction/np.linalg.norm(direction)
        if xy is None:
            self.update_status()
            xy=self.position['xy']
        #find out at what point xy+l*direction we stop changing one of the variables
        epsilon=0.001
        l=self.binary_search_edge(0,10000,xy,direction,epsilon)
        #find a paralell vector to the boundary
        p1=self.from_steps(*self.to_steps((l+2*epsilon)*direction+xy))
        p2=self.from_steps(*self.to_steps((l+3*epsilon)*direction+xy))
        if np.linalg.norm(p1-p2)<epsilon**2: # the direction is only X or Y
            new_direction=-direction
        else:
            b=p2-p1
            b/=np.linalg.norm(b)
            bn=np.array([-b[1],b[0]])
            _xy=self.from_steps(*self.to_steps(xy))
            if np.linalg.norm(self.from_steps(*self.to_steps(_xy+bn))-_xy)<epsilon:
                bn=-bn
            new_direction=np.dot(direction,b)*b-np.dot(direction,bn)*bn
        to_points=[]
        _l=0
        while _l<l:
            _l=min(_l+step_size,l)
            to_points.append(_l*direction+xy)
        theta=np.random.uniform(2*np.pi)
        percent_random=0.05
        new_direction=(1-percent_random)*new_direction+percent_random*np.array([np.sin(theta),np.cos(theta)])
        return to_points,new_direction

    def move_to(self,p):
        a_motor_steps,b_motor_steps=self.to_steps(p)
        cmd="G0 X%0.2f Y%0.2f" % (b_motor_steps,a_motor_steps)
        time.sleep(0.01)
        self.s.write((cmd + '\n').encode()) # Send g-code block to grbl
        time.sleep(0.01)
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        time.sleep(0.01)
        #print("MOVE TO RESPONSE", grbl_out)

    def close(self):
        self.s.close()

if __name__=='__main__':
    if len(sys.argv)!=2:
            print("%s device" % sys.argv[0])
            sys.exit(1)

    serial_fn=sys.argv[1]

    gm=GRBLManager(serial_fn)
    for line in sys.stdin:
        line=line.strip()
        if line=='q':
            sys.exit(1)
        elif line=='r':
            r=push_reset(s)
            print(r)
        elif line=='bounce':
            #point=np.array([2491.49001749,2401.75483327])
            #direction=np.array([0.63471637,0.57157117])
            gm.bounce(20000)
                 
        elif line=='s':
            p=gm.update_status()
            print(p)
        elif line=='c':
            gm.calibrate()
        elif line=='e':
            gm.spiral()
        else:
            if True:
                p_main=np.array([ float(x) for x in line.split() ])
                a_motor_steps,b_motor_steps=gm.to_steps(p_main)
                cmd="G0 X%0.2f Y%0.2f" % (b_motor_steps,a_motor_steps)
                print(gm.from_steps(a_motor_steps,b_motor_steps))
                gm.s.write((cmd + '\n').encode()) # Send g-code block to grbl
                grbl_out = gm.s.readline() # Wait for grbl response with carriage return
                print("MAIN OUT",grbl_out)
        time.sleep(0.01)

    gm.close()
