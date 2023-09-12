import serial
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

# a is the left motor on the wall
# b is the right motor on the wall
# origin is very close to b
#origin_a=np.array([2880,-300]) # a1 / y 
#origin_b=np.array([-300,-365]) # a2 / x
#origin_a=np.array([2910,-350]) # a1 / y 
origin_a=np.array([2910+100,-350]) # a1 / y 
origin_b=np.array([-300,-370]) # a2 / x

def from_steps(a_motor_steps,b_motor_steps):
    r1=np.linalg.norm(origin_a)-a_motor_steps
    r2=np.linalg.norm(origin_b)-b_motor_steps
    d=np.linalg.norm(origin_a-origin_b)

    x=(d*d-r2*r2+r1*r1)/(2*d) # on different axis
    x_dir=(origin_b-origin_a)/d

    y=np.sqrt(r1*r1-x*x)
    y_dir=np.array([-x_dir[1],x_dir[0]]) #x_dir.T # orthogonal to x 

    xy_g=origin_a-y_dir*y+x_dir*x
    return xy_g

def to_steps(p):
    #limits (0,1500)  and (3180-300,4000)
    limit_topleft=np.array([2880-300,4000-200]) 
    limit_bottomright=np.array([0,1500-200]) 
    
    x_range=2880-300
    x_frac=p[0]/x_range
    y_limit=min(2500,(limit_bottomright*(1-x_frac)+limit_topleft*x_frac)[1])
    
    p[0]=min(x_range,max(0,p[0]))
    p[1]=min(y_limit,max(0,p[1]))

    a_motor_steps=np.linalg.norm(origin_a)-np.linalg.norm(origin_a-p)

    b_motor_steps=np.linalg.norm(origin_b)-np.linalg.norm(origin_b-p)#-np.linalg.norm(a2)
    return a_motor_steps,b_motor_steps

def spiral():
    center=np.array([1500,900])
    spiral_radius=1200
    t_max=6*2*np.pi
    v=spiral_radius/t_max
    w=1
    for t in np.linspace(0,t_max,256*16*2):
        x=(v*t)*np.cos(w*t)
        y=(v*t)*np.sin(w*t)
        print(x,y)
        p=np.array([x,y])+center
        a_motor_steps,b_motor_steps=to_steps(p)
        cmd="G0 X%0.2f Y%0.2f" % (b_motor_steps,a_motor_steps)
        #print("SENDING",x,y,cmd)
        s.write((cmd + '\n').encode()) # Send g-code block to grbl
        print(s.readline().strip())

def calibrate():
    for x in np.linspace(1500,500,5):
        for y in np.linspace(30,1200,5):
            x,y=to_steps(np.array([x,y]))
            cmd="G0 X%0.2f Y%0.2f" % (x,y)
            print("SENDING",cmd)
            s.write((cmd + '\n').encode()) # Send g-code block to grbl


def open_grbl(serial_fn):
    # Open grbl serial port ==> CHANGE THIS BELOW TO MATCH YOUR USB LOCATION
    s = serial.Serial(serial_fn,115200,timeout=2,write_timeout=2.0) # GRBL operates at 115200 baud. Leave that part alone.
    s.write("?".encode())
    grbl_out=s.readline() # get the response
    print("GRBL ONLINE",grbl_out)
    time.sleep(0.05)
    return s

def push_reset(s):
    s.write(b'\x18')
    return s.readline().decode().strip()

def get_status(s):
    time.sleep(0.1)
    s.write("?".encode())
    time.sleep(0.1)
    response=s.readline().decode().strip()
    time.sleep(0.1)
    #print("STATUS",response)
    #<Idle|MPos:-3589.880,79.560,0.000,0.000|FS:0,0>
    try:
        motor_position_str=response.split("|")[1]
    except:
        print("FAILED TO PARSE",response)
        return get_status(s)
    b_motor_steps,a_motor_steps,_,_=map(float,motor_position_str[len('MPos:'):].split(','))
    time.sleep(0.01)
    return a_motor_steps,b_motor_steps,from_steps(a_motor_steps,b_motor_steps)

def wait_while_moving(s):
    while True:
        a,b,xy=get_status(s)
        time.sleep(0.1)
        _a,_b,_xy=get_status(s)
        if _a==a and _b==b:
            return 
        print(a,b)
        time.sleep(0.1)

def binary_search_edge(left, right, xy, direction, epsilon):
    if (right-left)<epsilon:
        return left
    l=(right+left)/2
    p=l*direction+xy
    steps=to_steps(np.copy(p))
    actual=from_steps(*steps)
    deviation=np.linalg.norm(p-actual)
    if deviation>0.0001:
        #go back left
        return binary_search_edge(left,l,xy, direction, epsilon)
    return binary_search_edge(l,right,xy,direction,epsilon)

def full_field_step(s,direction):
    #find out at what point xy+l*direction we stop changing one of the variables
    l=binary_search_edge(0,10000,xy,direction,0.01)
    #from_steps(*to_steps(l*direction+s))

def bounce(s,direction,xy=None,step_size=30):
    #find current position
    #a_motor_steps,b_motor_steps,xy=get_status(s)
    #pick a random direction
    #take full field step 
    #if hit a wall 
    #direction=np.array([1,0])
    #direction=direction/np.linalg.norm(direction)
    if xy is None:
        a_motor_steps,b_motor_steps,xy=get_status(s)
    #find out at what point xy+l*direction we stop changing one of the variables
    epsilon=0.001
    l=binary_search_edge(0,10000,xy,direction,epsilon)
    #find a paralell vector to the boundary
    p1=from_steps(*to_steps((l+2*epsilon)*direction+xy))
    p2=from_steps(*to_steps((l+3*epsilon)*direction+xy))
    if np.linalg.norm(p1-p2)<epsilon**2: # the direction is only X or Y
        new_direction=-direction
    else:
        b=p2-p1
        b/=np.linalg.norm(b)
        bn=np.array([-b[1],b[0]])
        _xy=from_steps(*to_steps(xy))
        if np.linalg.norm(from_steps(*to_steps(_xy+bn))-_xy)<epsilon:
            bn=-bn
        new_direction=np.dot(direction,b)*b-np.dot(direction,bn)*bn
    to_points=[]
    _l=0
    while _l<l:
        _l=min(_l+step_size,l)
        to_points.append(_l*direction+xy)
    #print("FROM",xy,"TO",_to)
    theta=np.random.uniform(2*np.pi)
    percent_random=0.05
    new_direction=(1-percent_random)*new_direction+percent_random*np.array([np.sin(theta),np.cos(theta)])
    return to_points,new_direction

def move_to(s,p):
    a_motor_steps,b_motor_steps=to_steps(p)
    cmd="G0 X%0.2f Y%0.2f" % (b_motor_steps,a_motor_steps)
    time.sleep(0.1)
    s.write((cmd + '\n').encode()) # Send g-code block to grbl
    time.sleep(0.1)
    grbl_out = s.readline() # Wait for grbl response with carriage return
    time.sleep(0.1)

if __name__=='__main__':
    if len(sys.argv)!=2:
            print("%s device" % sys.argv[0])
            sys.exit(1)

    serial_fn=sys.argv[1]

    s=open_grbl(serial_fn)
    for line in sys.stdin:
        line=line.strip()
        if line=='q':
            sys.exit(1)
        elif line=='r':
            r=push_reset(s)
            print(r)
        elif line=='f':
            theta=np.random.uniform(2*np.pi)
            direction=np.array([np.sin(theta),np.cos(theta)])

            #point=np.array([2491.49001749,2401.75483327])
            #direction=np.array([0.63471637,0.57157117])
            #print("Point",point,direction)
            for _ in range(20000):
                to_points,new_direction=bounce(s,direction)
                print("MOVE")
                for point in to_points:
                    move_to(s,point)
                if (new_direction!=direction).any(): # we are changing direction
                    wait_while_moving(s)
                    direction=new_direction
                 
            #full_field_step(s,np.array([1,0]))
        elif line=='s':
            r=get_status(s)
            #<Idle|MPos:-3589.880,79.560,0.000,0.000|FS:0,0>
            print(r)
        elif line=='c':
            calibrate()
        elif line=='e':
            spiral()
        else:
            if True:
                p_main=np.array([ float(x) for x in line.split() ])
                a_motor_steps,b_motor_steps=to_steps(p_main)
                cmd="G0 X%0.2f Y%0.2f" % (b_motor_steps,a_motor_steps)
                print("SENDING",cmd)
                print(from_steps(a_motor_steps,b_motor_steps))
                s.write((cmd + '\n').encode()) # Send g-code block to grbl
                grbl_out = s.readline() # Wait for grbl response with carriage return
                print(grbl_out)
        time.sleep(0.1)

    s.close()
