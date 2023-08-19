import serial
import time
import sys
import numpy as np
if len(sys.argv)!=2:
        print("%s device" % sys.argv[0])
        sys.exit(1)

serial_fn=sys.argv[1]

# a is the left motor on the wall
# b is the right motor on the wall
# origin is very close to b
origin_a=np.array([2880,-300]) # a1 / y 
origin_b=np.array([-300,-365]) # a2 / x

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
        cmd="G0 X%0.2f Y%0.2f" % (a_motor_steps,b_motor_steps)
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


# Open grbl serial port ==> CHANGE THIS BELOW TO MATCH YOUR USB LOCATION
s = serial.Serial(serial_fn,115200) # GRBL operates at 115200 baud. Leave that part alone.

s.write("?".encode())
print(s.readline().strip())
s.write("?".encode())
print(s.readline().strip())

def push_reset(s):
    s.write(b'\x18')
    return s.readline().decode().strip()

def get_status(s):
    s.write("?".encode())
    response=s.readline().decode().strip()
    #<Idle|MPos:-3589.880,79.560,0.000,0.000|FS:0,0>
    motor_position_str=response.split("|")[1]
    xmotor,ymotor,
    return s.readline().decode().strip()


#sys.exit(1)
#time.sleep(2)    # Wait for grbl to initialize
#s.flushInput()  # Flush startup text in serial input
for line in sys.stdin:
    line=line.strip()
    if line=='q':
        sys.exit(1)
    elif line=='r':
        r=push_reset(s)
        print(r)
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
            p=np.array([ float(x) for x in line.split() ])
            a_motor_steps,b_motor_steps=to_steps(p)
            cmd="G0 X%0.2f Y%0.2f" % (b_motor_steps,a_motor_steps)
            print("SENDING",cmd)
            print(from_steps(a_motor_steps,b_motor_steps))
            s.write((cmd + '\n').encode()) # Send g-code block to grbl
            grbl_out = s.readline() # Wait for grbl response with carriage return
            print(grbl_out)
        #except:
        #    print("ERROR IN INPUT")


# Close file and serial port
f.close()
s.close()
