import serial
import time
import sys
import numpy as np
if len(sys.argv)!=2:
        print("%s device" % sys.argv[0])
        sys.exit(1)

serial_fn=sys.argv[1]


def to_steps(p):
    #limits (0,1500)  and (3180-300,4000)
    limit_a1=np.array([3180-300,4000])
    limit_a2=np.array([0,1500])
    
    x_range=3180-300
    x_frac=p[0]/x_range
    y_limit=(limit_a2*(1-x_frac)+limit_a1*x_frac)[1]
    
    p[0]=min(x_range,max(0,p[0]))
    p[1]=min(y_limit,max(0,p[1]))

    a1=np.array([3180,-300])
    ymotor_steps=np.linalg.norm(a1)-np.linalg.norm(a1-p)

    a2=np.array([-300,-365])
    xmotor_steps=np.linalg.norm(a2)-np.linalg.norm(a2-p)#-np.linalg.norm(a2)
    return xmotor_steps,ymotor_steps



# Open grbl serial port ==> CHANGE THIS BELOW TO MATCH YOUR USB LOCATION
s = serial.Serial(serial_fn,115200) # GRBL operates at 115200 baud. Leave that part alone.

s.write("?".encode())
print(s.readline().strip())
s.write("?".encode())
print(s.readline().strip())

#sys.exit(1)
#time.sleep(2)    # Wait for grbl to initialize
#s.flushInput()  # Flush startup text in serial input
for line in sys.stdin:
    line=line.strip()
    if line=='q':
        sys.exit(1)
    elif line=='r':
        print("SENT RESET")
        s.write(b'\x18')
        print(s.readline().strip())
    elif line=='s':
        print("STATUS")
        s.write("?".encode())
        print(s.readline().strip())
    else:
        if True:
            p=np.array([ float(x) for x in line.split() ])
            x,y=to_steps(p)
            cmd="G0 X%0.2f Y%0.2f" % (x,y)
            print("SENDING",cmd)
            s.write((cmd + '\n').encode()) # Send g-code block to grbl
            grbl_out = s.readline() # Wait for grbl response with carriage return
            print(grbl_out)
        #except:
        #    print("ERROR IN INPUT")


# Close file and serial port
f.close()
s.close()
