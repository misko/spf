import serial
import time
import sys

if len(sys.argv)!=3:
        print("%s device file" % sys.argv[0])
        sys.exit(1)

serial_fn,gcode_fn=sys.argv[1:]


# Open grbl serial port ==> CHANGE THIS BELOW TO MATCH YOUR USB LOCATION
s = serial.Serial(serial_fn,115200) # GRBL operates at 115200 baud. Leave that part alone.

# Open g-code file
f = open(gcode_fn,'r');

# Wake up grbl
s.write("?".encode())
print(s.readline().strip())
s.write(b'\x18')
print(s.readline().strip())
s.write("?".encode())
print(s.readline().strip())
s.write("?".encode())
print(s.readline().strip())
#s.write("G0 X100\n".encode())
#print(s.readline().strip())

#sys.exit(1)
#time.sleep(2)    # Wait for grbl to initialize
#s.flushInput()  # Flush startup text in serial input

# Stream g-code to grbl
for line in f:
        l = line.strip() # Strip all EOL characters for consistency
        if l[0]==';':
            continue
        print('Sending: ' + l)
        s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = s.readline() # Wait for grbl response with carriage return
        print(grbl_out)

        # Wait here until grbl is finished to close serial port and file.
        #raw_input("     Press <Enter> to exit and disable grbl.")

# Close file and serial port
f.close()
s.close()
