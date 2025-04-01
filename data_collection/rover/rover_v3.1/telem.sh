    #start a screen session for each proxy
    # proxy directs roverX to UDP 1457X and 1458X
    
screen -S rover1 -d -m bash -c "/Users/miskodzamba/.virtualenvs/spf/bin/mavproxy.py --force-connected --master=/dev/tty.usbserial-DK0G4IOK --baudrate 57600 --out=127.0.0.1:14571 --out=127.0.0.1:14581 --daemon"
screen -S rover2 -d -m bash -c "/Users/miskodzamba/.virtualenvs/spf/bin/mavproxy.py --force-connected --master=/dev/tty.usbserial-DK0G4W25 --baudrate 57600 --out=127.0.0.1:14572 --out=127.0.0.1:14582 --daemon"
screen -S rover3 -d -m bash -c "/Users/miskodzamba/.virtualenvs/spf/bin/mavproxy.py --force-connected --master=/dev/tty.usbserial-DK0G5WCE --baudrate 57600 --out=127.0.0.1:14573 --out=127.0.0.1:14583 --daemon"

