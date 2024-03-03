# 2D electric rover (v3.1)

Instead of simulations we can collect real world data by moving several rovers around autonomously in a field. Some rovers will transmit and others will listen and collect data.

## Design

Youtube [link](https://youtu.be/6D6IM0DY81c)

## Construction

### Flash rpi firmware

![select lite os](./flash_pi1.png)

![flash with wifi settings](./flash_pi2.png)

### USB port setup

```

USB  4     |     USB 3

USB  2     |     USB 1

Ethernet

```

### Taranis Q setup

```
Setup (Internal RF)
XJT D16
Ch Range CH1-16

Binding
Press bind, beeping starts
With receiver off, hold down F/S button, power on while holding, let go 1 second later
Solid green/red -> no connection
Flashing green/red -> connection established
Reset receiver
Press ok on controller
Try moving around controller sticks to see X8R light go green


Input 1 -> 100 Rud
Input 2 -> 100 Ele
Input 3 -> 100 Thr
Input 4 -> 100 Alie

CH 1 -> 100 Ail
CH 2 -> 100 Ele
CH 3 -> 100 Thr
CH 4 -> 100 Rud
CH 5 -> 100 SF


```

### SikRadio

[screen shot](./sikradio.jpg)
https://www.youtube.com/watch?v=i5lE2cWJJhM
Connect using mission planner
Set different NetIDs for each pair
Make sure to copy over settings
```
Rover 1 -> NetID 25
Rover 2 -> NetID 32
Rover 3 -> NetID 39
```


### Low power disconnect programming

```
press up to 5s +
press hold set 5s
use up down to set UP to 12.1v
press set
use up down to set DOWN to 12.0v
```

### Rover 1

```
SYSID_THISMAV 1
auto eth0
iface eth0 inet static
    address 192.168.1.41/24
    gateway 192.168.1.254
```

### 3D printed parts

See [here](https://www.dropbox.com/s/egpfn434aox6vvk/roverv3_3dparts.zip?dl=0)

## Missions

### Lab check

### Mission 1

## PDFs

[GoBilda recon](https://www.dropbox.com/scl/fi/ks1fxsgilpyjsh96b6yut/gobilda_recon_assembly.pdf?rlkey=jf0m082piixa4lvxsqi4eruph&dl=0)

[Low power disconnect](https://www.dropbox.com/scl/fi/wmjql1251xnxs90oqn2jd/lower_power_disconnect_30A.pdf?rlkey=h3vitle22f5xrkcthws3yf8ft&dl=0)

[Cytron Smart duo 30](https://www.dropbox.com/scl/fi/eeqg87gi8wzy2aa1k1yx3/MDDS30_User_Manual.pdf?rlkey=xe49gu88bpqspxbg2dh6x139w&dl=0)