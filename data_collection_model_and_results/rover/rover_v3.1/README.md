# 2D electric rover (v3.1)

Instead of simulations we can collect real world data by moving several rovers around autonomously in a field. Some rovers will transmit and others will listen and collect data.

## Design

Youtube [link](https://youtu.be/6D6IM0DY81c)

## Construction

### Flash rpi firmware

![select lite os](./flash_pi1.png)

![flash with wifi settings](./flash_pi2.png)

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