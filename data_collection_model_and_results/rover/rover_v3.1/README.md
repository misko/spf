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

See [here](https://www.dropbox.com/s/egpfn434aox6vvk/3D_printed_parts.zip?dl=0)

## Missions

### Lab check

### Mission 1