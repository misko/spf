import iio
ctx = iio.Context('ip:pluto.local')
for dev in ctx.devices:
    for chan in dev.channels:
        for attr in chan.attrs:
            print(dev.name,chan.name,attr)
            #print(f'{dev.name}: {chan.name}: {attr.name}')
