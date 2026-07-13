# Power-board supervisor firmware.
#   make test   — host unit tests of the pure state machine (no AVR toolchain)
#   make avr    — cross-compile for ATtiny816 (needs avr-gcc >= 8 with the
#                 Microchip ATtiny device pack, or the megaTinyCore toolchain;
#                 pass DFP=/path/to/Atmel.ATtiny_DFP if headers aren't found)
#   make flash  — program over UPDI via pymcuprog (SerialUPDI adapter on PORT)

MCU = attiny816
F_CPU = 3333333UL
PORT ?= /dev/ttyUSB0
DFP ?=

CC_AVR = avr-gcc
AVRFLAGS = -mmcu=$(MCU) -DF_CPU=$(F_CPU) -Os -Wall -Wextra -std=gnu11 \
           -ffunction-sections -fdata-sections -Wl,--gc-sections
ifneq ($(DFP),)
AVRFLAGS += -B $(DFP)/gcc/dev/$(MCU) -I $(DFP)/include
endif

test: test_lpd_core
	./test_lpd_core

test_lpd_core: test_lpd_core.c lpd_core.h
	cc -O2 -Wall -Wextra -std=c11 -o $@ test_lpd_core.c

avr: supervisor.elf
supervisor.elf: supervisor.c lpd_core.h
	$(CC_AVR) $(AVRFLAGS) -o $@ supervisor.c
	avr-size $@

supervisor.hex: supervisor.elf
	avr-objcopy -O ihex -R .eeprom $< $@

flash: supervisor.hex
	pymcuprog write -t uart -u $(PORT) -d $(MCU) -f $< --erase --verify

clean:
	rm -f test_lpd_core supervisor.elf supervisor.hex

.PHONY: test avr flash clean
