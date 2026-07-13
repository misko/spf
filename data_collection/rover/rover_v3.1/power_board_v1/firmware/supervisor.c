/* supervisor.c — ATtiny816 power-board supervisor (AVR shell around lpd_core.h).
 *
 * Pin map: P1_DETAIL_DESIGN.md §3.  I2C register map: README.md.
 * Clock: 20 MHz internal / 6 -> 3.33 MHz (safe at 3.3 V, low power).
 * Tick: RTC PIT @ ~32 Hz; state machine stepped every ~100 ms.
 * WDT 1 s, kicked only from the main loop after a successful step.
 */
#include <avr/interrupt.h>
#include <avr/io.h>
#include <avr/sleep.h>
#include <avr/wdt.h>
#include <stdbool.h>
#include <stdint.h>

#include "lpd_core.h"

#define FW_VERSION 0x11 /* v1.1 */
#define I2C_ADDR 0x36

/* ---- pins (PORTA/B/C bit indices per P1_DETAIL_DESIGN.md §3) ---- */
#define PA_USB_EN1 1
#define PA_USB_EN2 2
#define PA_LED 3
#define PA_VIN_AIN 4  /* AIN4  */
#define PA_FAULT_USB 5
#define PA_V5A_AIN 6  /* AIN6  */
#define PA_NTC_AIN 7  /* AIN7  */
#define PB_SCL 0
#define PB_SDA 1
#define PB_SW_SENSE 2
#define PB_AUX_CTL 3
#define PB_FE_EN 4
#define PB_BUCKA_EN 5
#define PC_PGOOD_A 0
#define PC_PGOOD_B 1
#define PC_LOW_BATT 2
#define PC_SHDN_ACK 3

/* VIN divider 680k/100k (ratio 7.8, high-Z: 16 uA; 100 nF hold cap on the tap),
 * ADC ref 2.5 V, 10-bit, 16x accumulate.
 * mv = acc/16 * 2500/1024 * 7.8  ->  mv = acc * 1190 / 1000 (fits u32). */
#define VIN_MV_FROM_ACC(acc) ((uint16_t)(((uint32_t)(acc) * 1190ul) / 1000ul))
/* V5A divider 30k/10k (ratio 4): mv = acc/16 * 2500/1024 * 4 = acc * 610 / 1000 */
#define V5A_MV_FROM_ACC(acc) ((uint16_t)(((uint32_t)(acc) * 610ul) / 1000ul))

/* ---- I2C register file ---- */
enum {
    REG_STATE = 0x00,
    REG_VIN_L = 0x01,
    REG_VIN_H = 0x02,
    REG_V5A_L = 0x03,
    REG_V5A_H = 0x04,
    REG_TEMP = 0x05,
    REG_FLAGS = 0x06,
    REG_CMD_USB = 0x10, /* RW: bit0 cycle port1, bit1 cycle port2 */
    REG_CMD_AUX = 0x11, /* RW: bit0 AUX enable request (ANDed with state) */
    REG_CMD_SHDN = 0x12, /* RW: write 0xA5 = Pi requests shutdown (as SHDN_ACK) */
    REG_FWVER = 0xF0,
    REG_FILE_SIZE = 0xF1
};
static volatile uint8_t regs[REG_FILE_SIZE];
static volatile uint8_t reg_ptr;
static volatile bool reg_ptr_pending;

/* flags bits */
#define FLG_PGOOD_A 0
#define FLG_PGOOD_B 1
#define FLG_FAULT_USB 2
#define FLG_SW 3
#define FLG_LOW_BATT 4
#define FLG_AUX 5

static volatile uint8_t rtc_ticks; /* ~32 Hz from PIT */
static uint32_t now_ms;

ISR(RTC_PIT_vect) {
    RTC.PITINTFLAGS = RTC_PI_bm;
    rtc_ticks++;
}

/* ---- TWI0 slave: register-file protocol (ptr write, then read/write) ---- */
ISR(TWI0_TWIS_vect) {
    uint8_t status = TWI0.SSTATUS;
    if (status & TWI_APIF_bm) { /* address or stop */
        if (status & TWI_AP_bm) {
            reg_ptr_pending = true; /* first data byte sets the pointer */
            TWI0.SCTRLB = TWI_SCMD_RESPONSE_gc;
        } else {
            TWI0.SCTRLB = TWI_SCMD_COMPTRANS_gc;
        }
        return;
    }
    if (status & TWI_DIF_bm) {
        if (status & TWI_DIR_bm) { /* master reads */
            TWI0.SDATA = (reg_ptr < REG_FILE_SIZE) ? regs[reg_ptr] : 0xFF;
            reg_ptr++;
            TWI0.SCTRLB = TWI_SCMD_RESPONSE_gc;
        } else { /* master writes */
            uint8_t d = TWI0.SDATA;
            if (reg_ptr_pending) {
                reg_ptr = d;
                reg_ptr_pending = false;
            } else {
                /* only command registers are writable */
                if (reg_ptr == REG_CMD_USB || reg_ptr == REG_CMD_AUX ||
                    reg_ptr == REG_CMD_SHDN)
                    regs[reg_ptr] = d;
                reg_ptr++;
            }
            TWI0.SCTRLB = TWI_SCMD_RESPONSE_gc;
        }
    }
}

static uint16_t adc_read_acc16(uint8_t muxpos) {
    ADC0.MUXPOS = muxpos;
    ADC0.COMMAND = ADC_STCONV_bm;
    while (!(ADC0.INTFLAGS & ADC_RESRDY_bm))
        ;
    ADC0.INTFLAGS = ADC_RESRDY_bm;
    return ADC0.RES; /* 16x accumulated 10-bit -> /16 folded into scaling */
}

static void hw_init(void) {
    /* FIRST action out of reset (review fix): drive FE_EN low so the LM74800
     * EN divider cannot self-enable the rails during boot (pack plug-in / WDT
     * reset would otherwise pulse the rails ON for ~100+ ms). C16 (2.2 uF on
     * FE_EN) covers the few ms before this line executes. */
    PORTB.OUTCLR = (1 << PB_FE_EN);
    PORTB.DIRSET = (1 << PB_FE_EN);

    /* clock: 20 MHz OSC / 6 */
    CCP = CCP_IOREG_gc;
    CLKCTRL.MCLKCTRLB = CLKCTRL_PDIV_6X_gc | CLKCTRL_PEN_bm;

    /* outputs, safe defaults: everything off/deasserted */
    PORTA.OUTCLR = (1 << PA_LED);
    PORTA.OUTSET = (1 << PA_USB_EN1) | (1 << PA_USB_EN2); /* TPS2553 EN active-high: ports on when rail is on */
    PORTA.DIRSET = (1 << PA_USB_EN1) | (1 << PA_USB_EN2) | (1 << PA_LED);
    PORTB.OUTCLR = (1 << PB_AUX_CTL);
    PORTB.DIRSET = (1 << PB_AUX_CTL);
    /* FE_EN emulated open-drain: input = released (divider pulls up), output-low = off */
    PORTB.OUTCLR = (1 << PB_FE_EN);
    PORTB.DIRCLR = (1 << PB_FE_EN);
    PORTB.DIRCLR = (1 << PB_BUCKA_EN); /* released; reserved override */
    /* LOW_BATT to Pi: active-low, open-drain emulated (Pi-side pull-up to its 3V3) */
    PORTC.OUTCLR = (1 << PC_LOW_BATT);
    PORTC.DIRCLR = (1 << PC_LOW_BATT); /* released = not asserted */

    /* inputs with pullups where the net is a bare switch/open-drain */
    PORTB.PIN2CTRL = PORT_PULLUPEN_bm; /* SW_SENSE: switch shorts to GND = ON */
    PORTA.PIN5CTRL = PORT_PULLUPEN_bm; /* FAULT_USB wired-OR, active low */
    PORTC.PIN0CTRL = 0;                /* PGOODs have board pullups to 3V3_AON */
    PORTC.PIN3CTRL = PORT_PULLUPEN_bm; /* SHDN_ACK from Pi, active low */

    /* ADC0: 2.5 V internal ref, 16x accumulation, reduced clock */
    VREF.CTRLA = VREF_ADC0REFSEL_2V5_gc;
    ADC0.CTRLC = ADC_REFSEL_INTREF_gc | ADC_PRESC_DIV16_gc | ADC_SAMPCAP_bm;
    ADC0.CTRLB = ADC_SAMPNUM_ACC16_gc;
    ADC0.CTRLA = ADC_ENABLE_bm;

    /* RTC PIT from 32 kHz ULP, period 1024 -> 32 Hz tick */
    while (RTC.STATUS || RTC.PITSTATUS)
        ;
    RTC.CLKSEL = RTC_CLKSEL_INT32K_gc;
    RTC.PITINTCTRL = RTC_PI_bm;
    RTC.PITCTRLA = RTC_PERIOD_CYC1024_gc | RTC_PITEN_bm;

    /* TWI0 slave */
    TWI0.SADDR = I2C_ADDR << 1;
    TWI0.SCTRLA = TWI_DIEN_bm | TWI_APIEN_bm | TWI_PIEN_bm | TWI_ENABLE_bm;

    /* WDT 1 s */
    CCP = CCP_IOREG_gc;
    WDT.CTRLA = WDT_PERIOD_1KCLK_gc;

    regs[REG_FWVER] = FW_VERSION;
    sei();
}

static void apply_outputs(const lpd_t *s) {
    /* FE_EN: open-drain emulation — output-low drives the LM74800 EN below
     * V(ENF)=0.67 V (2.87 uA shutdown); input mode releases to the divider.
     * HARD RULE (review finding): NEVER drive PB4 push-pull HIGH. 3.3 V on the
     * EN tap back-feeds the OV tap through R2 (2.45 V > 1.231 V OV threshold)
     * and actively turns the power path OFF. Legal states: OUT-LOW or INPUT. */
    if (s->fe_en)
        PORTB.DIRCLR = (1 << PB_FE_EN);
    else
        PORTB.DIRSET = (1 << PB_FE_EN); /* OUT already 0 */

    /* LOW_BATT: assert = drive low */
    if (s->low_batt)
        PORTC.DIRSET = (1 << PC_LOW_BATT);
    else
        PORTC.DIRCLR = (1 << PC_LOW_BATT);

    /* AUX (motor contactor): state gate AND Pi request */
    bool aux = s->aux_ctl && (regs[REG_CMD_AUX] & 1);
    if (aux)
        PORTB.OUTSET = (1 << PB_AUX_CTL);
    else
        PORTB.OUTCLR = (1 << PB_AUX_CTL);
}

/* USB port power-cycle: EN low for ~500 ms, handled as a mini state machine */
static uint8_t usb_cycle_ticks[2];
static void service_usb_cycle(void) {
    uint8_t cmd = regs[REG_CMD_USB];
    for (uint8_t p = 0; p < 2; p++) {
        uint8_t bit = (p == 0) ? (1 << PA_USB_EN1) : (1 << PA_USB_EN2);
        if ((cmd & (1 << p)) && usb_cycle_ticks[p] == 0)
            usb_cycle_ticks[p] = 6; /* 6 ticks x ~94 ms ~= 560 ms */
        if (usb_cycle_ticks[p] > 0) {
            PORTA.OUTCLR = bit;
            if (--usb_cycle_ticks[p] == 0) {
                PORTA.OUTSET = bit;
                regs[REG_CMD_USB] &= (uint8_t)~(1 << p);
            }
        }
    }
}

int main(void) {
    hw_init();
    lpd_t lpd;
    lpd_init(&lpd);
    uint8_t last_ticks = 0;
    uint8_t led_phase = 0;

    for (;;) {
        /* wait for ~3 PIT ticks ~= 94 ms */
        while ((uint8_t)(rtc_ticks - last_ticks) < 3) {
            set_sleep_mode(SLEEP_MODE_IDLE); /* TWI must stay clocked */
            sleep_mode();
        }
        last_ticks = rtc_ticks;
        now_ms += 94;

        /* sample */
        uint16_t vin_mv = VIN_MV_FROM_ACC(adc_read_acc16(ADC_MUXPOS_AIN4_gc));
        uint16_t v5a_mv = V5A_MV_FROM_ACC(adc_read_acc16(ADC_MUXPOS_AIN6_gc));
        uint16_t ntc_raw = adc_read_acc16(ADC_MUXPOS_AIN7_gc) >> 6; /* 8-bit-ish */

        lpd_inputs_t in = {
            .vin_mv = vin_mv,
            .switch_on = !(PORTB.IN & (1 << PB_SW_SENSE)), /* low = ON */
            .shdn_ack = !(PORTC.IN & (1 << PC_SHDN_ACK)) ||
                        (regs[REG_CMD_SHDN] == 0xA5),
            .pgood_a = (PORTC.IN & (1 << PC_PGOOD_A)) != 0,
            .pgood_b = (PORTC.IN & (1 << PC_PGOOD_B)) != 0,
        };
        lpd_step(&lpd, &in, now_ms);
        apply_outputs(&lpd);
        service_usb_cycle();

        /* publish registers (byte writes are atomic on AVR) */
        regs[REG_STATE] = (uint8_t)lpd.state;
        regs[REG_VIN_L] = (uint8_t)(vin_mv & 0xFF);
        regs[REG_VIN_H] = (uint8_t)(vin_mv >> 8);
        regs[REG_V5A_L] = (uint8_t)(v5a_mv & 0xFF);
        regs[REG_V5A_H] = (uint8_t)(v5a_mv >> 8);
        regs[REG_TEMP] = (uint8_t)ntc_raw;
        regs[REG_FLAGS] =
            (in.pgood_a << FLG_PGOOD_A) | (in.pgood_b << FLG_PGOOD_B) |
            ((!(PORTA.IN & (1 << PA_FAULT_USB))) << FLG_FAULT_USB) |
            (in.switch_on << FLG_SW) | (lpd.low_batt << FLG_LOW_BATT) |
            (((PORTB.OUT >> PB_AUX_CTL) & 1) << FLG_AUX);
        if (lpd.state == LPD_OFF || lpd.state == LPD_CUT)
            regs[REG_CMD_SHDN] = 0; /* consume ack once acted upon */

        /* LED: heartbeat pattern = state number blinks */
        led_phase = (uint8_t)((led_phase + 1) & 0x1F);
        if ((led_phase >> 2) <= (uint8_t)lpd.state && (led_phase & 3) == 0)
            PORTA.OUTTGL = (1 << PA_LED);

        wdt_reset();
    }
}
