/* Host unit tests for the pure LPD state machine (make test).
 * Simulated at 100 ms tick, same cadence as the AVR shell. */
#include <assert.h>
#include <stdio.h>

#include "lpd_core.h"

#define TICK_MS 100u

static lpd_t s;
static lpd_inputs_t in;
static uint32_t now;

static void run_ms(uint32_t ms) {
    for (uint32_t t = 0; t < ms; t += TICK_MS) {
        now += TICK_MS;
        lpd_step(&s, &in, now);
    }
}

static void reset(uint16_t mv) {
    lpd_init(&s);
    now = 1000;
    in = (lpd_inputs_t){.vin_mv = mv, .switch_on = false,
                        .shdn_ack = false, .pgood_a = false, .pgood_b = false};
}

int main(void) {
    /* 1. cold start: switch on with a healthy pack -> PRECHARGE -> ON */
    reset(12300);
    run_ms(500);
    assert(s.state == LPD_OFF && !s.fe_en);
    in.switch_on = true;
    run_ms(TICK_MS);
    assert(s.state == LPD_PRECHARGE && s.fe_en && !s.aux_ctl);
    in.pgood_a = in.pgood_b = true;
    run_ms(TICK_MS);
    assert(s.state == LPD_ON && s.aux_ctl && !s.low_batt);

    /* 2. motor-stall sag immunity: 8 s dip below cut must NOT trip anything */
    in.vin_mv = 9800;
    run_ms(8000);
    assert(s.state == LPD_ON);
    in.vin_mv = 12000;
    run_ms(2000);
    assert(s.state == LPD_ON);

    /* 3. sustained decline: WARN after 10 s under warn threshold */
    in.vin_mv = 10400;
    run_ms(LPD_QUALIFY_MS - TICK_MS);
    assert(s.state == LPD_ON);
    run_ms(2 * TICK_MS);
    assert(s.state == LPD_WARN && s.low_batt && !s.aux_ctl);

    /* 4. WARN -> DYING needs 10 s below cut; brief sag doesn't count */
    in.vin_mv = 10100;
    run_ms(5000);
    in.vin_mv = 10400; /* bounce back above cut */
    run_ms(1000);
    in.vin_mv = 10100;
    run_ms(LPD_QUALIFY_MS - TICK_MS);
    assert(s.state == LPD_WARN); /* timer restarted at the bounce */
    run_ms(2 * TICK_MS);
    assert(s.state == LPD_DYING && s.low_batt && s.fe_en);

    /* 5. DYING: cut after 60 s (no ack) */
    run_ms(LPD_DYING_MS - TICK_MS);
    assert(s.state == LPD_DYING);
    run_ms(2 * TICK_MS);
    assert(s.state == LPD_CUT && !s.fe_en && !s.low_batt);

    /* 6. CUT latches: 11.0 V is NOT enough to reconnect (hysteresis) */
    in.vin_mv = 11000;
    run_ms(60000);
    assert(s.state == LPD_CUT);
    /* 11.7 V sustained 10 s -> PRECHARGE (switch still on) */
    in.vin_mv = 11750;
    in.pgood_a = in.pgood_b = false;
    run_ms(LPD_QUALIFY_MS + 2 * TICK_MS);
    assert(s.state == LPD_PRECHARGE);
    in.pgood_a = in.pgood_b = true;
    run_ms(TICK_MS);
    assert(s.state == LPD_ON);

    /* 7. Pi ack shortens DYING */
    in.vin_mv = 10400;
    run_ms(LPD_QUALIFY_MS + 2 * TICK_MS); /* -> WARN */
    in.vin_mv = 10100;
    run_ms(LPD_QUALIFY_MS + 2 * TICK_MS); /* -> DYING */
    assert(s.state == LPD_DYING);
    in.shdn_ack = true;
    run_ms(TICK_MS);
    assert(s.state == LPD_CUT);
    in.shdn_ack = false;

    /* 8. graceful switch-off from ON: LOW_BATT as shutdown request, 5 s budget */
    in.vin_mv = 12300;
    run_ms(LPD_QUALIFY_MS + 2 * TICK_MS); /* CUT -> PRECHARGE */
    run_ms(TICK_MS);                      /* -> ON (pgoods held true) */
    assert(s.state == LPD_ON);
    in.switch_on = false;
    run_ms(TICK_MS);
    assert(s.state == LPD_SWOFF && s.low_batt && s.fe_en);
    run_ms(LPD_SWITCH_OFF_MS + TICK_MS);
    assert(s.state == LPD_OFF && !s.fe_en);

    /* 9. WARN recovers to ON if voltage comes back (charger plugged, load shed) */
    in.switch_on = true;
    run_ms(2 * TICK_MS); /* PRECHARGE -> ON */
    assert(s.state == LPD_ON);
    in.vin_mv = 10400;
    run_ms(LPD_QUALIFY_MS + 2 * TICK_MS);
    assert(s.state == LPD_WARN);
    in.vin_mv = 11500; /* above warn + 200 mV */
    run_ms(LPD_QUALIFY_MS + 2 * TICK_MS);
    assert(s.state == LPD_ON && !s.low_batt && s.aux_ctl);

    /* 10. PRECHARGE gives up if rails never rise -> latched FAULT (no chatter);
     * a switch toggle is required to retry */
    reset(12300);
    in.switch_on = true;
    run_ms(TICK_MS);
    assert(s.state == LPD_PRECHARGE);
    run_ms(4 * LPD_PRECHARGE_MS + 2 * TICK_MS);
    assert(s.state == LPD_FAULT && !s.fe_en);
    run_ms(60000); /* switch still on: stays latched */
    assert(s.state == LPD_FAULT);
    in.switch_on = false;
    run_ms(TICK_MS);
    assert(s.state == LPD_OFF);
    in.switch_on = true;
    run_ms(TICK_MS);
    assert(s.state == LPD_PRECHARGE); /* retry allowed after toggle */

    /* 11. cold turn-on floor (review fix): a mid-charge pack (10.8 V) must be
     * switchable ON from OFF; below the floor (10.4 V) it must stay OFF */
    reset(10400);
    in.switch_on = true;
    run_ms(1000);
    assert(s.state == LPD_OFF);
    in.vin_mv = 10800; /* above LPD_COLDON_MV = 10650 */
    run_ms(TICK_MS);
    assert(s.state == LPD_PRECHARGE);
    in.pgood_a = in.pgood_b = true;
    run_ms(TICK_MS);
    assert(s.state == LPD_ON);

    printf("lpd_core: all 11 scenarios passed\n");
    return 0;
}
