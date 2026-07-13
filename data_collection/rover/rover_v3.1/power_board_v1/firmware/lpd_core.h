/* lpd_core.h — pure LPD/handshake state machine (no hardware access).
 *
 * Contract (P1_DETAIL_DESIGN.md §4): CUT 10.2 V / RECONNECT 11.7 V, warn 10.55 V,
 * 10 s qualifier on every voltage-driven transition, 60 s LOW_BATT->cut handshake
 * (shortened by Pi SHDN_ACK), graceful 5 s panel-switch off.
 *
 * Pure function: lpd_step(state*, inputs, now_ms). Unit-testable on the host
 * (make test) — the AVR shell in supervisor.c only wires ADC/GPIO/I2C to it.
 */
#ifndef LPD_CORE_H
#define LPD_CORE_H

#include <stdbool.h>
#include <stdint.h>

/* thresholds (mV) — single source of truth */
#define LPD_CUT_MV 10200u
#define LPD_RECONNECT_MV 11700u
#define LPD_WARN_MV 10550u

/* timing (ms) */
#define LPD_QUALIFY_MS 10000ul   /* sag immunity: sustain before acting   */
#define LPD_DYING_MS 60000ul     /* LOW_BATT asserted -> hard cut          */
#define LPD_SWITCH_OFF_MS 5000ul /* graceful budget on panel switch off    */
#define LPD_PRECHARGE_MS 2000ul  /* front-end soft-start + rails settle    */

typedef enum {
    LPD_OFF = 0,       /* front-end disabled, waiting for switch          */
    LPD_PRECHARGE = 1, /* FE_EN released, waiting for PGOODs              */
    LPD_ON = 2,        /* normal operation                                */
    LPD_WARN = 3,      /* below warn threshold (qualified); LOW_BATT out  */
    LPD_DYING = 4,     /* below cut threshold (qualified); 60 s countdown */
    LPD_CUT = 5,       /* front-end off; recover only at RECONNECT_MV     */
    LPD_SWOFF = 6,     /* user switch-off handshake in progress           */
    LPD_FAULT = 7      /* rails failed to rise; latched until switch off  */
} lpd_state_t;

typedef struct {
    uint16_t vin_mv;
    bool switch_on;  /* debounced panel switch */
    bool shdn_ack;   /* Pi says "capture closed, halting" */
    bool pgood_a;
    bool pgood_b;
} lpd_inputs_t;

typedef struct {
    lpd_state_t state;
    uint32_t t_enter_ms;    /* when current state was entered           */
    uint32_t t_qualify_ms;  /* start of current qualification window (0 = none) */
    uint32_t t_qualify2_ms; /* second independent window (WARN's cut timer) */
    /* outputs (level-driven, idempotent) */
    bool fe_en;    /* LM74800 EN released (true = rails allowed)  */
    bool low_batt; /* LOW_BATT line to Pi asserted                */
    bool aux_ctl;  /* motor contactor allowed (follows ON-ish states) */
} lpd_t;

static inline void lpd_init(lpd_t *s) {
    s->state = LPD_OFF;
    s->t_enter_ms = 0;
    s->t_qualify_ms = 0;
    s->t_qualify2_ms = 0;
    s->fe_en = false;
    s->low_batt = false;
    s->aux_ctl = false;
}

static inline void lpd__enter(lpd_t *s, lpd_state_t st, uint32_t now) {
    s->state = st;
    s->t_enter_ms = now;
    s->t_qualify_ms = 0;
    s->t_qualify2_ms = 0;
}

/* qualification helper: returns true once `cond` has held for LPD_QUALIFY_MS */
static inline bool lpd__qualified(lpd_t *s, bool cond, uint32_t now) {
    if (!cond) {
        s->t_qualify_ms = 0;
        return false;
    }
    if (s->t_qualify_ms == 0) {
        s->t_qualify_ms = now ? now : 1;
        return false;
    }
    return (now - s->t_qualify_ms) >= LPD_QUALIFY_MS;
}

static inline void lpd_step(lpd_t *s, const lpd_inputs_t *in, uint32_t now) {
    switch (s->state) {
    case LPD_OFF:
        if (in->switch_on && in->vin_mv >= LPD_RECONNECT_MV)
            lpd__enter(s, LPD_PRECHARGE, now);
        break;

    case LPD_PRECHARGE:
        if (!in->switch_on) { lpd__enter(s, LPD_OFF, now); break; }
        if (in->pgood_a && in->pgood_b) { lpd__enter(s, LPD_ON, now); break; }
        if ((now - s->t_enter_ms) > 4 * LPD_PRECHARGE_MS)
            lpd__enter(s, LPD_FAULT, now); /* rails never came up: latch off;
                                              OFF would instantly re-arm (chatter) */
        break;

    case LPD_ON:
        if (!in->switch_on) { lpd__enter(s, LPD_SWOFF, now); break; }
        if (lpd__qualified(s, in->vin_mv < LPD_WARN_MV, now))
            lpd__enter(s, LPD_WARN, now);
        break;

    case LPD_WARN:
        if (!in->switch_on) { lpd__enter(s, LPD_SWOFF, now); break; }
        /* cut condition runs on its own timer (t_qualify2) so the recovery
         * qualifier below can't reset it every tick */
        if (in->vin_mv < LPD_CUT_MV) {
            if (s->t_qualify2_ms == 0)
                s->t_qualify2_ms = now ? now : 1;
            if ((now - s->t_qualify2_ms) >= LPD_QUALIFY_MS) {
                lpd__enter(s, LPD_DYING, now);
                break;
            }
        } else {
            s->t_qualify2_ms = 0;
        }
        /* recovery back to ON, 200 mV above warn (no flapping) */
        if (lpd__qualified(s, in->vin_mv >= LPD_WARN_MV + 200, now))
            lpd__enter(s, LPD_ON, now);
        break;

    case LPD_DYING:
        /* no return path: pack at cut threshold under load will sag-flap; commit */
        if (in->shdn_ack || (now - s->t_enter_ms) >= LPD_DYING_MS)
            lpd__enter(s, LPD_CUT, now);
        break;

    case LPD_CUT:
        if (lpd__qualified(s, in->vin_mv >= LPD_RECONNECT_MV, now))
            lpd__enter(s, in->switch_on ? LPD_PRECHARGE : LPD_OFF, now);
        break;

    case LPD_SWOFF:
        if (in->shdn_ack || (now - s->t_enter_ms) >= LPD_SWITCH_OFF_MS)
            lpd__enter(s, LPD_OFF, now);
        break;

    case LPD_FAULT:
        if (!in->switch_on) /* user must toggle the switch to retry */
            lpd__enter(s, LPD_OFF, now);
        break;
    }

    /* level outputs derived from state — one place, no drift */
    s->fe_en = (s->state == LPD_PRECHARGE || s->state == LPD_ON ||
                s->state == LPD_WARN || s->state == LPD_DYING ||
                s->state == LPD_SWOFF);
    s->low_batt = (s->state == LPD_WARN || s->state == LPD_DYING ||
                   s->state == LPD_SWOFF);
    s->aux_ctl = (s->state == LPD_ON); /* motors only in healthy state */
}

#endif /* LPD_CORE_H */
