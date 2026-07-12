#!/usr/bin/env python3
"""
Generate `pipeline.pdf` — a reviewed, verified summary of the full SPF processing
pipeline: collection -> segmentation -> dataset -> targets -> model -> training ->
inference -> filters, with per-stage parameters, contracts, issues, and test coverage.

Content source: claude_docs/ (adversarially verified docs + KNOWN_ISSUES + the
stage-1 audit), synthesized in claude_docs/reference/pipeline_report_content.md.
Run:  /home/mouse9911/virtual-envs/spf/bin/python tutorials/build_pipeline_report.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_signal_path_tutorial import ACC, INK, MUT, Doc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


class RDoc(Doc):
    def _footer(self):
        self.fig.text(0.5, 0.032,
                      f"SPF · Processing Pipeline Review · page {self.pageno}",
                      ha="center", fontsize=8, color=MUT)


def diag_pipeline(ax):
    ax.axis("off")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 6.4)
    rows = [
        [("1 collection\nGRBL / MAVLink\nraw IQ + positions", "#eef4fb"),
         ("2 segmentation\nv3.7 windows\nbeamform + stats", "#dcebfb"),
         ("3 dataset\nv5spfdataset\nframes + flip aug", "#dcebfb"),
         ("4 targets\nGaussian +\nsymmetry folds", "#e8f0e8")],
        [("5 model\n68ch conv -> 12\n18 -> FFNN -> P(theta)", "#e8f6ec"),
         ("6 training\nAdamW + StepLR\nmse single_loss", "#e8f6ec"),
         ("7 inference\ncache + empirical\nP(theta|phi)", "#fdf3e3"),
         ("8 filters\nEKF / PF\ntheta(t) tracks", "#fdf3e3")],
    ]
    for r, row in enumerate(rows):
        y0 = 3.7 - r * 2.9
        x = 0.25
        w = 2.35
        for i, (t, c) in enumerate(row):
            ax.add_patch(Rectangle((x, y0), w, 1.9, facecolor=c,
                                   edgecolor="#456", lw=0.9))
            ax.text(x + w / 2, y0 + 0.95, t, ha="center", va="center", fontsize=7.6)
            if i < 3:
                ax.annotate("", xy=(x + w + 0.15, y0 + 0.95),
                            xytext=(x + w - 0.02, y0 + 0.95),
                            arrowprops=dict(arrowstyle="-|>", color="#456"))
            x += w + 0.18
    # wrap arrow row1 -> row2
    ax.annotate("", xy=(1.42, 3.4 - 0.5 - 0.32), xytext=(9.85, 3.75),
                arrowprops=dict(arrowstyle="-|>", color="#456", lw=1,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_title("The SPF processing pipeline (stages reviewed in this report)",
                 fontsize=10)


def diag_shapes(ax):
    ax.axis("off")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 3)
    boxes = [
        ("raw IQ\n(2, 524288)\ncomplex64", "#eef4fb"),
        ("windowed bf\n(256, 65) f16\n+ stats (3,256)", "#dcebfb"),
        ("batch (collate)\n(512, 1, ...)\nfloat32", "#dcebfb"),
        ("model out\n(512, 1, 65)\nL1-normed", "#e8f6ec"),
        ("target\n(512, 1, 65)\nsum = 1", "#e8f0e8"),
    ]
    x, w = 0.2, 1.85
    for i, (t, c) in enumerate(boxes):
        ax.add_patch(Rectangle((x, 0.8), w, 1.5, facecolor=c, edgecolor="#456",
                               lw=0.8))
        ax.text(x + w / 2, 1.55, t, ha="center", va="center", fontsize=7.6)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + 0.14, 1.55), xytext=(x + w - 0.02, 1.55),
                        arrowprops=dict(arrowstyle="-|>", color="#456"))
        x += w + 0.17
    ax.set_title("Shape flow for the stage-1 training path (batch 256 x 2 radios "
                 "= 512 rows)", fontsize=9.5)


def kv(d, a, b):
    d.p(a + r"$\;\;\rightarrow\;\;$" + b, gap=0.004, indent=0.01, size=9.5)


def build(path):
    d = RDoc(path)
    f = d.fig
    # ---------------- title ----------------
    f.text(0.5, 0.82, "The SPF Processing Pipeline", ha="center", fontsize=25,
           fontweight="bold", color=ACC)
    f.text(0.5, 0.765, "A Reviewed, Verified End-to-End Summary", ha="center",
           fontsize=14, color=INK)
    f.text(0.5, 0.725, "collection  ·  segmentation  ·  dataset  ·  targets  ·  "
           "model  ·  training  ·  inference  ·  filters",
           ha="center", fontsize=9.5, color=MUT)
    ax = f.add_axes([0.10, 0.40, 0.80, 0.26])
    diag_pipeline(ax)
    f.text(0.5, 0.30, "Every claim in this report traces to the adversarially "
           "verified documentation in claude_docs/", ha="center", fontsize=9.5)
    f.text(0.5, 0.272, "(inventories, deep dives, KNOWN_ISSUES, and the stage-1 "
           "audit), spot-checked against source at commit 574c454.",
           ha="center", fontsize=9.5)
    f.text(0.5, 0.20, "Generated from tutorials/build_pipeline_report.py",
           ha="center", fontsize=9, color=MUT)
    d._new_page()

    # ---------------- exec summary ----------------
    d.h1("1  Executive summary")
    d.p(r"SPF localizes RF emitters by measuring the inter-antenna phase difference "
        r"$\phi=2\pi(d/\lambda)\sin\theta$ at two 2-element receivers and carrying "
        r"that phase through eight stages: physical capture, windowed beamforming, "
        r"a dataset layer with symmetry augmentation, a small neural model that "
        r"outputs a 65-bin distribution over angle, and time-fusion filters.")
    d.p(r"VERDICT: the pipeline is internally consistent and correct on its active "
        r"paths. The stage-1 training path was audited end-to-end (all shape "
        r"contracts verified; reflections proven correct; both scheduler/key hazards "
        r"inapplicable) and the latest models reproduce their predecessors exactly "
        r"(single val 0.0984 vs 0.0984). The main risks are NOT math errors -- they "
        r"are silently-wrong side paths (a cache-key collision, a segmentation "
        r"filter that drops abutting signal), rover-safety gaps, and thin direct "
        r"test coverage on the most load-bearing numerics.")
    d.h2("Top cross-stage findings")
    d.bullet(r"The production model NEVER sees the segmentation mask: it consumes "
             r"the raw 256-window beamformer. So the headline segmentation bug "
             r"(#45, dropped abutting signal) skips the NN path -- but lands on "
             r"$\mathtt{mean\_phase}$, exactly what the empirical P($\theta$|$\phi$) "
             r"table and all non-NN filters key on.")
    d.bullet(r"One sign convention is load-bearing across three stages: the dataset's "
             r"$-\sin$ target, $\mathtt{get\_phase\_diff}$ returning $-\phi$, and "
             r"every filter negating antenna spacing must agree; none of the three "
             r"is pinned by a test.")
    d.bullet(r"d/$\lambda$ spans 0.122--1.549 in production: a large fraction of the "
             r"fleet operates ABOVE the 0.5 unambiguous limit, so $\phi$ aliases "
             r"multiple $\theta$. The learned 65-bin distribution + device/spacing "
             r"embeddings quietly do ambiguity resolution the analytic arcsin "
             r"cannot -- likely part of the +36% BladeRF OOD gap.")
    d.diagram(diag_shapes, h=0.155, w=0.92)

    # ---------------- key numbers ----------------
    d.h1("2  Key numbers (verified)")
    for a, b in [
        (r"carrier $f_c$ ($\mathtt{rx\_lo}$)",
         r"868/915 MHz, 2.412--2.467 GHz, 5.766--5.866 GHz"),
        (r"antenna spacing $d$", r"0.025--0.080 m (rover prod 50.75 mm)"),
        (r"$d/\lambda$ across datasets", r"$0.122-1.549$ ($>0.5$: aliased regime)"),
        (r"sample rate / RF bandwidth", r"30 MHz / 3 MHz (prod)"),
        (r"buffer per snapshot", r"524,288 samples ($2^{19}$) $\times$ 2 elements, complex64"),
        (r"window size / stride / windows", r"2048 / 2048 / 256 per snapshot"),
        (r"angle grid $n_\theta$", r"65 bins on $[-\pi,\pi]$"),
        (r"segmentation version", r"3.7 (trim 20%, stddev $\leq$ 0.5, min seg 3000, min $|$sig$|$ 40)"),
        (r"batch", r"256 sessions $\rightarrow$ 512 collated rows ($\times$2 radios)"),
        (r"optimizer", r"AdamW; single lr 1e-4 wd 1e-3; paired lr 2e-4 wd 0"),
        (r"scheduler", r"StepLR $\gamma$ 0.5, step 6 (single) / 12 (paired), per epoch"),
        (r"target", r"Gaussian $\sigma$ 0.1 rad, $\pm\pi$ wrap-fold + front/back fold"),
        (r"model size", r"conv 68ch$\rightarrow$12 + FFNN 18$\rightarrow$512 depth 4 (thin)"),
        (r"jun26 single (final)", r"val/single 0.0984 @ 1.875M steps (== dec15)"),
        (r"jun26 paired (running)", r"val/paired 0.1517 @ 1.05M vs dec15 0.1615 @ 2.75M"),
        (r"BladeRF OOD gap", r"+36% (0.1323 vs 0.0971 mean per-dataset val loss)"),
        (r"uniform-prediction floor", r"val/uniform $\approx$ 0.129 (context for 0.0984)"),
    ]:
        kv(d, a, b)

    # ---------------- stage 1 ----------------
    d.h1("3  Stage 1 -- Data collection (raw IQ -> .zarr)")
    d.p(r"Two platform families move radios through space while recording raw IQ + "
        r"ground-truth positions: the WALL ARRAY (CNC gantry, GRBL steppers, XY in "
        r"mm, v5 format) and the ROVER (ArduPilot over MAVLink, GPS, v4 format). "
        r"Both drive a motion routine (bounce/circle/center/diamond) and record two "
        r"2-element receivers plus an emitter (ESP32 WiFi / LoRa / VTX / SDR-TX). "
        r"Per-receiver threads read the SDR, tag position+time, and write the zarr; "
        r"the AD9361 RX1-RX2 phase-inversion register calibration is load-bearing "
        r"for coherence.")
    kv(d, "output", r"$\mathtt{signal\_matrix}$ (snapshots, 2, 524288) c64 + per-snapshot metadata")
    kv(d, "v5 adds", r"tx/rx positions (mm) + $\mathtt{rx\_heading\_in\_pis}$; v4 instead GPS + heading (degrees)")
    kv(d, "collection size", r"600,000 records per receiver (rover prod)")
    d.note(r"Issues (verified): #42 GRBL out-of-bounds exception kills the motion "
           r"thread while collection keeps stamping the FROZEN position (silently "
           r"wrong data); rover safety cluster #40 (EKF arm gate omits HORIZ_ABS), "
           r"#43 (unmapped mode KeyError strands planner), #44 (RTL no timeout), "
           r"#18 (sudo shutdown from handler thread). Real serial/MAVLink hot paths "
           r"are untested (fake-hardware tests only).")

    # ---------------- stage 2 ----------------
    d.h1("4  Stage 2 -- Segmentation (v3.7: .zarr -> .yarr)")
    d.p(r"Decides which 2048-sample windows contain signal vs noise and precomputes "
        r"everything training needs, so the NN never touches raw IQ. Per session: "
        r"detrend $\rightarrow$ per-window delay-and-sum beamform against "
        r"$e^{-j2\pi(d/\lambda)\sin\theta}$ steering $\rightarrow$ trimmed circular "
        r"mean/stddev of $\phi$ + median $|$signal$|$ $\rightarrow$ "
        r"threshold/merge/drop $\rightarrow$ boolean mask + weighted session stats.")
    kv(d, "outputs (per receiver, f16 cache)",
       r"$\mathtt{windowed\_beamformer}$ (snap, 256, 65) · $\mathtt{all\_windows\_stats}$ (snap, 3, 256)")
    kv(d, "plus (f32)", r"weighted beamformer/stats, bool mask, $\mathtt{mean\_phase}$ (NaN if no signal)")
    kv(d, "correctness note", r"several properties silently require stride == window_size (true in prod)")
    d.note(r"Issues: #45 keep_signal_surrounded_by_noise drops BOTH of two abutting "
           r"signal runs the merge declined to join -- fires on the default config; "
           r"hits mean_phase (empirical table + filters), not the NN input. #37 the "
           r"live ceil-rank percentile is untested while the tested variant has no "
           r"callers. Coverage: the windowed-beamformer writer and the live "
           r"circular-stats engine have NO direct tests (audit P0 gaps).")

    # ---------------- stage 3 ----------------
    d.h1("5  Stage 3 -- Dataset layer (v5spfdataset)")
    d.p(r"Reads zarr + yarr + yaml, computes ground-truth angles from positions, and "
        r"serves torch tensors. Three angle frames: $\mathtt{y\_rad}$ = "
        r"array-relative, $\mathtt{craft\_y\_rad}$ = craft-heading-relative, "
        r"$\mathtt{absolute\_theta}$ = world. The $\phi$ target "
        r"$\mathtt{y\_phi}$ carries the $-\sin$ sign convention that matches "
        r"$\mathtt{get\_phase\_diff}$. Augmentation in render_session: FLIP (50%: "
        r"negate $\theta$/$\phi$, flip the beamformer $\theta$-axis + stats ch0 -- "
        r"verified consistent across every key); DOUBLE_FLIP (paired stage: "
        r"$\theta\rightarrow\mathrm{sign}(\theta)\pi-\theta$).")
    kv(d, "collate", r"paired rows stacked flat: B = 256 $\times$ 2 = 512; f16 upcast to f32")
    kv(d, "stage-1 batch keys",
       r"windowed_beamformer, all_windows_stats, gains, spacing, freq, vehicle, sdr, targets")
    kv(d, "quirk", r"$\mathtt{empirical}$ is force-appended despite empirical_input false (shape/device only)")
    d.note(r"Issues: #7 two live breakpoint()s on the only realtime/absolute-north "
           r"path (P0); #32/#39/#33 realtime-dataset races (unlocked read vs "
           r"eviction, init race, thread leak); #11 mutable-default mutation. "
           r"Coverage: ground-truth angles tested (zero-heading only); the FLIP "
           r"augmentation has NO test (audit P0 gap).")

    # ---------------- stage 4 ----------------
    d.h1("6  Stage 4 -- Targets and symmetry")
    d.p(r"Ground-truth $\theta$ becomes a 65-bin distribution: an analytic Gaussian "
        r"in radians ($\sigma=0.1$), built on a 3-block extended grid so mass "
        r"crossing $\pm\pi$ folds to the correct bins, L1-normalized. For the "
        r"single model the target is averaged with its front/back reflection "
        r"$\theta\rightarrow\mathrm{sign}(\theta)\pi-\theta$ -- valid because "
        r"$\sin(\mathrm{sign}(\theta)\pi-\theta)=\sin\theta$ gives the identical "
        r"$\phi$, so the supervision is deliberately bimodal.")
    d.p(r"Symmetry composition (proven in the audit): left/right is handled as "
        r"INPUT augmentation (flip), front/back in the TARGET (the $\pi$-fold); "
        r"different reflections at different stages, so they compose with no "
        r"double-fold. $\mathtt{scatter\_k:21}$ in the configs is INERT on this "
        r"path (kernel blur belongs to the unused onehot path).")
    d.note(r"Coverage: NONE direct -- no assertion pins unit-sum, the wrap-fold, or "
           r"the fold symmetry (audit P0 gap; proposed tests exist in the plan).")

    # ---------------- stage 5 ----------------
    d.h1("7  Stage 5 -- Model")
    d.p(r"Stage 1, SinglePointWithBeamformer: AllWindowsStatsNet concatenates the 3 "
        r"stats channels + 65 beamformer channels $\rightarrow$ (B, 68, 256), an "
        r"8-layer Conv1d(68$\rightarrow$64) stack $\rightarrow$ 12 window features "
        r"(with train-only window augmentation: fraction 0.5, dropout 0.25, shuffle "
        r"0.15, shrink 0.5). PrepareInput appends 6 scalars (gains 2, spacing, "
        r"freq, vehicle, sdr) $\rightarrow$ 18-dim $\rightarrow$ FFNN depth-4 "
        r"hidden-512 (layer norm) $\rightarrow$ abs $\rightarrow$ L1-norm over the "
        r"angle axis $\rightarrow$ single: (B, 1, 65). Both hard shape contracts "
        r"(68-channel, 18-dim) were verified against source.")
    d.p(r"Stage 2, PairedSinglePointWithBeamformer: runs the (detached) single net "
        r"on both radios, rotates radio-0's distribution into the craft frame, "
        r"concatenates, and a fusion FFNN outputs the craft-relative paired "
        r"distribution; initialized from stage-1 best.pth (load_single).")
    kv(d, "inputs NOT used", r"raw IQ, phase stats, session-averaged beamformer, empirical (all flags false)")
    kv(d, "scale caveat", r"beamformer enters raw (no normalization) scaled by a hardcoded /500")
    d.note(r"Issues: #34 PrepareInput mutates vehicle_type batch views in place "
           r"every training step (benign single-consumer, fragile). Coverage: the "
           r"CI training fixture uses a DIFFERENT model variant (phase input, no "
           r"windows_stats_net) -- the production forward path is never exercised "
           r"by tests (the audit's keystone gap).")

    # ---------------- stage 6 ----------------
    d.h1("8  Stage 6 -- Training (and the latest runs)")
    d.p(r"Exactly ONE term is backpropagated per stage: mse_loss(single, target) "
        r"(stage 1) or the paired equivalent (stage 2) -- SSE over 65 bins between "
        r"two unit-sum distributions. Everything else logged (craft/aligned/rand/"
        r"uniform losses) is no_grad diagnostics. fp32 throughout, AMP off. "
        r"best.pth is selected on val loss; resume parses checkpoint step names.")
    kv(d, "jun26 single", r"val/single 0.0984 @ 1.875M -- exact tie with dec15 (0.0984): faithful reproduction")
    kv(d, "jun26 paired", r"val/paired 0.1517 @ 1.05M, still improving -- ahead of dec15's best-ever")
    kv(d, "dec15 paired precedent", r"overfit after ~epoch 12 (0.1548 $\rightarrow$ 0.1615); watch for the turn")
    kv(d, "OOD", r"BladeRF val +36% above non-blade -- excluded from training (noblade split)")
    d.note(r"Issues: #1 cosine-scheduler branch crashes (UnboundLocalError) but only "
           r"one abandoned config selects it; numpy RNG is NOT seeded (torch/random "
           r"only) so seed 10 does not fully pin the data path. Coverage: loss fn, "
           r"optimizer, and the production train path have no direct tests.")

    # ---------------- stage 7 ----------------
    d.h1("9  Stage 7 -- Inference and the empirical baseline")
    d.p(r"Checkpoints run over datasets into an .npz cache keyed "
        r"dataset-basename / seg-version / checkpoint-md5 / config-md5; the NN "
        r"particle filters consume this cache. The non-learned baseline is the "
        r"empirical P($\theta$|$\phi$) table: a symmetrized 2-D histogram of "
        r"(true $\theta$, observed mean_phase) per device+spacing group, "
        r"column-normalized; the producer-consumer join was verified end-to-end.")
    d.note(r"Issues: #41 the cache key OMITS the v4 flag (which changes outputs) and "
           r"keys on basename only -- colliding runs silently return wrong cached "
           r"inference (P1, silently-wrong). #47 never-invalidated .md5 sidecars "
           r"compound it for in-place checkpoint edits. #48 the empirical builder "
           r"is broken for EVEN bin counts -- and the CLI default is 50; production "
           r"artifacts dodge it with 65/7. Coverage: cache-key correctness and the "
           r"empirical builder have no tests.")

    # ---------------- stage 8 ----------------
    d.h1("10  Stage 8 -- Filters (EKF / PF)")
    d.p(r"Fuse per-snapshot angle evidence into smooth $\theta(t)$ tracks. "
        r"Observation model $\phi=2\pi(d/\lambda)\sin(\theta-\mathrm{offset})$ per "
        r"radio; every concrete filter NEGATES the spacing at construction to match "
        r"the dataset's $-\sin$ convention (load-bearing, untested). Motion: "
        r"constant velocity. Likelihoods: empirical P($\theta$|$\phi$) lookup, or "
        r"the cached NN distribution indexed by each particle's $\theta$-bin.")
    kv(d, "active classes", r"EKF single/dual $\theta$, PF single/dual $\theta$, PF single-NN (5 of 8)")
    kv(d, "broken / excluded", r"PF dual-NN (live breakpoint, #2) · XY variants commented out of prod grid")
    d.note(r"Issues: #2 the wired-in dual-radio-NN PF hits an unconditional "
           r"breakpoint() on step 0 -- under the prod multiprocessing pool the "
           r"worker hangs forever (P0, one-line fix). #3 the whole B2/DynamoDB "
           r"cloud harness is un-importable since a module move (~14 months, "
           r"2-line fix). #31 return_particles=True crashes (.copy() on a torch "
           r"tensor). Coverage: single/dual paths tested on fake data with loose "
           r"bounds; edge paths untested.")

    # ---------------- conventions ----------------
    d.h1("11  Conventions that hold the pipeline together")
    d.bullet(r"Angles: radians on $[-\pi,\pi]$, normalized by pi_norm (plain modulo "
             r"mishandles negatives). $\phi$ = observable phase difference; "
             r"$\theta$ = angle of arrival; $\phi=2\pi(d/\lambda)\sin\theta$ with "
             r"the global $-\sin$ / negated-spacing sign pact.")
    d.bullet(r"Orientations stored as multiples of $\pi$ ($\mathtt{*\_in\_pis}$); "
             r"v4 rover heading is DEGREES by design (the v4$\rightarrow$v5 "
             r"converter bridges it -- do not 'fix', it is load-bearing).")
    d.bullet(r"Positions: zarr stores mm; dataset tensors emit meters. rx_spacing "
             r"in meters; rx_wavelength_spacing = d/$\lambda$ dimensionless.")
    d.bullet(r"Vocabulary: snapshot (one capture, one receiver) $\supset$ 256 "
             r"windows of 2048 samples; session = 1 snapshot in prod; 'paired' = "
             r"both radios of one craft.")
    d.bullet(r"A 2-element array cannot separate $\theta$ from $\pi-\theta$; the "
             r"two receivers are mounted 90$^\circ$ apart precisely so pairing "
             r"resolves it.")

    # ---------------- issues + coverage ----------------
    d.h1("12  Review: top issues and test-coverage state")
    d.h2("Top issues by severity x reachability (all adversarially verified)")
    for line in [
        r"1. #2 P0 active -- live breakpoint() in the wired-in dual-radio-NN PF; prod pool worker hangs on step 0.",
        r"2. #7 P0 active -- two breakpoint()s on the only realtime/absolute-north inference path (headless rover hang).",
        r"3. #41 P1 silent -- inference cache key omits v4 + basename-only keying: wrong cached outputs on collision.",
        r"4. #45 P1 silent -- segmentation drops abutting signal runs on the default config (hits mean_phase/filters).",
        r"5. #42 P1 silent -- GRBL out-of-bounds kills motion thread; collection stamps frozen positions thereafter.",
        r"6. #40 P1 safety -- rover EKF arm gate duplicates HORIZ_REL, omits HORIZ_ABS.",
        r"7. #43/#44 P1 safety -- unmapped mode KeyError strands planner; RTL/move has no effective timeout.",
        r"8. #3 P0 dormant -- entire B2 cloud filter pipeline un-importable (module move); 2-line fix.",
        r"9. #32/#39 P1 -- realtime dataset races (KeyError under prod settings; init AttributeError).",
        r"10. #47 P2 -- stale .md5 sidecars silently reuse old inference caches (compounds #41).",
    ]:
        d.p(line, gap=0.005, size=9.3, indent=0.008)
    d.h2("Test-coverage verdict (from the stage-1 audit matrix)")
    d.p(r"All 11 stage-1 segments are CORRECT as written, but the most "
        r"load-bearing pieces have the weakest direct coverage: the flip "
        r"augmentation, the target folds, the live circular-stats engine, and the "
        r"windowed-beamformer writer have NO direct tests, and the CI training "
        r"fixture exercises a non-production model variant. A zero-prod-change "
        r"test plan exists (claude_docs/reference/_stage1_audit/TEST_PLAN.md): "
        r"keystone prod-shaped smoke test first, then flip-consistency, "
        r"target-fold symmetry, and circular-stats tests.")
    d.note(r"Where to go deeper: claude_docs/README.md is the index -- per-stage "
           r"overviews (01--05), verified per-function contracts "
           r"(reference/spf/*), the canonical issue list (KNOWN_ISSUES.md), the "
           r"stage-1 audit (reference/_stage1_audit/), and the companion physics "
           r"tutorial (tutorials/spf_signal_path.pdf).")

    d.end()
    return d.pageno


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.pdf")
    n = build(out)
    print(f"wrote {out}  ({n} pages)")
