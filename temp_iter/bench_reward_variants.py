"""Quick what-if analysis on the baseline gait_eval JSON.

Reconstructs traction_reward_scale and target_progress_scale per step, then
recomputes total episode reward under several reward-design variants WITHOUT
retraining, to estimate which knobs would unblock the gradient.
"""
import json
import sys
import numpy as np

JSON_PATH = "visualizations/debug_gait_eval_baseline_iter0.json"

CONTACT_QUALITY_SCALE_FLOOR = 0.25
FOOT_SLIP_SPEED_THRESHOLD = 0.75
FOOT_SLIP_SPEED_TOLERANCE = 0.20


def reconstruct_traction_scale(slip_speed_max):
    excess = max(slip_speed_max - FOOT_SLIP_SPEED_THRESHOLD - FOOT_SLIP_SPEED_TOLERANCE, 0.0)
    ratio = min(excess / max(FOOT_SLIP_SPEED_THRESHOLD, 1e-9), 1.0)
    return CONTACT_QUALITY_SCALE_FLOOR + (1.0 - CONTACT_QUALITY_SCALE_FLOOR) * (1.0 - ratio)


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    ts = data["timeseries"]
    print(f"Loaded {len(ts)} steps from {JSON_PATH}")

    by_ep = {}
    for s in ts:
        by_ep.setdefault(s["episode"], []).append(s)

    variants = {
        "baseline": [],
        "no_traction": [],
        "no_target_progress": [],
        "no_both_scales": [],
        "boost_speed_x3": [],
        "all_off_plus_boost": [],
    }
    forward_speeds = []
    progress_per_ep = []

    for ep_idx, steps in by_ep.items():
        sums = {k: 0.0 for k in variants}
        max_progress = 0.0
        for s in steps:
            r_loc = s["locomotion_reward"]
            r_fwd = s["forward_speed_reward"]
            r_diag = s["diagonal_gait_reward"]
            r_swing = s["swing_clearance_reward"]
            r_other = s["reward"] - (r_loc + r_fwd + r_diag + r_swing)

            traction = reconstruct_traction_scale(s["foot_slip_speed_max"])
            tprog = s["target_progress_scale"]

            # baseline
            sums["baseline"] += s["reward"]

            # remove traction scale from positive locomotion + forward + diag + swing
            # these were multiplied by traction; divide it out
            scale_back = 1.0 / max(traction, 1e-9)
            r_loc_no_tr = r_loc * scale_back if r_loc > 0 else r_loc
            r_fwd_no_tr = r_fwd * scale_back
            r_diag_no_tr = r_diag * scale_back
            r_swing_no_tr = r_swing * scale_back
            sums["no_traction"] += r_loc_no_tr + r_fwd_no_tr + r_diag_no_tr + r_swing_no_tr + r_other

            # remove target_progress_scale from positive locomotion only
            scale_back_tp = 1.0 / max(tprog, 1e-9)
            r_loc_no_tp = r_loc * scale_back_tp if r_loc > 0 else r_loc
            sums["no_target_progress"] += r_loc_no_tp + r_fwd + r_diag + r_swing + r_other

            # both off
            r_loc_no_both = r_loc * scale_back * scale_back_tp if r_loc > 0 else r_loc
            sums["no_both_scales"] += r_loc_no_both + r_fwd_no_tr + r_diag_no_tr + r_swing_no_tr + r_other

            # boost forward_speed_reward x3 (keeps existing scales)
            sums["boost_speed_x3"] += r_loc + r_fwd * 3 + r_diag + r_swing + r_other

            # all-off + boost x3
            sums["all_off_plus_boost"] += r_loc_no_both + r_fwd_no_tr * 3 + r_diag_no_tr + r_swing_no_tr + r_other

            forward_speeds.append(s["forward_speed_m_s"])
            max_progress = max(max_progress, s["forward_progress_m"])

        for k in variants:
            variants[k].append(sums[k])
        progress_per_ep.append(max_progress)

    print()
    print(f"{'variant':30s} {'mean_reward':>14s} {'vs_baseline':>14s}")
    base = np.mean(variants["baseline"])
    for name, sums in variants.items():
        m = np.mean(sums)
        delta = (m - base) / abs(base) * 100 if base else 0
        print(f"{name:30s} {m:14.3f} {delta:+13.1f}%")

    print()
    print(f"Forward speed (m/s): mean={np.mean(forward_speeds):.4f} p50={np.percentile(forward_speeds,50):.4f} p95={np.percentile(forward_speeds,95):.4f}")
    print(f"Forward progress per ep (m): mean={np.mean(progress_per_ep):.4f}")

    # Reward landscape diagnostic
    print()
    tractions = [reconstruct_traction_scale(s["foot_slip_speed_max"]) for s in ts]
    tprogs = [s["target_progress_scale"] for s in ts]
    print(f"traction_reward_scale: mean={np.mean(tractions):.3f} p10={np.percentile(tractions,10):.3f} p90={np.percentile(tractions,90):.3f}")
    print(f"target_progress_scale: mean={np.mean(tprogs):.3f} p10={np.percentile(tprogs,10):.3f} p90={np.percentile(tprogs,90):.3f}")
    print(f"product of both:         mean={np.mean([a*b for a,b in zip(tractions,tprogs)]):.3f}")

    # Is the agent slipping or not? Compare contact slip speeds
    slips = [s["foot_slip_speed_max"] for s in ts]
    print(f"foot_slip_speed_max:   mean={np.mean(slips):.3f} p50={np.percentile(slips,50):.3f} p95={np.percentile(slips,95):.3f}")


if __name__ == "__main__":
    main()
