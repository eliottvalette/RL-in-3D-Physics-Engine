"""Compare two gait_eval JSON files on key metrics."""
import json
import sys
import numpy as np


def load_eval(path):
    with open(path) as f:
        return json.load(f)


def summarize(d, label):
    g = d["global_summary"]
    ts = d["timeseries"]
    speeds = [s["forward_speed_m_s"] for s in ts]
    slips = [s["foot_slip_speed_max"] for s in ts]
    progs = [ep["net_forward_progress_m"] for ep in d["episodes"]]
    # impulse balance
    fwd_pos = g["front_positive_forward_impulse_share"]
    rear_pos = g["rear_positive_forward_impulse_share"]
    return {
        "label": label,
        "forward_speed_mean": float(np.mean(speeds)),
        "forward_speed_p95": float(np.percentile(speeds, 95)),
        "net_progress_mean_m": float(np.mean(progs)),
        "slip_max_mean": float(np.mean(slips)),
        "slip_max_p95": float(np.percentile(slips, 95)),
        "front_pos_impulse_share": float(fwd_pos),
        "rear_pos_impulse_share": float(rear_pos),
        "diagonal_contact_frac": float(g["contact_pattern_mean"]["diagonal"]),
        "rear_pair_no_front_frac": float(g["contact_pattern_mean"]["rear_pair_no_front"]),
        "grounded_leg_count_mean": float(g["grounded_leg_count_mean"]),
        "action_switches_per_joint_s": float(g["action_switches_per_joint_s"]),
        "max_abs_tilt_p95_deg": float(g["max_abs_tilt_p95_deg"]),
        "done_running_count": int(g["done_reasons"].get("running", 0)),
    }


def main():
    if len(sys.argv) != 3:
        print("usage: python compare_iters.py <baseline.json> <iter.json>")
        sys.exit(1)
    base = summarize(load_eval(sys.argv[1]), "baseline")
    new = summarize(load_eval(sys.argv[2]), "iter")

    print()
    print(f"{'metric':35s} {'baseline':>14s} {'iter':>14s} {'delta':>14s}")
    print("-" * 80)
    metrics_target_higher = {
        "forward_speed_mean", "forward_speed_p95",
        "net_progress_mean_m", "front_pos_impulse_share",
        "diagonal_contact_frac", "grounded_leg_count_mean",
        "done_running_count",
    }
    metrics_target_lower = {
        "slip_max_mean", "slip_max_p95",
        "rear_pair_no_front_frac", "action_switches_per_joint_s",
        "max_abs_tilt_p95_deg",
    }
    for k in base:
        if k == "label":
            continue
        v0, v1 = base[k], new[k]
        if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            if abs(v0) > 1e-9:
                pct = (v1 - v0) / abs(v0) * 100
                marker = ""
                if k in metrics_target_higher:
                    marker = " ↑good" if pct > 0 else " ↓bad"
                elif k in metrics_target_lower:
                    marker = " ↓good" if pct < 0 else " ↑bad"
                print(f"{k:35s} {v0:14.4f} {v1:14.4f} {pct:+12.1f}%{marker}")
            else:
                delta = v1 - v0
                print(f"{k:35s} {v0:14.4f} {v1:14.4f} {delta:+14.4f}")
    print()
    target_speed = 1.0
    print(f"Speed target {target_speed} m/s — baseline {base['forward_speed_mean']:.3f}m/s ({base['forward_speed_mean']/target_speed*100:.0f}%), iter {new['forward_speed_mean']:.3f}m/s ({new['forward_speed_mean']/target_speed*100:.0f}%)")


if __name__ == "__main__":
    main()
