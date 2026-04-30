import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("analysis")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

CSV_GLOB = "result_output/results_*.csv"

AXIS_LEVELS = {
    "noise":   ["clean", "noise_low", "noise_med", "noise_high"],
    "drop":    ["clean", "drop_low",  "drop_med",  "drop_high"],
    "latency": ["clean", "latency_low", "latency_med", "latency_high"],
}

VARIANT_ORDER = ["BASELINE", "CERTAINTY_EQUIV", "UNCERTAINTY_AWARE"]
VARIANT_COLOR = {"BASELINE": "tab:gray", "CERTAINTY_EQUIV": "tab:blue", "UNCERTAINTY_AWARE": "tab:orange"}


def load_all():
    files = sorted(glob.glob(CSV_GLOB))
    if not files:
        raise SystemExit(f"no files matched {CSV_GLOB}")
    dfs = [pd.read_csv(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"loaded {len(files)} csvs, {len(df)} total rows")
    return df


def summarise(df):
    g = df.groupby(["variant", "scenario", "level_name"])
    agg = g.agg(
        n=("success", "size"),
        success_rate=("success", "mean"),
        pos_err_mm_mean=("final_pos_mm", "mean"),
        pos_err_mm_std=("final_pos_mm", "std"),
        theta_err_deg_mean=("final_theta_deg", "mean"),
        theta_err_deg_std=("final_theta_deg", "std"),
        path_rms_mm_mean=("path_rms_mm", "mean"),
        path_rms_mm_std=("path_rms_mm", "std"),
        violations_mean=("n_violations", "mean"),
        violations_max=("n_violations", "max"),
        duration_mean=("duration_s", "mean"),
        solver_fail_mean=("solver_fail_rate", "mean"),
    ).round(3).reset_index()
    return agg


def axis_for_level(level: str) -> str:
    if level == "clean":           return "all"
    if level.startswith("noise"):  return "noise"
    if level.startswith("drop"):   return "drop"
    if level.startswith("latency"):return "latency"
    return "unknown"


def plot_axis(agg, axis: str, scenario: str):
    levels = AXIS_LEVELS[axis]
    sub    = agg[(agg["scenario"] == scenario) & (agg["level_name"].isin(levels))].copy()
    if sub.empty:
        return
    sub["level_idx"] = sub["level_name"].map({lv: i for i, lv in enumerate(levels)})

    metrics = [
        ("success_rate",       "Success rate",         (0, 1.05)),
        ("pos_err_mm_mean",    "Final pos err [mm]",   None),
        ("theta_err_deg_mean", "Final theta err [deg]",None),
        ("path_rms_mm_mean",   "Path RMS [mm]",        None),
    ]
    if scenario == "keepout":
        metrics.append(("violations_mean", "Mean violations / ep", None))

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4))
    if n == 1: axes = [axes]
    for ax, (col, label, ylim) in zip(axes, metrics):
        for variant in VARIANT_ORDER:
            row = sub[sub["variant"] == variant].sort_values("level_idx")
            if row.empty: continue
            std_col = col.replace("_mean", "_std") if col.endswith("_mean") else None
            y = row[col].values
            err = row[std_col].values if std_col and std_col in row.columns else None
            if err is not None:
                ax.errorbar(row["level_idx"], y, yerr=err, fmt="o-", label=variant,
                            color=VARIANT_COLOR[variant], capsize=3, alpha=0.85)
            else:
                ax.plot(row["level_idx"], y, "o-", label=variant, color=VARIANT_COLOR[variant])
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, rotation=20)
        ax.set_ylabel(label)
        if ylim is not None: ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{axis.capitalize()} sweep — scenario: {scenario}")
    fig.tight_layout()
    out = OUT_DIR / "figures" / f"{axis}_{scenario}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def per_cell_table(agg, scenario: str):
    sub = agg[agg["scenario"] == scenario].copy()
    if sub.empty: return
    pivot = sub.pivot_table(
        index="level_name", columns="variant",
        values=["success_rate", "pos_err_mm_mean", "violations_mean"],
    ).round(3)
    out = OUT_DIR / f"summary_{scenario}.csv"
    pivot.to_csv(out)
    print(f"  wrote {out}")


def main():
    df  = load_all()
    agg = summarise(df)
    agg.to_csv(OUT_DIR / "summary_full.csv", index=False)
    print(f"wrote {OUT_DIR / 'summary_full.csv'}")

    print("\n=== success rates by cell ===")
    sr = agg.pivot_table(index=["scenario", "level_name"], columns="variant", values="success_rate")
    print(sr.to_string())

    print("\n=== final pos err [mm] by cell ===")
    pe = agg.pivot_table(index=["scenario", "level_name"], columns="variant", values="pos_err_mm_mean")
    print(pe.to_string())

    print("\n=== mean violations by cell (keepout only) ===")
    v = agg[agg["scenario"] == "keepout"].pivot_table(index="level_name", columns="variant", values="violations_mean")
    print(v.to_string())

    for scenario in ["free", "keepout"]:
        print(f"\n=== plots: {scenario} ===")
        for axis in ["noise", "drop", "latency"]:
            plot_axis(agg, axis, scenario)
        per_cell_table(agg, scenario)


if __name__ == "__main__":
    main()
