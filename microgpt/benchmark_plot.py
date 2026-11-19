import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def archived_samples_in_us() -> dict[str, list[float]]:
    samples = {}
    for name in PLOT_NAMES:
        with (ROOT / "times" / f"{name}.csv").open(newline="") as f:
            samples[name] = [float(row["time_ms"]) * 1000 for row in csv.DictReader(f)]
    return samples


PLOT_NAMES = {"exojit": "ExoJIT", "jax": "JAX", "numpy": "NumPy", "torch": "PyTorch", "plain": "Python (Optimized)", "original": "Python (Original)"}
PLOT_BLUE = "#2459b5"
PLOT_STYLE = {"font.family": "cmr10", "font.size": 7.5, "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "svg.fonttype": "path"}


def save_figure(fig, path: Path, title: str, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=216, metadata={"Title": title, "Description": description, "Date": None})
    if path.suffix.lower() == ".svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")


def save_archived_plot(path: Path) -> None:
    from matplotlib import rcParams
    from matplotlib.figure import Figure
    from matplotlib.transforms import blended_transform_factory

    samples = archived_samples_in_us()
    centers = {name: int(statistics.mean(values)) for name, values in samples.items()}
    order = sorted(centers, key=centers.get)
    whole_loop = {"exojit", "jax"}

    rcParams.update(PLOT_STYLE | {"svg.hashsalt": "exojit-microgpt-training"})
    width, height = 360, 126
    fig = Figure(figsize=(width / 72, height / 72), facecolor="white")
    ax = fig.add_axes((96 / width, 29 / height, 175 / width, 81 / height))
    step_rows = []
    for row, name in enumerate(order):
        if name in whole_loop:
            # the archived csv repeats one whole-loop mean, not individual samples
            assert len(set(samples[name])) == 1, f"{name}: expected a repeated whole-loop mean"
            ax.scatter([centers[name]], [row], s=11, facecolors="none", edgecolors=PLOT_BLUE, linewidths=0.63, zorder=3)
        else:
            step_rows.append(row)
    ax.boxplot(
        [samples[order[row]] for row in step_rows],
        orientation="horizontal",
        manage_ticks=False,
        positions=step_rows,
        widths=0.46,
        showfliers=True,
        patch_artist=True,
        boxprops={"facecolor": "#e7eef9", "edgecolor": PLOT_BLUE, "linewidth": 0.55},
        medianprops={"color": "#173d83", "linewidth": 0.65},
        whiskerprops={"color": PLOT_BLUE, "linewidth": 0.45},
        capprops={"color": PLOT_BLUE, "linewidth": 0.45},
        flierprops={"marker": ".", "markersize": 1.6, "markerfacecolor": PLOT_BLUE, "markeredgewidth": 0, "alpha": 0.28, "zorder": 2},
    )
    ticks = [10, 100, 1000, 10_000, 100_000]
    ax.set_xscale("log")
    ax.set_xlim(ticks[0], 2 * ticks[-1])
    ax.set_xticks(ticks, [f"{tick / 1000:g} ms" if tick >= 1000 else "$" + rf"{tick:g}\,\mu\mathrm{{s}}" + "$" for tick in ticks])
    ax.set_yticks(range(len(order)), [PLOT_NAMES[name] for name in order])
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_xlabel("time per training step (log scale)", fontsize=7, labelpad=6)
    ax.tick_params(axis="both", which="both", length=0)
    ax.tick_params(axis="x", pad=3)
    ax.tick_params(axis="y", pad=5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major", color="#d9d9d9", linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    columns = [(306, "mean"), (348, "vs. original")]
    row_text = blended_transform_factory(fig.transFigure, ax.transData)
    for x, heading in columns:
        fig.text(x / width, 1 - 10.5 / height, heading, ha="right", va="center")
    for row, name in enumerate(order):
        us = centers[name]
        elapsed = "$" + (rf"{us:.0f}\,\mu\mathrm{{s}}" if us < 1000 else rf"{us / 1000:.3f}\,\mathrm{{ms}}") + "$"
        speedup = f"{centers['original'] / us:,.0f}x"
        for (x, _), label in zip(columns, (elapsed, speedup)):
            ax.text(x / width, row, label, transform=row_text, ha="right", va="center")

    description = "Four horizontal boxplots summarize individual training steps. Hollow circles mark the whole-loop means for ExoJIT and JAX. Durations are shown on a logarithmic axis. " + "; ".join(f"{PLOT_NAMES[name]}: {centers[name]} microseconds" for name in order) + f". The ExoJIT port is about {centers['original'] / centers['exojit']:,.0f} times faster than the original implementation."
    save_figure(fig, path, "MicroGPT training time", description)


def save_median_plot(medians: dict[str, float], path: Path) -> None:
    from matplotlib import rcParams
    from matplotlib.figure import Figure
    from matplotlib.transforms import blended_transform_factory

    names = PLOT_NAMES | {"plain": "Optimized Python", "original": "Original Python"}
    order = sorted(medians, key=medians.get)
    times = [medians[name] for name in order]
    lo = math.floor(math.log10(min(times)))
    if 10**lo == min(times):
        lo -= 1
    hi = max(lo + 1, math.ceil(math.log10(max(times))))
    ticks = [10**exponent for exponent in range(lo, hi + 1)]
    left = ticks[0]

    rcParams.update(PLOT_STYLE | {"svg.hashsalt": "microgpt-benchmark"})
    width, height = 360, 126
    fig = Figure(figsize=(width / 72, height / 72), facecolor="white")
    ax = fig.add_axes((96 / width, 29 / height, 201 / width, 81 / height))
    ax.set_xscale("log")
    ax.barh(range(len(order)), [time - left for time in times], left=left, height=0.48, color=PLOT_BLUE)
    ax.set_xlim(left, ticks[-1])
    ax.set_xticks(ticks, [f"{tick / 1000:g} ms" if tick >= 1000 else "$" + rf"{tick:g}\,\mu\mathrm{{s}}" + "$" for tick in ticks])
    ax.set_yticks(range(len(order)), [names[name] for name in order])
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_xlabel("median time per training step (log scale, lower is better)", fontsize=7, labelpad=6)
    ax.tick_params(axis="both", which="both", length=0, pad=4)
    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major", color="#d9d9d9", linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    row_text = blended_transform_factory(fig.transFigure, ax.transData)
    fig.text(348 / width, 1 - 10.5 / height, "median", ha="right", va="center")
    for row, us in enumerate(times):
        elapsed = "$" + (rf"{us:.0f}\,\mu\mathrm{{s}}" if us < 1000 else rf"{us / 1000:.1f}\,\mathrm{{ms}}") + "$"
        ax.text(348 / width, row, elapsed, transform=row_text, ha="right", va="center")

    save_figure(fig, path, "MicroGPT training time", "Horizontal bars show median runtime per training step on a logarithmic scale. Each bar starts at the smallest labeled tick, not zero. Shorter is faster.")


if __name__ == "__main__":
    import tyro

    tyro.cli(save_archived_plot)
