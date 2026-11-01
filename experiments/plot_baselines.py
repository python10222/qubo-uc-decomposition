"""Plot the certified CSV without rerunning any solves."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT/'results/baselines.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True,
                             layout='constrained')
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#6A51A3']
    for ax, size in zip(axes, (10, 20)):
        for index, color in zip(range(1, 6), colors):
            name = f'{size}_0_{index}_w'
            group = sorted((r for r in rows if r['instance'] == name), key=lambda r: int(r['B']))
            if [int(r['B']) for r in group] != [2, 3, 4, 5, 6] or any(r['disc_status'] != 'optimal' for r in group):
                raise ValueError(f'Incomplete certified data: {name}')
            ax.plot([int(r['B']) for r in group], [float(r['G_disc_pct']) for r in group],
                    marker='o', markersize=4, linewidth=1.7, color=color, label=name)
        ax.set(title=f'{size}-unit instances', xlabel='Bit depth B', xticks=range(2, 7))
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
        ax.grid(axis='y', alpha=.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel(r'Discretization gap $G_{disc}$ (% of $z_{cont}$)')
    fig.suptitle('Discretization gap with full-range output grids', fontsize=13)
    output = ROOT/'reports/figures/gdisc_vs_B.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
