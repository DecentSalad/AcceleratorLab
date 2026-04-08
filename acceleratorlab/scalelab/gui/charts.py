from __future__ import annotations
import matplotlib
import matplotlib.ticker
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

BG    = "#131825"
FG    = "#9aa0bc"
GRID  = "#1e2540"
COLS  = ["#5b8af5","#3ecf8e","#f5a623","#e05c5c","#a78bfa","#5cc8fa","#f59b23"]


class ChartCanvas(FigureCanvasQTAgg):
    def __init__(self, w: float = 5, h: float = 3.4) -> None:
        self.fig = Figure(figsize=(w, h), facecolor=BG, tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        self._style()
        super().__init__(self.fig)
        self.setStyleSheet("background:transparent;")

    def _style(self) -> None:
        ax = self.ax
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=9)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)

    def clear(self) -> None:
        self.ax.cla(); self._style()

    def bar(self, labels, values, title, ylabel) -> None:
        self.clear()
        bars = self.ax.bar(labels, values, color=COLS[:len(labels)],
                           edgecolor=BG, linewidth=0.5, width=0.5)
        top = max(values) if max(values) > 0 else 1
        for b, v in zip(bars, values):
            self.ax.text(b.get_x()+b.get_width()/2, v + top * 0.03,
                         f"{v:,.1f}", ha="center", va="bottom", color=FG, fontsize=9,
                         fontweight="bold")
        self.ax.set_title(title, fontsize=11, pad=10)
        self.ax.set_ylabel(ylabel, fontsize=9)
        # Always start y axis at 0; give 35% headroom so labels don't clip
        self.ax.set_ylim(bottom=0, top=top * 1.35)
        # More y-axis tick marks for readability
        self.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, min_n_ticks=4))
        self.ax.tick_params(axis="x", rotation=15, labelsize=9)
        self.ax.tick_params(axis="y", labelsize=8)
        self.draw()

    def lines(self, series: dict, title, xlabel, ylabel) -> None:
        self.clear()
        all_ys = [y for _, (_, ys) in series.items() for y in ys]
        for i, (lbl, (xs, ys)) in enumerate(series.items()):
            self.ax.plot(xs, ys, marker="o", label=lbl,
                         color=COLS[i%len(COLS)], linewidth=2, markersize=7)
        self.ax.set_title(title, fontsize=11, pad=10)
        self.ax.set_xlabel(xlabel, fontsize=9)
        self.ax.set_ylabel(ylabel, fontsize=9)
        # Start y at 0 with headroom so points don't sit on the top edge
        if all_ys:
            top = max(all_ys) if max(all_ys) > 0 else 1
            self.ax.set_ylim(bottom=0, top=top * 1.3)
        self.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, min_n_ticks=4))
        self.ax.tick_params(axis="y", labelsize=8)
        if len(series) > 1:
            self.ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=FG)
        self.draw()

    def hbar(self, labels, values, title, xlabel) -> None:
        self.clear()
        y = range(len(labels))
        self.ax.barh(list(y), values, color=COLS[:len(labels)],
                     edgecolor=BG, linewidth=0.5, height=0.5)
        # Label each bar with its value
        right = max(values) if max(values) > 0 else 1
        for i, v in enumerate(values):
            self.ax.text(v + right * 0.02, i, f"{v:,.0f}",
                         va="center", ha="left", color=FG, fontsize=8)
        self.ax.set_yticks(list(y))
        self.ax.set_yticklabels(labels, fontsize=9)
        self.ax.set_title(title, fontsize=11, pad=10)
        self.ax.set_xlabel(xlabel, fontsize=9)
        # Start x at 0 with headroom for value labels
        self.ax.set_xlim(left=0, right=right * 1.25)
        self.ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, min_n_ticks=4))
        self.ax.tick_params(axis="x", labelsize=8)
        self.draw()