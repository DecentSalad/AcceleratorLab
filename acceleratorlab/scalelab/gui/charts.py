from __future__ import annotations
import matplotlib
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
                           edgecolor=BG, linewidth=0.5)
        for b, v in zip(bars, values):
            self.ax.text(b.get_x()+b.get_width()/2, v+max(values)*0.01,
                         f"{v:,.0f}", ha="center", va="bottom", color=FG, fontsize=8)
        self.ax.set_title(title, fontsize=11, pad=8)
        self.ax.set_ylabel(ylabel, fontsize=9)
        self.ax.tick_params(axis="x", rotation=15, labelsize=9)
        self.draw()

    def lines(self, series: dict, title, xlabel, ylabel) -> None:
        self.clear()
        for i, (lbl, (xs, ys)) in enumerate(series.items()):
            self.ax.plot(xs, ys, marker="o", label=lbl,
                         color=COLS[i%len(COLS)], linewidth=1.5, markersize=5)
        self.ax.set_title(title, fontsize=11, pad=8)
        self.ax.set_xlabel(xlabel, fontsize=9)
        self.ax.set_ylabel(ylabel, fontsize=9)
        if len(series) > 1:
            self.ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=FG)
        self.draw()

    def hbar(self, labels, values, title, xlabel) -> None:
        self.clear()
        y = range(len(labels))
        self.ax.barh(list(y), values, color=COLS[:len(labels)],
                     edgecolor=BG, linewidth=0.5)
        self.ax.set_yticks(list(y))
        self.ax.set_yticklabels(labels, fontsize=9)
        self.ax.set_title(title, fontsize=11, pad=8)
        self.ax.set_xlabel(xlabel, fontsize=9)
        self.draw()
