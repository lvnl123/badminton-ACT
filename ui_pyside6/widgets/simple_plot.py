from __future__ import annotations

from math import ceil, sqrt, pi, exp
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtWidgets import QToolTip, QWidget
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QBrush, QPainterPath, QPolygonF

try:
    from scipy.stats import gaussian_kde
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SimpleLinePlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._xs: List[float] = []
        self._ys: List[float] = []
        self._x_label = "x"
        self._y_label = "y"
        self._title = ""
        self._highlight_x: Optional[float] = None
        self.setMinimumHeight(160)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_series(self, xs: List[float], ys: List[float], *, x_label: str = "x", y_label: str = "y", title: str = ""):
        self._xs = xs
        self._ys = ys
        self._x_label = x_label
        self._y_label = y_label
        self._title = title
        self.update()

    def set_highlight_x(self, x: Optional[float]):
        self._highlight_x = x
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        rect = self.rect()
        # No background fill

        margin = 12
        title_h = 18 if self._title else 0
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin - 18)

        if self._title:
            p.setPen(QPen(QColor("#93c5fd")))
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)

        if len(self._xs) < 2 or len(self._ys) < 2:
            p.setPen(QPen(QColor("#6b7280")))
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")
            return

        x_min = min(self._xs)
        x_max = max(self._xs)
        y_min = min(self._ys)
        y_max = max(self._ys)
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        p.setPen(QPen(QColor("#2a2f3a"), 1))
        p.drawRect(plot)

        def to_px(x: float, y: float) -> Tuple[float, float]:
            px = plot.left() + (x - x_min) / (x_max - x_min) * plot.width()
            py = plot.bottom() - (y - y_min) / (y_max - y_min) * plot.height()
            return px, py

        p.setPen(QPen(QColor("#22c55e"), 2))
        last = to_px(self._xs[0], self._ys[0])
        for i in range(1, min(len(self._xs), len(self._ys))):
            cur = to_px(self._xs[i], self._ys[i])
            p.drawLine(int(last[0]), int(last[1]), int(cur[0]), int(cur[1]))
            last = cur

        if self._highlight_x is not None:
            hx = max(x_min, min(x_max, self._highlight_x))
            px, _ = to_px(hx, y_min)
            p.setPen(QPen(QColor("#f59e0b"), 1))
            p.drawLine(int(px), plot.top(), int(px), plot.bottom())

        p.setPen(QPen(QColor("#9ca3af")))
        p.drawText(rect.adjusted(margin, rect.height() - 18, -margin, -2), Qt.AlignLeft | Qt.AlignVCenter, self._x_label)
        p.drawText(rect.adjusted(margin, rect.height() - 18, -margin, -2), Qt.AlignRight | Qt.AlignVCenter, self._y_label)


class MetricCard(QWidget):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._value = "--"
        self._subtitle = ""
        self._accent = QColor("#3b82f6")
        self.setMinimumHeight(74)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_content(self, *, title: Optional[str] = None, value: Optional[str] = None, subtitle: Optional[str] = None):
        if title is not None:
            self._title = title
        if value is not None:
            self._value = value
        if subtitle is not None:
            self._subtitle = subtitle
        self.update()

    def set_accent(self, color: QColor):
        self._accent = color
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect().adjusted(0, 0, -1, -1)
        # No background fill
        
        # Draw translucent background for readability
        p.setBrush(QColor(15, 18, 22, 180))
        p.setPen(QPen(QColor("#1f2937"), 1))
        p.drawRoundedRect(r, 12, 12)

        p.setPen(QPen(self._accent, 4))
        p.drawLine(r.left() + 12, r.top() + 14, r.left() + 12, r.bottom() - 14)

        title_font = QFont("Segoe UI", 9)
        value_font = QFont("Segoe UI", 16, QFont.Weight.DemiBold)
        sub_font = QFont("Segoe UI", 9)

        p.setFont(title_font)
        p.setPen(QPen(QColor("#93c5fd")))
        p.drawText(r.adjusted(22, 10, -10, -10), Qt.AlignLeft | Qt.AlignTop, self._title)

        p.setFont(value_font)
        p.setPen(QPen(QColor("#e5e7eb")))
        p.drawText(r.adjusted(22, 22, -10, -22), Qt.AlignLeft | Qt.AlignVCenter, self._value)

        p.setFont(sub_font)
        p.setPen(QPen(QColor("#9ca3af")))
        p.drawText(r.adjusted(22, 0, -10, 10), Qt.AlignLeft | Qt.AlignBottom, self._subtitle)


class SimpleBarChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: List[str] = []
        self._values: List[float] = []
        self._title = ""
        self.setMinimumHeight(180)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_data(self, labels: Sequence[str], values: Sequence[float], *, title: str = ""):
        self._labels = list(labels)
        self._values = [float(v) for v in values]
        self._title = title
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        # No background fill

        margin = 12
        title_h = 18 if self._title else 0
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin - 18)

        if self._title:
            p.setPen(QPen(QColor("#93c5fd")))
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)

        if not self._labels or not self._values:
            p.setPen(QPen(QColor("#6b7280")))
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")
            return

        n = min(len(self._labels), len(self._values))
        max_v = max(self._values[:n]) if n > 0 else 1.0
        if max_v <= 0:
            max_v = 1.0

        p.setPen(QPen(QColor("#1f2937"), 1))
        p.drawRoundedRect(plot.adjusted(0, 0, -1, -1), 10, 10)

        gap = 8
        bar_w = max(8, int((plot.width() - gap * (n + 1)) / max(1, n)))
        x = plot.left() + gap

        p.setFont(QFont("Segoe UI", 8))
        for i in range(n):
            v = max(0.0, float(self._values[i]))
            h = int((v / max_v) * max(1, plot.height() - 18))
            bar = (x, plot.bottom() - 18 - h, bar_w, h)
            p.fillRect(*bar, QColor("#22c55e") if i % 2 == 0 else QColor("#3b82f6"))
            p.setPen(QPen(QColor("#111827"), 1))
            p.drawRect(*bar)
            p.setPen(QPen(QColor("#9ca3af")))
            label = self._labels[i]
            p.drawText(x - 4, plot.bottom() - 16, bar_w + 8, 16, Qt.AlignHCenter | Qt.AlignVCenter, label[:6])
            x += bar_w + gap


class TimelineMarkers(QWidget):
    markerActivated = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._xs: List[float] = []
        self._frames: List[int] = []
        self._colors: List[QColor] = []
        self._selected = -1
        self._title = ""
        self.setMinimumHeight(86)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_markers(self, xs: Sequence[float], frames: Sequence[int], *, colors: Optional[Sequence[QColor]] = None, title: str = ""):
        self._xs = [float(x) for x in xs]
        self._frames = [int(f) for f in frames]
        self._colors = list(colors) if colors is not None else []
        self._title = title
        self._selected = -1
        self.update()

    def set_selected_by_frame(self, frame: int):
        if not self._frames:
            return
        try:
            idx = self._frames.index(int(frame))
        except ValueError:
            return
        self._selected = idx
        self.update()

    def mousePressEvent(self, event):
        if not self._xs or not self._frames:
            return
        rect = self.rect().adjusted(12, 24, -12, -18)
        if rect.width() <= 0:
            return
        x_min = min(self._xs)
        x_max = max(self._xs)
        if x_max <= x_min:
            x_max = x_min + 1.0
        px = float(event.position().x())
        best = -1
        best_dist = 1e18
        for i, x in enumerate(self._xs):
            t = (x - x_min) / (x_max - x_min)
            mx = rect.left() + t * rect.width()
            d = abs(mx - px)
            if d < best_dist:
                best_dist = d
                best = i
        if best >= 0:
            self._selected = best
            self.markerActivated.emit(int(self._frames[best]))
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        # No background fill

        margin = 12
        title_h = 18 if self._title else 0
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin - 8)

        if self._title:
            p.setPen(QPen(QColor("#93c5fd")))
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)

        if not self._xs or not self._frames:
            p.setPen(QPen(QColor("#6b7280")))
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无事件")
            return

        x_min = min(self._xs)
        x_max = max(self._xs)
        if x_max <= x_min:
            x_max = x_min + 1.0

        y = (plot.top() + plot.bottom()) / 2
        p.setPen(QPen(QColor("#334155"), 2))
        p.drawLine(plot.left(), int(y), plot.right(), int(y))

        for i, x in enumerate(self._xs):
            t = (x - x_min) / (x_max - x_min)
            mx = plot.left() + t * plot.width()
            color = self._colors[i] if i < len(self._colors) else QColor("#f59e0b")
            r = 6
            if i == self._selected:
                r = 9
            p.setBrush(color)
            p.setPen(QPen(QColor("#0b0f14"), 2))
            p.drawEllipse(int(mx - r), int(y - r), int(r * 2), int(r * 2))


class SimpleHeatmap(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: List[Tuple[float, float]] = []
        self._title = ""
        self._bins = (72, 40)
        self._image: Optional[QImage] = None
        self._w = 0
        self._h = 0
        self.setMinimumHeight(220)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_points(self, points: Sequence[Tuple[float, float]], *, title: str = "", canvas_size: Optional[Tuple[int, int]] = None):
        self._points = [(float(x), float(y)) for x, y in points]
        self._title = title
        if canvas_size is not None:
            self._w, self._h = int(canvas_size[0]), int(canvas_size[1])
        self._rebuild_image()
        self.update()

    def _rebuild_image(self):
        if not self._points:
            self._image = None
            return
        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        bw, bh = self._bins
        grid = [[0 for _ in range(bw)] for _ in range(bh)]
        for x, y in self._points:
            gx = int((x - x_min) / (x_max - x_min) * (bw - 1))
            gy = int((y - y_min) / (y_max - y_min) * (bh - 1))
            gx = max(0, min(bw - 1, gx))
            gy = max(0, min(bh - 1, gy))
            grid[gy][gx] += 1

        max_c = max(max(row) for row in grid) if grid else 1
        if max_c <= 0:
            max_c = 1

        img = QImage(bw, bh, QImage.Format_RGB32)
        for y in range(bh):
            for x in range(bw):
                v = grid[y][x] / max_c
                r = int(255 * v)
                g = int(80 + 80 * (1.0 - v))
                b = int(40 + 140 * (1.0 - v))
                img.setPixel(x, y, QColor(r, g, b).rgb())
        self._image = img

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        # No background fill

        margin = 12
        title_h = 18 if self._title else 0
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin)

        if self._title:
            p.setPen(QPen(QColor("#93c5fd")))
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)

        if self._image is None:
            p.setPen(QPen(QColor("#6b7280")))
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")
            return

        target = plot
        p.drawImage(target, self._image.scaled(target.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        p.setPen(QPen(QColor("#1f2937"), 1))
        p.drawRoundedRect(plot.adjusted(0, 0, -1, -1), 10, 10)


class DensityBubbleMap(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: List[Tuple[float, float]] = []
        self._title = ""
        self._bins = (44, 24)
        self._grid: List[List[int]] = []
        self._bounds: Optional[Tuple[float, float, float, float]] = None
        self._current_point: Optional[Tuple[float, float]] = None
        self._show_current_point = True
        self._last_hover_cell: Optional[Tuple[int, int]] = None
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_points(self, points: Sequence[Tuple[float, float]], *, title: str = ""):
        self._points = [(float(x), float(y)) for x, y in points]
        self._title = title
        self._rebuild_bins()
        self.update()

    def set_bins(self, bins: Tuple[int, int]):
        bw, bh = int(bins[0]), int(bins[1])
        bw = max(8, min(160, bw))
        bh = max(6, min(120, bh))
        self._bins = (bw, bh)
        self._rebuild_bins()
        self.update()

    def set_current_point(self, point: Optional[Tuple[float, float]]):
        self._current_point = (float(point[0]), float(point[1])) if point is not None else None
        self.update()

    def set_show_current_point(self, show: bool):
        self._show_current_point = bool(show)
        self.update()

    def _rebuild_bins(self):
        if not self._points:
            self._grid = []
            self._bounds = None
            return

        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0
        self._bounds = (x_min, x_max, y_min, y_max)

        bw, bh = self._bins
        grid = [[0 for _ in range(bw)] for _ in range(bh)]
        for x, y in self._points:
            gx = int((x - x_min) / (x_max - x_min) * (bw - 1))
            gy = int((y - y_min) / (y_max - y_min) * (bh - 1))
            gx = max(0, min(bw - 1, gx))
            gy = max(0, min(bh - 1, gy))
            grid[gy][gx] += 1
        self._grid = grid

    def leaveEvent(self, event):
        self._last_hover_cell = None
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        if not self._grid or self._bounds is None:
            return
        rect = self.rect()
        margin = 12
        title_h = 18 if self._title else 0
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin)
        if not plot.contains(int(event.position().x()), int(event.position().y())):
            self._last_hover_cell = None
            return

        bw, bh = self._bins
        x_step = plot.width() / max(1, bw)
        y_step = plot.height() / max(1, bh)
        gx = int((event.position().x() - plot.left()) / max(1e-6, x_step))
        gy = int((event.position().y() - plot.top()) / max(1e-6, y_step))
        gx = max(0, min(bw - 1, gx))
        gy = max(0, min(bh - 1, gy))
        cell = (gx, gy)
        if cell == self._last_hover_cell:
            return
        self._last_hover_cell = cell

        c = self._grid[gy][gx]
        max_c = max(max(row) for row in self._grid) if self._grid else 1
        if max_c <= 0:
            max_c = 1
        density = c / max_c
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"cell=({gx},{gy})\ncount={c}\ndensity={density:.2f}",
            self,
        )

    @staticmethod
    def _mix(a: QColor, b: QColor, t: float) -> QColor:
        t = max(0.0, min(1.0, float(t)))
        return QColor(
            int(a.red() + (b.red() - a.red()) * t),
            int(a.green() + (b.green() - a.green()) * t),
            int(a.blue() + (b.blue() - a.blue()) * t),
        )

    def _color_for(self, t: float) -> QColor:
        c1 = QColor("#2563eb")
        c2 = QColor("#22c55e")
        c3 = QColor("#f59e0b")
        c4 = QColor("#ef4444")
        if t < 0.35:
            return self._mix(c1, c2, t / 0.35)
        if t < 0.7:
            return self._mix(c2, c3, (t - 0.35) / 0.35)
        return self._mix(c3, c4, (t - 0.7) / 0.3)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        # No background fill

        margin = 12
        title_h = 18 if self._title else 0
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin)

        if self._title:
            p.setPen(QPen(QColor("#93c5fd")))
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)

        if not self._grid or self._bounds is None:
            p.setPen(QPen(QColor("#6b7280")))
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")
            return

        p.setPen(QPen(QColor("#1f2937"), 1))
        p.drawRoundedRect(plot.adjusted(0, 0, -1, -1), 10, 10)

        bw, bh = self._bins
        max_c = max(max(row) for row in self._grid) if self._grid else 1
        if max_c <= 0:
            max_c = 1

        x_step = plot.width() / max(1, bw)
        y_step = plot.height() / max(1, bh)

        p.setPen(QPen(QColor("#111827"), 1))
        for i in range(1, 6):
            x = plot.left() + int(i / 6 * plot.width())
            p.drawLine(x, plot.top(), x, plot.bottom())
        for i in range(1, 4):
            y = plot.top() + int(i / 4 * plot.height())
            p.drawLine(plot.left(), y, plot.right(), y)

        for gy in range(bh):
            row = self._grid[gy]
            for gx in range(bw):
                c = row[gx]
                if c <= 0:
                    continue
                t = c / max_c
                cx = plot.left() + (gx + 0.5) * x_step
                cy = plot.top() + (gy + 0.5) * y_step
                radius = 2.0 + 14.0 * sqrt(t)
                color = self._color_for(t)
                color.setAlpha(int(120 + 110 * t))
                p.setBrush(color)
                p.setPen(QPen(QColor("#0b0f14"), 1))
                p.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))

        if self._show_current_point and self._current_point is not None and self._bounds is not None:
            x_min, x_max, y_min, y_max = self._bounds
            px = (self._current_point[0] - x_min) / (x_max - x_min)
            py = (self._current_point[1] - y_min) / (y_max - y_min)
            px = max(0.0, min(1.0, px))
            py = max(0.0, min(1.0, py))
            cx = plot.left() + px * plot.width()
            cy = plot.top() + py * plot.height()
            p.setBrush(QColor(0, 0, 0, 0))
            p.setPen(QPen(QColor("#e5e7eb"), 2))
            p.drawEllipse(int(cx - 7), int(cy - 7), 14, 14)
            p.setPen(QPen(QColor("#f59e0b"), 2))
            p.drawLine(int(cx - 10), int(cy), int(cx + 10), int(cy))
            p.drawLine(int(cx), int(cy - 10), int(cx), int(cy + 10))

        legend_w = 120
        legend_h = 10
        legend = plot.adjusted(plot.width() - legend_w - 8, 8, -8, 0)
        legend.setHeight(legend_h)
        for i in range(legend_w):
            t = i / max(1, legend_w - 1)
            p.setPen(QPen(self._color_for(t), 1))
            p.drawLine(legend.left() + i, legend.top(), legend.left() + i, legend.bottom())
        p.setPen(QPen(QColor("#9ca3af")))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(legend.adjusted(0, legend_h + 2, 0, legend_h + 16), Qt.AlignLeft | Qt.AlignTop, "低")
        p.drawText(legend.adjusted(0, legend_h + 2, 0, legend_h + 16), Qt.AlignRight | Qt.AlignTop, "高")


class ProDistributionChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._values: List[float] = []
        self._title = ""
        self._x_label = ""
        self._color = QColor("#3b82f6")
        self.setMinimumHeight(220)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)

    def set_data(self, values: Sequence[float], *, title: str = "", x_label: str = "", color: str = "#3b82f6"):
        self._values = sorted([float(v) for v in values if v is not None])
        self._title = title
        self._x_label = x_label
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        margin = 12
        title_h = 18 if self._title else 0
        plot = rect.adjusted(margin, margin + title_h, -margin - 40, -margin - 20)  # Reserve right space for ECDF axis

        if self._title:
            p.setPen(QPen(QColor("#93c5fd")))
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)

        if len(self._values) < 2:
            p.setPen(QPen(QColor("#6b7280")))
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")
            return

        data = np.array(self._values)
        v_min, v_max = data[0], data[-1]
        if v_max <= v_min: v_max = v_min + 1.0
        
        # Grid
        p.setPen(QPen(QColor("#1f2937"), 1, Qt.DotLine))
        for i in range(5):
            y = plot.bottom() - i * plot.height() / 4
            p.drawLine(plot.left(), int(y), plot.right(), int(y))
        
        # 1. Histogram (Background)
        hist, bin_edges = np.histogram(data, bins=30, density=True)
        max_h = hist.max() if hist.max() > 0 else 1.0
        bin_w = plot.width() / 30
        
        p.setPen(Qt.NoPen)
        c_hist = QColor(self._color)
        c_hist.setAlpha(30)
        p.setBrush(c_hist)
        
        for i, h in enumerate(hist):
            bh = h / max_h * plot.height()
            bx = plot.left() + i * bin_w
            p.drawRect(int(bx), int(plot.bottom() - bh), int(bin_w) + 1, int(bh))

        # 2. KDE Curve
        xs = np.linspace(v_min, v_max, 100)
        ys = np.zeros_like(xs)
        if SCIPY_AVAILABLE:
            try:
                kde = gaussian_kde(data)
                ys = kde(xs)
            except: pass
        else:
             # simple smooth
             ys = np.interp(xs, (bin_edges[:-1]+bin_edges[1:])/2, hist)

        max_y = ys.max() if ys.max() > 0 else 1.0
        path = QPainterPath()
        path.moveTo(plot.left(), plot.bottom())
        for i, x in enumerate(xs):
            px = plot.left() + (x - v_min)/(v_max - v_min) * plot.width()
            py = plot.bottom() - (ys[i]/max_y) * plot.height()
            path.lineTo(px, py)
        path.lineTo(plot.right(), plot.bottom())
        path.closeSubpath()
        
        c_kde = QColor(self._color)
        c_kde.setAlpha(60)
        p.setBrush(c_kde)
        p.setPen(QPen(self._color, 2))
        p.drawPath(path)

        # 3. ECDF Curve (Right Axis)
        p.setPen(QPen(QColor("#f59e0b"), 2, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        path_ecdf = QPainterPath()
        path_ecdf.moveTo(plot.left(), plot.bottom())
        # Downsample for drawing
        step = max(1, len(data) // 100)
        for i in range(0, len(data), step):
            px = plot.left() + (data[i] - v_min)/(v_max - v_min) * plot.width()
            py = plot.bottom() - (i / len(data)) * plot.height()
            path_ecdf.lineTo(px, py)
        path_ecdf.lineTo(plot.right(), plot.top())
        p.drawPath(path_ecdf)

        # 4. Rug Plot (Bottom)
        p.setPen(QPen(QColor("#e5e7eb"), 1))
        p.setBrush(Qt.NoBrush)
        rug_y = plot.bottom() + 4
        for v in data:
            px = plot.left() + (v - v_min)/(v_max - v_min) * plot.width()
            p.drawLine(int(px), int(rug_y), int(px), int(rug_y + 4))

        # 5. Stats Markers
        stats = {
            "Mean": np.mean(data),
            "Max": v_max,
        }
        p.setFont(QFont("Segoe UI", 8))
        for k, v in stats.items():
            px = plot.left() + (v - v_min)/(v_max - v_min) * plot.width()
            p.setPen(QPen(QColor("#e5e7eb"), 1, Qt.DashLine))
            p.drawLine(int(px), plot.top(), int(px), plot.bottom())
            p.setPen(QColor("#e5e7eb"))
            p.drawText(int(px) + 4, plot.top() + 10 if k == "Mean" else plot.top() + 24, f"{k}:{v:.1f}")

        # Axes
        p.setPen(QColor("#9ca3af"))
        p.drawText(rect.adjusted(margin, rect.height()-18, -margin, -2), Qt.AlignLeft, f"{v_min:.1f}")
        p.drawText(plot.adjusted(0, plot.height()+2, 0, 18), Qt.AlignRight, f"{v_max:.1f}")
        
        # Right Axis Labels (0% - 100%)
        p.setPen(QColor("#f59e0b"))
        p.drawText(rect.adjusted(rect.width()-36, margin+title_h, -2, 0), Qt.AlignRight|Qt.AlignTop, "100%")
        p.drawText(rect.adjusted(rect.width()-36, rect.height()-margin-20, -2, -margin), Qt.AlignRight|Qt.AlignBottom, "0%")



class TerritoryScatterPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: List[Tuple[float, float]] = []
        self._title = ""
        self._current_point: Optional[Tuple[float, float]] = None
        self._color = QColor("#3b82f6")
        self.setMinimumHeight(220)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_points(self, points: Sequence[Tuple[float, float]], *, title: str = "", color: str = "#3b82f6"):
        self._points = [(float(x), float(y)) for x, y in points]
        self._title = title
        self._color = QColor(color)
        self.update()

    def set_current_point(self, point: Optional[Tuple[float, float]]):
        self._current_point = (float(point[0]), float(point[1])) if point is not None else None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        # No background fill

        margin = 12
        title_h = 18 if self._title else 0
        plot = rect.adjusted(margin, margin + title_h, -margin, -margin)

        if self._title:
            p.setPen(QPen(QColor("#93c5fd")))
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            p.drawText(rect.adjusted(margin, margin, -margin, -margin), Qt.AlignLeft | Qt.AlignTop, self._title)

        if not self._points:
            p.setPen(QPen(QColor("#6b7280")))
            p.drawText(plot, Qt.AlignLeft | Qt.AlignTop, "暂无数据")
            return

        xs = [pt[0] for pt in self._points]
        ys = [pt[1] for pt in self._points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Add some padding
        pad_x = (x_max - x_min) * 0.1 if x_max > x_min else 10.0
        pad_y = (y_max - y_min) * 0.1 if y_max > y_min else 10.0
        x_min -= pad_x
        x_max += pad_x
        y_min -= pad_y
        y_max += pad_y

        def to_pos(px, py):
            ix = plot.left() + (px - x_min) / (x_max - x_min) * plot.width()
            iy = plot.top() + (py - y_min) / (y_max - y_min) * plot.height()
            return QPointF(ix, iy)

        # Draw scatter
        p.setPen(Qt.NoPen)
        c = QColor(self._color)
        c.setAlpha(100)
        p.setBrush(c)
        for pt in self._points:
            pos = to_pos(pt[0], pt[1])
            p.drawEllipse(pos, 3, 3)

        # Draw Convex Hull if possible
        if SCIPY_AVAILABLE and len(self._points) >= 3:
            try:
                hull = ConvexHull(self._points)
                poly = QPolygonF()
                for v in hull.vertices:
                    poly.append(to_pos(self._points[v][0], self._points[v][1]))
                
                hull_c = QColor(self._color)
                hull_c.setAlpha(40)
                p.setBrush(hull_c)
                p.setPen(QPen(self._color, 1, Qt.DashLine))
                p.drawPolygon(poly)
            except Exception:
                pass

        # Draw current point
        if self._current_point:
            pos = to_pos(self._current_point[0], self._current_point[1])
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor("#f59e0b"), 2))
            p.drawLine(QPointF(pos.x() - 8, pos.y()), QPointF(pos.x() + 8, pos.y()))
            p.drawLine(QPointF(pos.x(), pos.y() - 8), QPointF(pos.x(), pos.y() + 8))
            p.drawEllipse(pos, 6, 6)

        p.setPen(QPen(QColor("#1f2937"), 1))
        p.drawRoundedRect(plot.adjusted(0, 0, -1, -1), 10, 10)
