import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.patches import Polygon, Rectangle, Circle, Arc
import matplotlib.pyplot as plt
import seaborn as sns
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QCursor, QPixmap
import io

# Set dark theme for matplotlib
plt.style.use('dark_background')
# Configure Chinese font support
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.fig.patch.set_facecolor('#1e1e1e') # Dark background
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e1e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self._hover_preview = None
        self.setMouseTracking(True)

    def _ensure_preview(self):
        if self._hover_preview is None:
            w = QWidget(None)
            w.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            w.setAttribute(Qt.WA_TranslucentBackground)
            lay = QVBoxLayout(w)
            lay.setContentsMargins(4, 4, 4, 4)
            lbl = QLabel(w)
            lbl.setStyleSheet("background-color: #202020; border: 1px solid #444;")
            lay.addWidget(lbl)
            self._hover_preview = w
            self._hover_label = lbl

    def _make_preview_pixmap(self):
        buf = io.BytesIO()
        try:
            self.fig.canvas.draw()
            self.fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            target_w = int(self.width() * 1.6)
            target_h = int(self.height() * 1.6)
            return pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        except Exception:
            return None

    def enterEvent(self, event):
        self._ensure_preview()
        pm = self._make_preview_pixmap()
        if pm:
            self._hover_label.setPixmap(pm)
            gp = QCursor.pos()
            self._hover_preview.move(gp + QPoint(20, 20))
            self._hover_preview.show()
        super().enterEvent(event)

    def mouseMoveEvent(self, event):
        if self._hover_preview and self._hover_preview.isVisible():
            gp = QCursor.pos()
            self._hover_preview.move(gp + QPoint(20, 20))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hover_preview:
            self._hover_preview.hide()
        super().leaveEvent(event)

    def cleanup(self):
        self.axes.cla()
        self.fig.clf()

class RadarChart(MplCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, width=5, height=4, dpi=100)
        self.axes.remove()
        self.axes = self.fig.add_subplot(111, polar=True)
        self.axes.set_facecolor('#1e1e1e')
        
    def plot(self, p1_stats, p2_stats, categories):
        self.axes.cla()
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1] # Close the loop
        
        # Draw Player 1
        values1 = list(p1_stats.values())
        values1 += values1[:1]
        self.axes.plot(angles, values1, linewidth=2, linestyle='solid', label='球员1', color='#ff4d4d')
        self.axes.fill(angles, values1, '#ff4d4d', alpha=0.25)
        
        # Draw Player 2
        values2 = list(p2_stats.values())
        values2 += values2[:1]
        self.axes.plot(angles, values2, linewidth=2, linestyle='solid', label='球员2', color='#4d79ff')
        self.axes.fill(angles, values2, '#4d79ff', alpha=0.25)
        
        # Labels
        self.axes.set_xticks(angles[:-1])
        self.axes.set_xticklabels(categories, color='white', size=10)
        
        # Y labels
        self.axes.set_rlabel_position(0)
        self.axes.set_yticks([20, 40, 60, 80, 100])
        self.axes.set_yticklabels(["20", "40", "60", "80", "100"], color="grey", size=7)
        self.axes.set_ylim(0, 100)
        
        # Grid color
        self.axes.grid(color='grey', alpha=0.3)
        self.axes.spines['polar'].set_visible(False)
        
        # Legend
        self.axes.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor='#333', edgecolor='none', labelcolor='white')
        
        self.draw()

class CourtMapBase(MplCanvas):
    def draw_court(self, ax=None):
        if ax is None: ax = self.axes
        
        # Court dimensions (standard) - mapped to image coords roughly if needed
        # Or just use standard dimensions 13.4 x 6.1 and normalize data points to it
        # Here we assume data points are in image pixels (e.g. 1280x720)
        # We need a way to map them. For now, let's assume we plot in pixel coordinates directly
        # and overlay a "schematic" court is hard without homography.
        # BETTER APPROACH: Just draw the data points (heatmap) on black, 
        # assuming the user knows the court shape from the points.
        # OR: Use a generic rectangle if we don't have homography.
        
        # Since we want "Professional", we ideally project points to top-down view.
        # But we don't have the homography matrix here easily. 
        # However, TrackNet usually outputs ball_x, ball_y in screen coordinates.
        # For a top-down view, we need a homography transform. 
        # IF we don't have it, we display in "Screen View" (Perspective).
        # Let's stick to Screen View for now, but flip Y so 0 is bottom? No, image coords 0 is top.
        
        ax.invert_yaxis() # Match image coords
        ax.set_aspect('equal')
        ax.axis('off')

class HitPoint3D(MplCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, width=5, height=4)
        self.axes.remove()
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_facecolor('#1e1e1e')
        self.axes.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.axes.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.axes.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))

    def plot(self, df):
        self.axes.cla()
        
        hits = df[df['is_hit'] == 1]
        if hits.empty: return
        
        # x, y, z(approx height)
        xs = hits['ball_x'].values
        ys = hits['ball_y'].values
        ys = 720 - ys
        
        # Z heuristic: assume linear relationship with Y for depth? 
        # Actually without 3D calibration, we use Y as depth, and 720-Y as height is wrong.
        # But let's assume standard camera angle:
        # X is width. Y is depth (slanted). Z is height.
        # Here we only have 2D (x, y). 
        # We can map Y to Depth, and try to infer Height? No, impossible without calibration.
        # So we just plot X, Y and "Color" as speed or type.
        # OR: We make a 3D scatter where Z is "Speed" or "Height Proxy"
        
        # Let's visualize: X=Width, Y=Depth, Z=Height (Proxy: 720 - ball_y, but that's wrong for depth)
        # Better: Z = Hit Impact Height (Low vs High)
        # Let's assume High Y (small pixel val) is Far Depth. Low Y (large pixel val) is Near Depth.
        # Height is unknown.
        # Let's plot 3D: X=Width, Y=Depth (Y-coord), Z=Speed
        
        zs = hits['ball_speed'].values
        
        scatter = self.axes.scatter(xs, ys, zs, c=hits['hit_player'], cmap='coolwarm', s=50, depthshade=True)
        
        self.axes.set_xlabel("宽度 (X)")
        self.axes.set_ylabel("深度 (Y)")
        self.axes.set_zlabel("球速")
        self.axes.set_title("3D击球分析（球速）", color='white')
        self.draw()

class LoadChart(MplCanvas):
    def plot(self, df):
        self.axes.cla()
        
        # Calculate cumulative distance (Load)
        # We need time series of distance for P1 and P2
        
        # Create a time index
        df_sorted = df.sort_values('time_seconds')
        t = df_sorted['time_seconds']
        
        # Cumulative sum of speed * dt = distance
        # Fill NA speeds
        v1 = df_sorted['p1_speed'].fillna(0)
        v2 = df_sorted['p2_speed'].fillna(0)
        dt = df_sorted['time_seconds'].diff().fillna(0.04)
        
        dist1 = (v1 * dt).cumsum()
        dist2 = (v2 * dt).cumsum()
        
        self.axes.plot(t, dist1, color='#ff4d4d', label='球员1负荷', linewidth=2)
        self.axes.plot(t, dist2, color='#4d79ff', label='球员2负荷', linewidth=2)
        
        self.axes.fill_between(t, dist1, color='#ff4d4d', alpha=0.1)
        self.axes.fill_between(t, dist2, color='#4d79ff', alpha=0.1)
        
        self.axes.set_xlabel("时间（秒）", color='white')
        self.axes.set_ylabel("累计距离（像素）", color='white')
        self.axes.set_title("体能负荷（距离）", color='white')
        self.axes.legend(facecolor='#333', edgecolor='none', labelcolor='white')
        self.axes.grid(True, color='#333')
        self.axes.tick_params(colors='white')
        
        self.draw()

class SankeyChart(MplCanvas):
    def plot(self, data):
        self.axes.cla()
        self.axes.axis('off')
        
        if not data or not data['sources']:
            self.axes.text(0.5, 0.5, "暂无桑基数据", ha='center', color='white')
            self.draw()
            return
            
        # Simplified Sankey using parallel coordinates or just connection lines
        # Since matplotlib doesn't have built-in Sankey for complex flows easily,
        # We simulate a 2-stage flow: Service (Left) -> Outcome (Right)
        
        sources = data['sources'] # List of source labels
        targets = data['targets'] # List of target labels
        values = data['values']   # List of counts
        
        # Get unique nodes and map to y-positions
        unique_src = sorted(list(set(sources)))
        unique_tgt = sorted(list(set(targets)))
        
        src_y = np.linspace(0.8, 0.2, len(unique_src))
        tgt_y = np.linspace(0.8, 0.2, len(unique_tgt))
        
        src_map = {k: v for k, v in zip(unique_src, src_y)}
        tgt_map = {k: v for k, v in zip(unique_tgt, tgt_y)}
        
        # Draw connections
        max_val = max(values) if values else 1
        
        for s, t, v in zip(sources, targets, values):
            y1 = src_map[s]
            y2 = tgt_map[t]
            width = (v / max_val) * 10
            
            # Draw bezier curve
            self.draw_bezier(0.2, y1, 0.8, y2, width, color='#00ffcc', alpha=0.5)
            
        # Draw Nodes
        for k, y in src_map.items():
            self.axes.text(0.1, y, k, ha='right', va='center', color='white', fontsize=10)
            self.axes.scatter(0.2, y, color='#ff4d4d', s=100, zorder=3)
            
        for k, y in tgt_map.items():
            self.axes.text(0.9, y, k, ha='left', va='center', color='white', fontsize=10)
            self.axes.scatter(0.8, y, color='#4d79ff', s=100, zorder=3)
            
        self.axes.set_xlim(0, 1)
        self.axes.set_ylim(0, 1)
        self.axes.set_title("战术流向：发球 → 结局", color='white')
        self.draw()
        
    def draw_bezier(self, x1, y1, x2, y2, width, color, alpha):
        t = np.linspace(0, 1, 100)
        # Cubic Bezier with control points
        xc1 = x1 + 0.3
        xc2 = x2 - 0.3
        
        x = (1-t)**3*x1 + 3*(1-t)**2*t*xc1 + 3*(1-t)*t**2*xc2 + t**3*x2
        y = (1-t)**3*y1 + 3*(1-t)**2*t*y1 + 3*(1-t)*t**2*y2 + t**3*y2
        
        self.axes.plot(x, y, linewidth=width, color=color, alpha=alpha)

class TransitionHeatmap(MplCanvas):
    def plot(self, df):
        self.axes.cla()
        
        if df.empty:
            self.axes.text(0.5, 0.5, "暂无转移数据", ha='center', color='white')
            self.draw()
            return
            
        sns.heatmap(df, ax=self.axes, cmap='viridis', annot=True, fmt='.2f', 
                    cbar=False, annot_kws={"size": 8})
        
        self.axes.set_title("战术转移概率", color='white')
        self.axes.set_xlabel("响应击球", color='white')
        self.axes.set_ylabel("来球类型", color='white')
        self.axes.tick_params(colors='white', rotation=45)
        self.draw()

class TransitionChordChart(MplCanvas):
    def plot(self, chord_data: dict, title="战术转移弦图"):
        self.axes.cla()
        self.axes.axis('off')
        nodes = chord_data.get('nodes', [])
        links = chord_data.get('links', [])
        if not nodes or not links:
            self.axes.text(0.5, 0.5, "暂无转移数据", ha='center', color='white')
            self.draw()
            return
        labels = [n['name'] for n in nodes]
        n = len(labels)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 1.0
        node_xy = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]
        cmap = plt.get_cmap('Set2')
        label_color = {labels[i]: cmap(i % cmap.N) for i in range(n)}
        # nodes
        for i, (x, y) in enumerate(node_xy):
            self.axes.scatter([x], [y], s=160, color=label_color[labels[i]], edgecolors='white', linewidths=0.8, alpha=0.9)
            self.axes.text(x*1.12, y*1.12, labels[i], color='white', ha='center', va='center', fontsize=10)
        # links
        max_v = max(l.get('value', 1) for l in links) if links else 1
        idx_map = {labels[i]: i for i in range(n)}
        for l in links:
            s = l['source']; t = l['target']; v = l.get('value', 1)
            if s not in idx_map or t not in idx_map: continue
            i = idx_map[s]; j = idx_map[t]
            x1, y1 = node_xy[i]; x2, y2 = node_xy[j]
            ctrl = ((x1+x2)/2.0, (y1+y2)/2.0)
            tlin = np.linspace(0, 1, 120)
            bx = (1-tlin)**2*x1 + 2*(1-tlin)*tlin*ctrl[0] + tlin**2*x2
            by = (1-tlin)**2*y1 + 2*(1-tlin)*tlin*ctrl[1] + tlin**2*y2
            lw = 0.6 + 6.0*(v/max_v)
            color = label_color[s]
            self.axes.plot(bx, by, color=color, linewidth=lw, alpha=0.75)
        circle = plt.Circle((0,0), radius, color='#333', fill=False, linewidth=1.0)
        self.axes.add_artist(circle)
        self.axes.axis('equal')
        self.axes.set_title(title, color='white')
        self.draw()
class ThemeRiverChart(MplCanvas):
    def plot(self, river_data: dict, title="战术类型流图"):
        self.axes.cla()
        times = river_data.get('times', [])
        series = river_data.get('series', {})
        if not times or not series:
            self.axes.text(0.5, 0.5, "暂无战术数据", ha='center', color='white')
            self.draw()
            return
        t = np.array(times, dtype=float)
        types = list(series.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(types)))
        vals = np.vstack([np.array(series[tp], dtype=float) for tp in types])
        # Normalize rows to ensure stack sums ~1
        vals = np.clip(vals, 0, 1)
        cum = np.cumsum(vals, axis=0)
        base = np.zeros_like(t)
        for i, tp in enumerate(types):
            upper = base + vals[i]
            self.axes.fill_between(t, base, upper, color=colors[i], alpha=0.8, label=tp)
            base = upper
        self.axes.set_title(f"{title}（滑窗{river_data.get('window_sec', 2.0)}秒）", color='white')
        self.axes.set_xlabel("时间（秒）", color='white')
        self.axes.set_ylabel("占比", color='white')
        self.axes.set_ylim(0, 1)
        self.axes.grid(True, color='#333', alpha=0.3)
        self.axes.tick_params(colors='white')
        self.axes.legend(facecolor='#333', edgecolor='none', labelcolor='white', ncol=min(3, len(types)))
        self.draw()
class MomentumChart(MplCanvas):
    def plot(self, rallies):
        self.axes.cla()
        
        # Calculate momentum: (Hit Count * Duration) as proxy for intensity
        x = range(1, len(rallies) + 1)
        y = [r.hit_count * r.duration_sec for r in rallies] # Intensity
        
        # Color by hit count
        colors = [r.hit_count for r in rallies]
        
        if not rallies:
            self.axes.text(0.5, 0.5, "暂无回合数据", ha='center', color='white')
        else:
            scatter = self.axes.scatter(x, y, c=colors, cmap='viridis', s=50, zorder=2)
            if len(rallies) > 1:
                self.axes.plot(x, y, color='grey', alpha=0.5, zorder=1)
        
        self.axes.set_xlabel("回合序号", color='white')
        self.axes.set_ylabel("强度（拍数×时长）", color='white')
        self.axes.tick_params(colors='white')
        
        # Force integer ticks for X axis if few rallies
        if len(rallies) < 10:
            from matplotlib.ticker import MaxNLocator
            self.axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            
        self.axes.grid(True, color='#333')
        
        # Add colorbar
        # cbar = self.fig.colorbar(scatter, ax=self.axes)
        # cbar.ax.yaxis.set_tick_params(color='white')
        
        self.draw()

class RallyComplexityChart(MplCanvas):
    def plot(self, rallies):
        import pandas as pd
        self.axes.cla()
        if not rallies:
            self.axes.text(0.5, 0.5, "暂无回合数据", ha='center', color='white')
            self.draw()
            return
        rows = []
        for r in rallies:
            df = r.trajectory
            avg_speed = float(pd.to_numeric(df.get('ball_speed', pd.Series()), errors='coerce').dropna().mean()) if df is not None else 0.0
            rows.append({'hits': r.hit_count, 'duration': r.duration_sec, 'avg_speed': avg_speed})
        d = pd.DataFrame(rows)
        if d.empty:
            self.axes.text(0.5, 0.5, "No Rally Stats", ha='center', color='white')
            self.draw()
            return
        try:
            sns.kdeplot(x=d['hits'], y=d['duration'], ax=self.axes, fill=True, cmap='plasma', levels=30, thresh=0.05, alpha=0.7)
        except Exception:
            pass
        sc = self.axes.scatter(d['hits'], d['duration'], c=d['avg_speed'], cmap='viridis', s=40, edgecolors='none')
        self.axes.set_xlabel("拍数", color='white')
        self.axes.set_ylabel("时长（秒）", color='white')
        self.axes.set_title("回合复杂度密度", color='white')
        cb = self.fig.colorbar(sc, ax=self.axes, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors='white')
        self.axes.tick_params(colors='white')
        self.axes.grid(True, color='#333')
        self.draw()

class RadarBarsChart(MplCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, width=5, height=4, dpi=100)
        self.axes.remove()
        self.axes = self.fig.add_subplot(111, polar=True)
        self.axes.set_facecolor('#1e1e1e')
    def plot(self, p1_stats, p2_stats, categories):
        import numpy as np
        self.axes.cla()
        N = len(categories)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        width = (2*np.pi / N) * 0.35
        vals1 = np.array(list(p1_stats.values()))
        vals2 = np.array(list(p2_stats.values()))
        self.axes.bar(angles - width*0.6, vals1, width=width, color='#ff4d4d', alpha=0.6, label='球员1')
        self.axes.bar(angles + width*0.6, vals2, width=width, color='#4d79ff', alpha=0.6, label='球员2')
        self.axes.set_xticks(angles)
        self.axes.set_xticklabels(categories, color='white', fontsize=10)
        self.axes.set_ylim(0, 100)
        self.axes.grid(color='grey', alpha=0.3)
        self.axes.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor='#333', edgecolor='none', labelcolor='white')
        self.draw()
class ShotQualityViolinChart(MplCanvas):
    def plot(self, df):
        import pandas as pd
        self.axes.cla()
        hits = df[df.get('is_hit', 0) == 1].copy()
        if hits.empty:
            self.axes.text(0.5, 0.5, "暂无击球数据", ha='center', color='white')
            self.draw()
            return
        hits['Type'] = hits['stroke_type_name'].astype(str)
        type_labels = {
            '杀': '杀球', 'smash': '杀球',
            '抽': '抽球', 'drive': '抽球',
            '吊': '吊球', 'drop': '吊球',
            '网': '网前', 'net': '网前',
            '挑': '挑球', 'lift': '挑球',
            '高': '高远', 'clear': '高远'
        }
        def norm(x):
            xl = x.lower()
            for k, v in type_labels.items():
                if k in xl:
                    return v
            return '其他'
        hits['Type'] = hits['Type'].apply(norm)
        hits = hits[hits['Type'] != '其他']
        if hits.empty:
            self.axes.text(0.5, 0.5, "No Typed Hit Data", ha='center', color='white')
            self.draw()
            return
        sns.violinplot(data=hits, x='Type', y='ball_speed', ax=self.axes, inner='quartile', cut=0, palette='Set3')
        self.axes.set_xlabel("击球类型", color='white')
        self.axes.set_ylabel("球速（px/s）", color='white')
        self.axes.set_title("击球质量（速度分布）", color='white')
        self.axes.tick_params(colors='white', axis='x', rotation=20)
        self.axes.tick_params(colors='white', axis='y')
        self.axes.grid(True, color='#333', alpha=0.3)
        self.draw()

class ParallelCoordinatesChart(MplCanvas):
    def plot(self, engine):
        import pandas as pd
        self.axes.cla()
        df = engine.df.copy()
        if df is None or df.empty:
            self.axes.text(0.5, 0.5, "暂无数据", ha='center', color='white')
            self.draw()
            return
        hits = df[df.get('is_hit', 0) == 1].copy()
        if hits.empty:
            self.axes.text(0.5, 0.5, "暂无击球数据", ha='center', color='white')
            self.draw()
            return
        hits['Height'] = 720 - hits['ball_y'].astype(float)
        hits['Speed'] = hits['ball_speed'].astype(float)
        p1_speed = hits['p1_speed'].astype(float)
        p2_speed = hits['p2_speed'].astype(float)
        hits['PlayerSpeed'] = p1_speed.where(hits['hit_player'] == 1, p2_speed)
        dims = ['Speed', 'Height', 'PlayerSpeed']
        scaled = hits[dims].copy()
        for d in dims:
            mn = float(scaled[d].min())
            mx = float(scaled[d].max())
            if mx - mn <= 1e-6:
                scaled[d] = 0.5
            else:
                scaled[d] = (scaled[d] - mn) / (mx - mn)
        x = np.arange(len(dims))
        self.axes.set_xticks(x)
        self.axes.set_xticklabels(['球速','高度','选手速度'], color='white')
        for _, row in scaled.iterrows():
            self.axes.plot(x, row.values, color='#00ffcc', alpha=0.3, linewidth=1.0)
        self.axes.set_ylim(0, 1)
        self.axes.grid(True, color='#333', alpha=0.4)
        self.axes.set_title("并行坐标：球速/高度/选手速度", color='white')
        self.draw()
class FeatureCourt3D(MplCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, width=6, height=5, dpi=100)
        self.axes.remove()
        self.axes = self.fig.add_subplot(111, projection='3d')
        # Transparent background
        self.fig.patch.set_alpha(0.0)
        self.axes.set_facecolor((0, 0, 0, 0))
        try:
            self.axes.xaxis.set_pane_color((0, 0, 0, 0))
            self.axes.yaxis.set_pane_color((0, 0, 0, 0))
            self.axes.zaxis.set_pane_color((0, 0, 0, 0))
        except Exception:
            pass
        try:
            self.axes.set_box_aspect((1, 1, 0.8))
        except Exception:
            pass
        try:
            self.axes.set_anchor('C')
        except Exception:
            pass
        self._color_by_speed = True
        self._show_hulls = True
        self._show_lines = True
        self._dist = 9.0
        self._engine = None
        self._last_data = {}
        self._cbar = None
    def wheelEvent(self, event):
        try:
            delta = event.angleDelta().y()
        except Exception:
            delta = 0
        if delta > 0:
            self.zoom(1.12)
        elif delta < 0:
            self.zoom(1/1.12)
        event.accept()
    def _normalize(self, arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return arr
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-6:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)
    def _type_weight(self, name):
        if not name:
            return 1.0
        n = str(name).lower()
        if 'smash' in n or '杀' in n:
            return 1.30
        if 'drive' in n or '抽' in n:
            return 1.10
        if 'drop' in n or '吊' in n:
            return 0.90
        if 'net' in n or '网' in n:
            return 0.80
        if 'lift' in n or 'clear' in n or '挑' in n or '高' in n:
            return 1.00
        return 1.00
    def set_color_mode(self, mode: str):
        self._color_by_speed = (mode == 'speed')
        self._replot()
    def set_show_hulls(self, flag: bool):
        self._show_hulls = flag
        self._replot()
    def set_show_lines(self, flag: bool):
        self._show_lines = flag
        self._replot()
    def zoom(self, factor: float):
        if factor <= 0:
            return
        self._dist = max(3.0, min(25.0, self._dist / factor))
        try:
            self.axes.dist = self._dist
        except Exception:
            pass
        self.draw()
    def reset_zoom(self):
        self._dist = 9.0
        try:
            self.axes.dist = self._dist
        except Exception:
            pass
        self.draw()
    def _replot(self):
        if self._engine is not None:
            self.plot(self._engine)
    def plot(self, engine):
        import pandas as pd
        # Recreate axes using GridSpec: main 3D axes + right-side colorbar axes
        try:
            self.fig.clear()
            gs = self.fig.add_gridspec(1, 2, width_ratios=[25, 1])
            self.axes = self.fig.add_subplot(gs[0], projection='3d')
            self._cax = self.fig.add_subplot(gs[1])
            # Transparent backgrounds
            self.fig.patch.set_alpha(0.0)
            self.axes.set_facecolor((0, 0, 0, 0))
            self._cax.set_facecolor((0, 0, 0, 0))
            try:
                self.axes.xaxis.set_pane_color((0, 0, 0, 0))
                self.axes.yaxis.set_pane_color((0, 0, 0, 0))
                self.axes.zaxis.set_pane_color((0, 0, 0, 0))
            except Exception:
                pass
            try:
                self.axes.set_box_aspect((1, 1, 0.8))
                self.axes.set_anchor('C')
            except Exception:
                pass
            # Reset previous colorbar
            self._cbar = None
        except Exception:
            # Fallback to clearing existing axes
            self.axes.cla()
        self._engine = engine
        df = engine.df
        if df is None or df.empty:
            self.axes.text(0.5, 0.5, "No Data", color='white')
            self.draw()
            return
        hits = df[df.get('is_hit', 0) == 1].copy()
        if hits.empty:
            self.axes.text(0.5, 0.5, "No Hit Data", color='white')
            self.draw()
            return
        W = float(df['ball_x'].max() if 'ball_x' in df else 1280)
        H = float(df['ball_y'].max() if 'ball_y' in df else 720)
        # 3D coordinates
        X = hits['ball_x'].astype(float) / max(W, 1.0)
        Y = hits['ball_y'].astype(float) / max(H, 1.0)
        Z = (H - hits['ball_y'].astype(float)) / max(H, 1.0)
        speed = hits['ball_speed'].astype(float).fillna(0.0)
        speed_n = self._normalize(speed)
        # size by threat
        type_names = hits.get('stroke_type_name', pd.Series([''] * len(hits)))
        weights = np.array([self._type_weight(t) for t in type_names], dtype=float)
        sizes = 26.0 + 22.0 * np.power(speed_n, 1.3) * weights
        sizes = np.maximum(sizes, 18.0)
        # color by mode
        cm = plt.get_cmap('turbo')
        colors = None
        if self._color_by_speed:
            colors = speed_n
        else:
            # Discrete palette by type or player
            palette = plt.cm.Set2(np.linspace(0, 1, 8))
            categories = type_names.fillna('').astype(str)
            # fallback to player color if type empty
            if (categories == '').all():
                cats = hits['hit_player'].astype(int).astype(str)
            else:
                cats = categories
            uniq = sorted(list(set(cats)))
            cmap_map = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
            colors = np.array([cmap_map[u] for u in cats])
        # draw court grid (floor)
        floor_x = np.linspace(0, 1, 20)
        floor_y = np.linspace(0, 1, 20)
        FX, FY = np.meshgrid(floor_x, floor_y)
        FZ = np.zeros_like(FX)
        self.axes.plot_wireframe(FX, FY, FZ, color='#555555', rstride=2, cstride=2, alpha=0.25)
        # draw net plane at mid-depth
        net_y = 0.5
        net_x = np.array([0, 1, 1, 0, 0])
        net_y_poly = np.array([net_y, net_y, net_y, net_y, net_y])
        net_z = np.array([0, 0.15, 0.15, 0, 0])
        self.axes.plot(net_x, net_y_poly, net_z, color='#8888ff', alpha=0.45)
        # scatter points
        glow = self.axes.scatter(X, Y, Z, c=colors, cmap=cm if self._color_by_speed else None, s=sizes*2.0, depthshade=True, alpha=0.14, edgecolors='none')
        sc = self.axes.scatter(X, Y, Z, c=colors, cmap=cm if self._color_by_speed else None, s=sizes, depthshade=True, alpha=0.95, edgecolors='white', linewidths=0.6)
        # colorbar
        if self._color_by_speed:
            try:
                self._cbar = self.fig.colorbar(sc, cax=self._cax)
                self._cbar.ax.tick_params(colors='white')
                self._cbar.set_label("球速（归一化）", color='white')
            except Exception:
                self._cbar = self.fig.colorbar(sc, ax=self.axes, fraction=0.02, pad=0.02)
                self._cbar.ax.tick_params(colors='white')
                self._cbar.set_label("球速（归一化）", color='white')
        # rally lines
        if self._show_lines and engine.rallies:
            for r in engine.rallies:
                r_hits = r.trajectory
                r_hits = r_hits[r_hits.get('is_hit', 0) == 1]
                if len(r_hits) < 2:
                    continue
                x = r_hits['ball_x'].astype(float) / max(W, 1.0)
                y = r_hits['ball_y'].astype(float) / max(H, 1.0)
                z = (H - r_hits['ball_y'].astype(float)) / max(H, 1.0)
                sp = r_hits['ball_speed'].astype(float).fillna(0.0)
                c = self._normalize(sp).mean()
                self.axes.plot(x, y, z, color=(cm(c) if self._color_by_speed else '#00ffcc'), alpha=0.65, linewidth=2.0)
                if len(x) >= 2:
                    dx = np.diff(x)
                    dy = np.diff(y)
                    dz = np.diff(z)
                    self.axes.quiver(x[:-1], y[:-1], z[:-1], dx, dy, dz, length=1.0, normalize=True, color=(cm(c) if self._color_by_speed else '#00ffcc'), alpha=0.35, linewidth=0.8)
                # Start/End markers
                try:
                    self.axes.scatter([x[0]], [y[0]], [z[0]], s=70, marker='^', color='#ffff66', edgecolors='white', linewidths=0.8, alpha=0.95)
                    self.axes.scatter([x[-1]], [y[-1]], [z[-1]], s=70, marker='s', color='#66ffcc', edgecolors='white', linewidths=0.8, alpha=0.95)
                except Exception:
                    pass
        # convex hulls per type
        if self._show_hulls and 'stroke_type_name' in hits.columns:
            from scipy.spatial import ConvexHull
            for tname, grp in hits.groupby('stroke_type_name'):
                if len(grp) < 20:
                    continue
                pts = np.vstack([
                    grp['ball_x'].astype(float) / max(W, 1.0),
                    grp['ball_y'].astype(float) / max(H, 1.0),
                    (H - grp['ball_y'].astype(float)) / max(H, 1.0)
                ]).T
                try:
                    hull = ConvexHull(pts)
                    for simplex in hull.simplices:
                        tri = pts[simplex]
                        self.axes.plot_trisurf(tri[:,0], tri[:,1], tri[:,2], color='#00ffcc', alpha=0.12, linewidth=0)
                    # centroid
                    c = pts.mean(axis=0)
                    self.axes.scatter([c[0]],[c[1]],[c[2]], s=40, color='#00ffcc', alpha=0.6)
                    self.axes.text(c[0], c[1], c[2]+0.02, str(tname), color='white', fontsize=8)
                except Exception:
                    continue
        # labels and view
        self.axes.set_xlabel("宽度（左→右）", color='white')
        self.axes.set_ylabel("深度（前→后）", color='white')
        self.axes.set_zlabel("高度（低→高）", color='white')
        self.axes.tick_params(colors='white')
        self.axes.view_init(elev=28, azim=-62)
        try:
            self.axes.dist = self._dist
        except Exception:
            pass
        self.axes.set_xlim(0, 1)
        self.axes.set_ylim(0, 1)
        self.axes.set_zlim(0, 1)
        # XY density projection
        try:
            gx = np.linspace(0,1,40)
            gy = np.linspace(0,1,40)
            Hxy, xedges, yedges = np.histogram2d(X, Y, bins=[gx, gy])
            Xg, Yg = np.meshgrid((xedges[:-1]+xedges[1:])/2, (yedges[:-1]+yedges[1:])/2)
            self.axes.contourf(Xg, Yg, Hxy.T, zdir='z', offset=0, cmap='inferno', alpha=0.25)
        except Exception:
            pass
        # highlight top-speed hits
        try:
            thresh = np.quantile(speed_n, 0.95)
            idx = np.where(speed_n >= thresh)[0]
            self.axes.scatter(np.asarray(X)[idx], np.asarray(Y)[idx], np.asarray(Z)[idx], s=np.asarray(sizes)[idx]*2.2, color='#ffcc00', alpha=0.95, marker='^', edgecolors='white', linewidths=0.8)
        except Exception:
            pass
        self.axes.set_title("击球落点-高度 3D 战术图", color='white')
        self.draw()
class VoronoiMap(CourtMapBase):
    def plot(self, p1_pos, p2_pos):
        self.axes.cla()
        self.draw_court()
        
        # p1_pos, p2_pos are (x, y) tuples
        points = np.array([p1_pos, p2_pos])
        
        # Define bounds (screen size)
        bounds = [0, 1280, 0, 720] # xmin, xmax, ymin, ymax
        
        # Simple Voronoi for 2 points is a perpendicular bisector line
        # We can just fill two polygons
        
        # Perpendicular bisector
        mid = (points[0] + points[1]) / 2
        vec = points[1] - points[0]
        # Normal vector (-dy, dx)
        normal = np.array([-vec[1], vec[0]])
        
        # This is a bit complex to clip to rectangle manually with matplotlib polygons quickly
        # Alternative: nearest neighbor classification for a grid of points (contourf)
        
        grid_x, grid_y = np.mgrid[0:1280:20j, 0:720:20j]
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
        
        # Distances to P1 and P2
        d1 = np.linalg.norm(grid_points - points[0], axis=1)
        d2 = np.linalg.norm(grid_points - points[1], axis=1)
        
        # Mask: 0 for P1, 1 for P2
        mask = (d2 < d1).astype(int)
        mask = mask.reshape(grid_x.shape)
        
        self.axes.contourf(grid_x, grid_y, mask, levels=[-0.1, 0.5, 1.1], 
                           colors=['#ff4d4d', '#4d79ff'], alpha=0.3)
        
        # Draw players
        self.axes.scatter(*points[0], c='#ff4d4d', s=200, label='球员1', edgecolors='white')
        self.axes.scatter(*points[1], c='#4d79ff', s=200, label='球员2', edgecolors='white')
        
        self.axes.set_title("空间控制（Voronoi）", color='white')
        self.draw()

class SpeedHeightScatter(MplCanvas):
    def plot(self, df):
        self.axes.cla()
        # x: speed, y: y-coordinate (height proxy in 2D image)
        # In image coords, smaller y is higher. So we invert y.
        
        # Filter hits
        hits = df[df['is_hit'] == 1]
        if hits.empty: return

        x = hits['ball_speed']
        y = hits['ball_y'] # Pixel height (0 is top)
        
        # Invert Y to show "Height" (0 at bottom) - approx
        # Assuming 720p
        y_height = 720 - y
        
        scatter = self.axes.scatter(x, y_height, c=hits['hit_player'], cmap='coolwarm', alpha=0.7)
        
        self.axes.set_xlabel("球速（px/s）", color='white')
        self.axes.set_ylabel("击球高度（px，自底向上）", color='white')
        self.axes.set_title("球速-击球高度分布", color='white')
        self.axes.tick_params(colors='white')
        self.axes.grid(True, color='#333')
        
        self.draw()

class HeatmapChart(CourtMapBase):
    def plot(self, x, y, title="热力图"):
        self.axes.cla()
        self.draw_court()
        
        if len(x) > 10:
            # KDE plot
            sns.kdeplot(x=x, y=y, ax=self.axes, fill=True, cmap='inferno', alpha=0.8, levels=20, thresh=0.05)
            # Scatter on top for density
            self.axes.scatter(x, y, color='white', s=1, alpha=0.3)
        
        self.axes.set_title(title, color='white')
        self.draw()

class ShotTypePie(MplCanvas):
    def plot(self, type_counts, title="击球类型"):
        self.axes.cla()
        
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        
        # Filter small
        total = sum(sizes)
        labels = [l for l, s in zip(labels, sizes) if s/total > 0.02]
        sizes = [s for s in sizes if s/total > 0.02]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
        
        wedges, texts, autotexts = self.axes.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            startangle=90, colors=colors,
            textprops=dict(color="w")
        )
        
        self.axes.set_title(title, color='white')
        self.draw()

class PhysicalKPI(MplCanvas):
    def plot(self, kpis: dict, zones: dict, title="运动表现指标"):
        self.axes.cla()
        self.axes.axis('off')
        y0 = 0.9
        dy = 0.1
        items = [
            ("平均速度", f"{kpis.get('avg_speed',0):.1f}"),
            ("95%分位速度", f"{kpis.get('p95_speed',0):.1f}"),
            ("最大速度", f"{kpis.get('max_speed',0):.1f}"),
            ("加速度峰值", f"{kpis.get('accel_peak',0):.2f}"),
            ("加速度均值", f"{kpis.get('accel_mean',0):.2f}"),
            ("前场占比", f"{zones.get('front',0)*100:.1f}%"),
            ("后场占比", f"{zones.get('back',0)*100:.1f}%"),
            ("左/右占比", f"{zones.get('left',0)*100:.1f}% / {zones.get('right',0)*100:.1f}%"),
            ("近网攻势指数", f"{zones.get('net_aggr',0):.2f}")
        ]
        self.axes.text(0.02, 0.98, title, color='white', fontsize=12, va='top')
        for i, (k, v) in enumerate(items):
            self.axes.text(0.05, y0 - i*dy, f"{k}: {v}", color='#00ffcc', fontsize=10)
        self.draw()

class SpeedHistogram(MplCanvas):
    def plot(self, series_df, title="速度分布"):
        self.axes.cla()
        s = series_df['speed'].dropna()
        if s.empty:
            self.axes.text(0.5, 0.5, "暂无数据", ha='center', color='white')
            self.draw()
            return
        sns.histplot(s, bins=30, kde=True, ax=self.axes, color='#4d79ff', alpha=0.7)
        self.axes.set_title(title, color='white')
        self.axes.set_xlabel("速度", color='white')
        self.axes.set_ylabel("频数", color='white')
        self.axes.tick_params(colors='white')
        self.draw()

class AccelTimeline(MplCanvas):
    def plot(self, accel_df, title="加速度时间序列"):
        self.axes.cla()
        if accel_df.empty:
            self.axes.text(0.5, 0.5, "暂无数据", ha='center', color='white')
            self.draw()
            return
        t = accel_df['time_seconds']
        a = accel_df['accel']
        self.axes.plot(t, a, color='#ffcc00', linewidth=1.2)
        thr = np.nanpercentile(np.abs(a), 90) if np.isfinite(a).all() else 0.0
        mask = np.abs(a) >= thr
        self.axes.scatter(t[mask], a[mask], s=15, color='#ff4d4d', alpha=0.8)
        self.axes.set_title(title, color='white')
        self.axes.set_xlabel("时间（秒）", color='white')
        self.axes.set_ylabel("加速度", color='white')
        self.axes.grid(True, color='#333', alpha=0.3)
        self.axes.tick_params(colors='white')
        self.draw()

class CoverageQuantile(CourtMapBase):
    def plot(self, x, y, title="覆盖分位等值线"):
        self.axes.cla()
        self.draw_court()
        if len(x) < 10:
            self.axes.text(0.5, 0.5, "数据不足", color='white')
            self.draw()
            return
        sns.kdeplot(x=x, y=y, ax=self.axes, fill=False, cmap='viridis', levels=[0.2, 0.5, 0.8])
        self.axes.set_title(title, color='white')
        self.draw()

class BarycenterEllipse(CourtMapBase):
    def plot(self, x, y, cov_info: dict, title="站位质心与稳定性"):
        self.axes.cla()
        self.draw_court()
        if len(x) >= 10:
            self.axes.scatter(x, y, s=2, color='white', alpha=0.2)
        cx = cov_info.get('cx', 0.0)
        cy = cov_info.get('cy', 0.0)
        var_x = max(cov_info.get('var_x', 0.0), 1e-3)
        var_y = max(cov_info.get('var_y', 0.0), 1e-3)
        rx = np.sqrt(var_x)
        ry = np.sqrt(var_y)
        theta = 0.0
        ang = np.linspace(0, 2*np.pi, 100)
        ex = cx + rx*np.cos(ang)
        ey = cy + ry*np.sin(ang)
        self.axes.plot(ex, ey, color='#00ffcc', alpha=0.8)
        self.axes.scatter([cx], [cy], color='#ffcc00', s=40)
        self.axes.set_title(title, color='white')
        self.draw()

class MomentumChart(MplCanvas):
    def plot(self, rallies):
        self.axes.cla()
        
        # Calculate momentum: (Hit Count * Duration) as proxy for intensity
        x = range(1, len(rallies) + 1)
        y = [r.hit_count * r.duration_sec for r in rallies] # Intensity
        
        # Color by hit count
        colors = [r.hit_count for r in rallies]
        
        scatter = self.axes.scatter(x, y, c=colors, cmap='viridis', s=50, zorder=2)
        self.axes.plot(x, y, color='grey', alpha=0.5, zorder=1)
        
        self.axes.set_xlabel("Rally Sequence", color='white')
        self.axes.set_ylabel("Intensity (Hits * Duration)", color='white')
        self.axes.tick_params(colors='white')
        self.axes.grid(True, color='#333')
        
        # Add colorbar
        # cbar = self.fig.colorbar(scatter, ax=self.axes)
        # cbar.ax.yaxis.set_tick_params(color='white')
        
        self.draw()

class SpeedHeightScatter(MplCanvas):
    def plot(self, df):
        self.axes.cla()
        # x: speed, y: y-coordinate (height proxy in 2D image)
        # In image coords, smaller y is higher. So we invert y.
        
        # Filter hits
        hits = df[df['is_hit'] == 1]
        if hits.empty: return

        x = hits['ball_speed']
        y = hits['ball_y'] # Pixel height (0 is top)
        
        # Invert Y to show "Height" (0 at bottom) - approx
        # Assuming 720p
        y_height = 720 - y
        
        scatter = self.axes.scatter(x, y_height, c=hits['hit_player'], cmap='coolwarm', alpha=0.7)
        
        self.axes.set_xlabel("Ball Speed (px/s)", color='white')
        self.axes.set_ylabel("Hit Height (px from bottom)", color='white')
        self.axes.set_title("Speed vs Height Distribution", color='white')
        self.axes.tick_params(colors='white')
        self.axes.grid(True, color='#333')
        
        self.draw()
