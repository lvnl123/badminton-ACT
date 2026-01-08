from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTabWidget, QLabel, QPushButton, QListWidget, QComboBox, QApplication, 
                               QListWidgetItem, QSplitter, QGroupBox, QGridLayout, 
                               QProgressDialog, QMessageBox)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QColor
from .engine import MatchEngine
from .panels import (RadarChart, HeatmapChart, ShotTypePie, FeatureCourt3D, 
                     SpeedHeightScatter, TransitionChordChart, TransitionHeatmap, ThemeRiverChart, 
                     VoronoiMap, HitPoint3D, LoadChart, PhysicalKPI, SpeedHistogram, 
                     AccelTimeline, CoverageQuantile, BarycenterEllipse)
from .arena import Arena3D

class MetricCard(QGroupBox):
    def __init__(self, title, value, unit="", parent=None):
        super().__init__(parent)
        self.setTitle(title)
        layout = QVBoxLayout(self)
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ffcc;")
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        if unit:
            unit_lbl = QLabel(unit)
            unit_lbl.setAlignment(Qt.AlignCenter)
            unit_lbl.setStyleSheet("color: #aaaaaa;")
            layout.addWidget(unit_lbl)

class MatchReviewWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TrackNetV3 专业比赛复盘系统")
        self.resize(1600, 900)
        
        self.engine = None
        
        self.central_widget = QTabWidget()
        container = QWidget()
        root_layout = QVBoxLayout(container)
        top_bar = QHBoxLayout()
        top_bar.addStretch(1)
        self.dataset_combo = QComboBox(container)
        self.dataset_combo.setMinimumWidth(200)
        self.dataset_combo.addItem("请选择数据集")
        for name, path in self._scan_results():
            self.dataset_combo.addItem(name, path)
        top_bar.addWidget(self.dataset_combo)
        refresh_btn = QPushButton("刷新", container)
        refresh_btn.setToolTip("重新扫描结果目录并更新列表")
        top_bar.addWidget(refresh_btn)
        root_layout.addLayout(top_bar)
        root_layout.addWidget(self.central_widget, 1)
        self.setCentralWidget(container)
        
        self._init_tabs()
        self._apply_dark_theme()
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_selected)
        refresh_btn.clicked.connect(self._refresh_dataset_list)

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #aaa; padding: 10px 20px; }
            QTabBar::tab:selected { background: #555; color: #fff; font-weight: bold; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
            QListWidget { background-color: #252525; border: 1px solid #444; }
            QListWidget::item:selected { background-color: #00ffcc; color: #000; }
        """)

    def _init_tabs(self):
        # 1. Dashboard
        self.dashboard_tab = QWidget()
        self.central_widget.addTab(self.dashboard_tab, "比赛概况 (Dashboard)")
        
        # 2. Rally Analysis
        self.rally_tab = QWidget()
        self.central_widget.addTab(self.rally_tab, "战术复盘 (Tactical)")
        
        # 3. Player Stats
        self.player_tab = QWidget()
        self.central_widget.addTab(self.player_tab, "球员表现 (Physical)")
        
        # 4. Tech Stats
        self.tech_tab = QWidget()
        self.central_widget.addTab(self.tech_tab, "技术统计 (Technical)")

        # 5. Deep Tactics
        self.deep_tab = QWidget()
        self.central_widget.addTab(self.deep_tab, "深度战术 (Deep Tactics)")
        
        # 6. Physio Load
        self.load_tab = QWidget()
        self.central_widget.addTab(self.load_tab, "体能负荷 (Load)")

    def load_match(self, folder_path):
        try:
            # Show loading
            progress = QProgressDialog("正在进行深度数据挖掘...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("加载比赛数据")
            progress.setMinimumDuration(0)
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            progress.setStyleSheet("QProgressDialog { background-color: #2b2b2b; color: #ffffff; } QLabel { color: #ffffff; }")
            progress.show()
            progress.setValue(10)
            QApplication.processEvents()
            
            self.engine = MatchEngine(folder_path)
            self.engine.load_data()
            
            progress.setValue(50)
            progress.setLabelText("生成可视化图表...")
            QApplication.processEvents()
            
            self._clear_tab(self.dashboard_tab)
            self._clear_tab(self.rally_tab)
            self._clear_tab(self.player_tab)
            self._clear_tab(self.tech_tab)
            self._clear_tab(self.deep_tab)
            self._clear_tab(self.load_tab)
            self._build_dashboard()
            self._build_rally_page()
            self._build_player_page()
            self._build_tech_page()
            self._build_deep_page()
            self._build_load_page()
            
            progress.setValue(100)
            QApplication.processEvents()
            progress.close()
            
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法加载比赛数据:\n{str(e)}")
    def _scan_results(self):
        from pathlib import Path
        res = []
        root = Path(__file__).resolve().parents[2] / "results"
        if root.exists():
            for d in sorted(root.iterdir()):
                if d.is_dir():
                    csvs = list(d.glob("*_data.csv"))
                    if csvs:
                        res.append((d.name, str(d)))
        return res
    def _on_dataset_selected(self, idx):
        if idx <= 0:
            return
        path = self.dataset_combo.itemData(idx)
        if isinstance(path, str) and path:
            self.load_match(path)
    def _refresh_dataset_list(self):
        current_text = self.dataset_combo.currentText()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItem("请选择数据集")
        items = list(self._scan_results())
        for name, path in items:
            self.dataset_combo.addItem(name, path)
        # try to keep previous selection if still exists
        idx = self.dataset_combo.findText(current_text)
        if idx >= 0:
            self.dataset_combo.setCurrentIndex(idx)
        else:
            self.dataset_combo.setCurrentIndex(0)
        self.dataset_combo.blockSignals(False)
    def _clear_tab(self, tab):
        lay = tab.layout()
        if not lay:
            return
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _build_dashboard(self):
        layout = self.dashboard_tab.layout() or QVBoxLayout(self.dashboard_tab)
        
        # Top: KPI Cards
        kpi_layout = QHBoxLayout()
        
        total_rallies = len(self.engine.rallies)
        total_hits = len(self.engine.hits)
        avg_rally_len = total_hits / total_rallies if total_rallies else 0
        total_dist = (self.engine.global_player_stats[1]['dist'] + self.engine.global_player_stats[2]['dist']) / 100 # px to m approx
        
        kpi_layout.addWidget(MetricCard("总回合数", str(total_rallies)))
        kpi_layout.addWidget(MetricCard("总击球数", str(total_hits)))
        kpi_layout.addWidget(MetricCard("平均拍数", f"{avg_rally_len:.1f}"))
        kpi_layout.addWidget(MetricCard("总跑动估算(m)", f"{total_dist:.0f}"))
        
        layout.addLayout(kpi_layout)
        
        # Middle: Radar & 3D Court
        mid_layout = QHBoxLayout()
        
        radar_group = QGroupBox("能力六维图")
        radar_layout = QVBoxLayout(radar_group)
        self.radar_chart = RadarChart()
        radar_layout.addWidget(self.radar_chart)
        
        # Update Radar
        p1_radar = self.engine.get_player_radar_data(1)
        p2_radar = self.engine.get_player_radar_data(2)
        self.radar_chart.plot(p1_radar, p2_radar, list(p1_radar.keys()))
        
        court3d_group = QGroupBox("击球落点-高度 3D 战术图")
        court3d_layout = QVBoxLayout(court3d_group)
        self.court3d_chart = FeatureCourt3D()
        court3d_layout.addWidget(self.court3d_chart, alignment=Qt.AlignCenter)
        self.court3d_chart.plot(self.engine)
        # Controls
        ctrl_bar = QHBoxLayout()
        color_speed_btn = QPushButton("按球速着色", court3d_group)
        color_type_btn = QPushButton("按类型着色", court3d_group)
        hull_toggle_btn = QPushButton("开/关包络", court3d_group)
        line_toggle_btn = QPushButton("开/关连线", court3d_group)
        zoom_in_btn = QPushButton("缩放 +", court3d_group)
        zoom_out_btn = QPushButton("缩放 -", court3d_group)
        zoom_reset_btn = QPushButton("重置缩放", court3d_group)
        ctrl_bar.addWidget(color_speed_btn)
        ctrl_bar.addWidget(color_type_btn)
        ctrl_bar.addSpacing(10)
        ctrl_bar.addWidget(hull_toggle_btn)
        ctrl_bar.addWidget(line_toggle_btn)
        ctrl_bar.addSpacing(10)
        ctrl_bar.addWidget(zoom_in_btn)
        ctrl_bar.addWidget(zoom_out_btn)
        ctrl_bar.addWidget(zoom_reset_btn)
        ctrl_bar.addStretch(1)
        court3d_layout.addLayout(ctrl_bar)
        # Bind
        color_speed_btn.clicked.connect(lambda: self.court3d_chart.set_color_mode('speed'))
        color_type_btn.clicked.connect(lambda: self.court3d_chart.set_color_mode('type'))
        hull_toggle_btn.clicked.connect(lambda: self.court3d_chart.set_show_hulls(not self.court3d_chart._show_hulls))
        line_toggle_btn.clicked.connect(lambda: self.court3d_chart.set_show_lines(not self.court3d_chart._show_lines))
        zoom_in_btn.clicked.connect(lambda: self.court3d_chart.zoom(1.2))
        zoom_out_btn.clicked.connect(lambda: self.court3d_chart.zoom(1/1.2))
        zoom_reset_btn.clicked.connect(self.court3d_chart.reset_zoom)
        
        mid_layout.addWidget(radar_group, 1)
        mid_layout.addWidget(court3d_group, 2)
        
        layout.addLayout(mid_layout, 2)

    def _build_rally_page(self):
        layout = self.rally_tab.layout() or QHBoxLayout(self.rally_tab)
        
        # Left: Rally List
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("回合列表"))
        
        self.rally_list = QListWidget()
        for r in self.engine.rallies:
            item = QListWidgetItem(f"Rally {r.id}: {r.hit_count} hits, {r.duration_sec:.1f}s")
            item.setData(Qt.UserRole, r.id)
            self.rally_list.addItem(item)
            
        self.rally_list.currentRowChanged.connect(self._on_rally_selected)
        left_layout.addWidget(self.rally_list)
        
        layout.addWidget(left_panel, 1)
        
        # Right: 3D Arena & Stats
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.arena = Arena3D()
        right_layout.addWidget(self.arena, 2)
        
        self.rally_speed_chart = SpeedHeightScatter() # Reusing for rally stats if needed, or create new
        # Let's put text details here instead
        self.rally_info_lbl = QLabel("选择一个回合查看详情")
        self.rally_info_lbl.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.rally_info_lbl)
        
        layout.addWidget(right_panel, 3)

    def _on_rally_selected(self, row):
        if row < 0: return
        rally_id = self.rally_list.item(row).data(Qt.UserRole)
        rally = next((r for r in self.engine.rallies if r.id == rally_id), None)
        
        if rally:
            self.arena.plot_rally(rally.trajectory, rally.strokes)
            
            info = f"""
            <h3>Rally {rally.id}</h3>
            <p>持续时间: {rally.duration_sec:.2f} 秒 | 击球数: {rally.hit_count}</p>
            <p>Player 1 跑动: {rally.player_stats[1]['dist']:.1f} | 平均速度: {rally.player_stats[1]['avg_speed']:.1f}</p>
            <p>Player 2 跑动: {rally.player_stats[2]['dist']:.1f} | 平均速度: {rally.player_stats[2]['avg_speed']:.1f}</p>
            """
            self.rally_info_lbl.setText(info)

    def _build_player_page(self):
        layout = self.player_tab.layout() or QGridLayout(self.player_tab)
        
        p1_grp = QGroupBox("Player 1 覆盖热区")
        p1_layout = QGridLayout(p1_grp)
        kpi1 = PhysicalKPI()
        p1_map = HeatmapChart()
        sp1 = SpeedHistogram()
        ac1 = AccelTimeline()
        q1 = CoverageQuantile()
        bc1 = BarycenterEllipse()
        # Top row: 左侧KPI，右侧移动热区
        p1_layout.addWidget(kpi1, 0, 0, 1, 1)
        p1_layout.addWidget(p1_map, 0, 1, 1, 1)
        # Second row: 左速度分布，右加速度时间序列
        p1_layout.addWidget(sp1, 1, 0, 1, 1)
        p1_layout.addWidget(ac1, 1, 1, 1, 1)
        # Third row: 左覆盖分位线，右质心椭圆
        p1_layout.addWidget(q1, 2, 0, 1, 1)
        p1_layout.addWidget(bc1, 2, 1, 1, 1)
        # Stretch to尽可能大
        p1_layout.setColumnStretch(0, 1)
        p1_layout.setColumnStretch(1, 1)
        p1_layout.setRowStretch(0, 2)
        p1_layout.setRowStretch(1, 2)
        p1_layout.setRowStretch(2, 2)

        p2_grp = QGroupBox("Player 2 覆盖热区")
        p2_layout = QGridLayout(p2_grp)
        kpi2 = PhysicalKPI()
        p2_map = HeatmapChart()
        sp2 = SpeedHistogram()
        ac2 = AccelTimeline()
        q2 = CoverageQuantile()
        bc2 = BarycenterEllipse()
        # Top row: 左侧KPI，右侧移动热区
        p2_layout.addWidget(kpi2, 0, 0, 1, 1)
        p2_layout.addWidget(p2_map, 0, 1, 1, 1)
        # Second row: 左速度分布，右加速度时间序列
        p2_layout.addWidget(sp2, 1, 0, 1, 1)
        p2_layout.addWidget(ac2, 1, 1, 1, 1)
        # Third row: 左覆盖分位线，右质心椭圆
        p2_layout.addWidget(q2, 2, 0, 1, 1)
        p2_layout.addWidget(bc2, 2, 1, 1, 1)
        # Stretch to尽可能大
        p2_layout.setColumnStretch(0, 1)
        p2_layout.setColumnStretch(1, 1)
        p2_layout.setRowStretch(0, 2)
        p2_layout.setRowStretch(1, 2)
        p2_layout.setRowStretch(2, 2)
        
        # Data
        p1_x = self.engine.df['player1_joint0_x'].dropna().tolist()
        p1_y = self.engine.df['player1_joint0_y'].dropna().tolist()
        p1_map.plot(p1_x, p1_y, "球员1移动热区")
        kpi1.plot(self.engine.get_physical_kpis(1), self.engine.get_player_zone_ratios(1), "球员1 运动表现指标")
        sp1.plot(self.engine.get_speed_series(1), "球员1 速度分布")
        ac1.plot(self.engine.get_accel_series(1), "球员1 加速度时间序列")
        q1.plot(p1_x, p1_y, "球员1 覆盖分位等值线")
        bc1.plot(p1_x, p1_y, self.engine.get_barycenter_cov(1), "球员1 站位质心与稳定性")
        
        p2_x = self.engine.df['player2_joint0_x'].dropna().tolist()
        p2_y = self.engine.df['player2_joint0_y'].dropna().tolist()
        p2_map.plot(p2_x, p2_y, "球员2移动热区")
        kpi2.plot(self.engine.get_physical_kpis(2), self.engine.get_player_zone_ratios(2), "球员2 运动表现指标")
        sp2.plot(self.engine.get_speed_series(2), "球员2 速度分布")
        ac2.plot(self.engine.get_accel_series(2), "球员2 加速度时间序列")
        q2.plot(p2_x, p2_y, "球员2 覆盖分位等值线")
        bc2.plot(p2_x, p2_y, self.engine.get_barycenter_cov(2), "球员2 站位质心与稳定性")
        
        layout.addWidget(p1_grp, 0, 0)
        layout.addWidget(p2_grp, 0, 1)

    def _build_tech_page(self):
        layout = self.tech_tab.layout() or QGridLayout(self.tech_tab)
        
        # Shot Types P1
        p1_pie = ShotTypePie()
        p1_pie.plot(self.engine.global_player_stats[1]['type_counts'], "Player 1 击球类型")
        layout.addWidget(p1_pie, 0, 0)
        
        # Shot Types P2
        p2_pie = ShotTypePie()
        p2_pie.plot(self.engine.global_player_stats[2]['type_counts'], "Player 2 击球类型")
        layout.addWidget(p2_pie, 0, 1)
        
        # Speed vs Height
        scatter = SpeedHeightScatter()
        scatter.plot(self.engine.df)
        layout.addWidget(scatter, 1, 0, 1, 2)

    def _build_deep_page(self):
        layout = self.deep_tab.layout() or QGridLayout(self.deep_tab)
        
        # 1. ThemeRiver Flow
        river = ThemeRiverChart()
        river.plot(self.engine.get_theme_river_data(window_sec=2.0, player_id=None), "战术类型流图")
        layout.addWidget(river, 0, 0, 1, 2)
        
        # 2. Transition Matrix (P1)
        trans_p1 = TransitionHeatmap()
        trans_p1.plot(self.engine.get_transition_matrix(1))
        layout.addWidget(trans_p1, 1, 0)
        
        # 3. Voronoi Space Control (Sample from last frame)
        # Just sample one frame for demo
        voronoi = VoronoiMap()
        last_row = self.engine.df.iloc[-100] if len(self.engine.df) > 100 else self.engine.df.iloc[0]
        p1_pos = (last_row['player1_joint0_x'], last_row['player1_joint0_y'])
        p2_pos = (last_row['player2_joint0_x'], last_row['player2_joint0_y'])
        voronoi.plot(p1_pos, p2_pos)
        layout.addWidget(voronoi, 1, 1)

    def _build_load_page(self):
        layout = self.load_tab.layout() or QGridLayout(self.load_tab)
        
        # 1. Load Chart
        load_chart = LoadChart()
        load_chart.plot(self.engine.df)
        layout.addWidget(load_chart, 0, 0, 1, 2)
        
        # 2. 3D Impact Analysis
        impact_3d = HitPoint3D()
        impact_3d.plot(self.engine.df)
        layout.addWidget(impact_3d, 1, 0, 1, 2)
