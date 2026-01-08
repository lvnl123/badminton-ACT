from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt
from PySide6.QtGui import QColor
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableView,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHeaderView,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_pyside6.widgets.pipeline_runner import PipelineConfig
from ui_pyside6.widgets.pipeline_worker import PipelineWorker, WorkerThread
from ui_pyside6.widgets.data_models import DataFrameModel
from ui_pyside6.widgets.simple_plot import (
    DensityBubbleMap,
    MetricCard,
    ProDistributionChart,
    SimpleBarChart,
    SimpleLinePlot,
    TerritoryScatterPlot,
    TimelineMarkers,
)
from ui_pyside6.widgets.video_player import VideoPlayer
from ui_pyside6.match_review.review_window import MatchReviewWindow


def _apply_style(app: QApplication):
    app.setStyleSheet(
        """
        QWidget{font-family:Segoe UI,Microsoft YaHei;font-size:12px;color:#e5e7eb;background:#0b0f14;}
        QMainWindow::separator{background:#111827;width:1px;height:1px;}
        QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox{background:#0f1216;border:1px solid #2a2f3a;border-radius:8px;padding:8px;}
        QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus,QComboBox:focus{border:1px solid #3b82f6;}
        QPushButton{background:#111827;border:1px solid #2a2f3a;border-radius:10px;padding:10px 12px;}
        QPushButton:hover{background:#0f172a;}
        QPushButton:disabled{color:#6b7280;background:#0b0f14;border:1px solid #1f2937;}
        QGroupBox{border:1px solid #1f2937;border-radius:12px;margin-top:10px;padding:10px;}
        QGroupBox:title{subcontrol-origin:margin;left:12px;top:-2px;padding:0 6px;color:#93c5fd;}
        QTabWidget::pane{border:1px solid #1f2937;border-radius:10px;padding:0px;}
        QTabBar::tab{background:#0f1216;border:1px solid #1f2937;border-bottom:none;border-top-left-radius:8px;border-top-right-radius:8px;padding:8px 12px;margin-right:4px;}
        QTabBar::tab:selected{background:#111827;border:1px solid #334155;}
        QHeaderView::section{background:#0f1216;border:1px solid #1f2937;padding:6px;}
        QTableView{gridline-color:#1f2937;selection-background-color:#1e3a8a;border:1px solid #1f2937;border-radius:10px;}
        QTextEdit{background:#0f1216;border:1px solid #1f2937;border-radius:10px;}
        QProgressBar{background:#0f1216;border:1px solid #1f2937;border-radius:10px;text-align:center;height:18px;}
        QProgressBar::chunk{background:#22c55e;border-radius:10px;}
        """
    )


class StepperWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps = ["球场/球网检测", "羽毛球检测", "姿态检测", "事件检测", "击球类型识别", "合成可视化", "导出数据"]
        self._current = ""
        self._done: set[str] = set()
        self._failed = False
        self.setMinimumHeight(60)

    def reset(self):
        self._current = ""
        self._done.clear()
        self._failed = False
        self.update()

    def set_current(self, step_name: str):
        if self._current and self._current != step_name:
            self._done.add(self._current)
        self._current = step_name
        self.update()

    def set_finished(self, ok: bool):
        self._failed = not ok
        if ok:
            self._done.update(self._steps)
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.fillRect(r, QColor("#0f1216"))
        p.setPen(QPen(QColor("#1f2937"), 1))
        p.drawRoundedRect(r, 12, 12)

        if not self._steps:
            return

        left = r.left() + 14
        right = r.right() - 14
        y = r.center().y()
        n = len(self._steps)
        if n <= 1:
            return

        p.setPen(QPen(QColor("#334155"), 2))
        p.drawLine(left, y, right, y)

        step_w = (right - left) / (n - 1)
        for i, name in enumerate(self._steps):
            cx = left + int(i * step_w)
            is_done = name in self._done
            is_current = name == self._current
            color = QColor("#6b7280")
            if is_done:
                color = QColor("#22c55e")
            elif is_current:
                color = QColor("#3b82f6")
            if self._failed and is_current:
                color = QColor("#ef4444")

            radius = 7 if is_current else 6
            p.setBrush(color)
            p.setPen(QPen(QColor("#0b0f14"), 2))
            p.drawEllipse(cx - radius, y - radius, radius * 2, radius * 2)

        p.setPen(QPen(QColor("#9ca3af")))
        p.drawText(r.adjusted(12, 6, -12, -6), Qt.AlignLeft | Qt.AlignTop, "步骤流")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrackNetV3_Attention - 比赛视频训练分析工作台")
        self.resize(1650, 820)
        self._cwd = os.getcwd()
        self._review_window = None

        self._worker_thread: Optional[WorkerThread] = None
        self._worker: Optional[PipelineWorker] = None
        self._current_outputs: Optional[Dict[str, Any]] = None
        self._events_raw_df = pd.DataFrame()
        self._current_df = pd.DataFrame()

        self._input_player = VideoPlayer()
        self._preview_player = VideoPlayer()
        self._output_player = VideoPlayer()
        self._compare_input = VideoPlayer()
        self._compare_output = VideoPlayer()
        self._compare_sync_cb = QCheckBox("同步对比播放")
        self._compare_sync_cb.setChecked(True)
        self._compare_follow_combo = QComboBox()
        self._compare_follow_combo.addItems(["输入驱动输出", "输出驱动输入"])
        self._compare_follow_combo.setCurrentIndex(0)
        self._compare_guard = False

        self._log = QTextEdit()
        self._log.setReadOnly(True)

        self._status_step = QLabel("就绪")
        self._status_progress = QLabel("0%")
        self._elapsed_label = QLabel("耗时: --")
        self._eta_label = QLabel("预计剩余: --")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._stepper = StepperWidget()

        self._overall_label = QLabel("总体进度: 0%")
        self._overall_label.setStyleSheet("QLabel{color:#a7f3d0;}")

        self._video_path_edit = QLineEdit()
        self._video_path_edit.setReadOnly(True)
        self._result_dir_edit = QLineEdit(str(ROOT / "results"))

        self._result_combo = QComboBox()
        self._refresh_results_btn = QPushButton("刷新结果")
        self._load_result_btn = QPushButton("加载结果")

        self._device_combo = QComboBox()
        self._device_combo.addItems(["cuda", "cpu"])

        self._pose_model_combo = QComboBox()
        self._pose_model_combo.addItems(["rtmpose-t", "rtmpose-s", "rtmpose-m", "rtmpose-l"])
        self._pose_model_combo.setCurrentText("rtmpose-m")

        self._use_court_cb = QCheckBox("启用球场/球网检测与区域高亮")
        self._use_court_cb.setChecked(True)

        self._model_path_edit = QLineEdit(str(ROOT / "models" / "ball_track_attention.pt"))
        self._court_model_edit = QLineEdit(str(ROOT / "models" / "court_kpRCNN.pth"))
        self._net_model_edit = QLineEdit(str(ROOT / "models" / "net_kpRCNN.pth"))

        self._num_frames_spin = QSpinBox()
        self._num_frames_spin.setRange(1, 9)
        self._num_frames_spin.setValue(3)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1.0)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setValue(0.5)

        self._traj_len_spin = QSpinBox()
        self._traj_len_spin.setRange(1, 60)
        self._traj_len_spin.setValue(10)

        self._court_interval_spin = QSpinBox()
        self._court_interval_spin.setRange(1, 300)
        self._court_interval_spin.setValue(30)

        self._emit_every_spin = QSpinBox()
        self._emit_every_spin.setRange(1, 60)
        self._emit_every_spin.setValue(5)

        self._viz_emit_every_spin = QSpinBox()
        self._viz_emit_every_spin.setRange(1, 60)
        self._viz_emit_every_spin.setValue(2)

        self._run_btn = QPushButton("开始训练分析")
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setEnabled(False)
        self._open_video_btn = QPushButton("导入视频")
        self._open_output_btn = QPushButton("打开输出目录")

        self._csv_model = DataFrameModel()
        self._csv_view = QTableView()
        self._csv_view.setModel(self._csv_model)
        self._csv_view.setSortingEnabled(True)

        self._events_model = DataFrameModel()
        self._events_view = QTableView()
        self._events_view.setModel(self._events_model)
        self._events_view.setSortingEnabled(True)
        self._events_view.horizontalHeader().setStretchLastSection(True)
        self._events_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._events_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._events_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)

        self._event_search = QLineEdit()
        self._event_search.setPlaceholderText("搜索事件（frame/player/stroke/type 等）")
        self._event_player_filter = QComboBox()
        self._event_player_filter.addItems(["全部"])
        self._event_stroke_filter = QComboBox()
        self._event_stroke_filter.addItems(["全部"])
        self._event_reset_btn = QPushButton("清空筛选")
        self._events_page = QWidget()

        self._speed_plot = SimpleLinePlot()
        self._ball_y_plot = SimpleLinePlot()
        self._hit_count_plot = SimpleLinePlot()
        self._speed_hist = ProDistributionChart()
        self._hit_interval_hist = ProDistributionChart()
        self._hit_height_hist = ProDistributionChart()
        self._player_speed_hist = ProDistributionChart()
        self._timeline = TimelineMarkers()
        self._stroke_bar = SimpleBarChart()
        self._heatmap = DensityBubbleMap()
        self._density_source_combo = QComboBox()
        self._density_source_combo.addItems(["可见帧", "全部帧", "仅击球帧", "仅选手0击球帧", "仅选手1击球帧"])
        self._density_bins_combo = QComboBox()
        self._density_bins_combo.addItems(["粗(24x14)", "中(44x24)", "细(72x40)"])
        self._density_bins_combo.setCurrentText("中(44x24)")
        self._density_show_current_cb = QCheckBox("显示当前帧位置")
        self._density_show_current_cb.setChecked(True)
        self._density_export_btn = QPushButton("导出密度图…")

        self._m_hits = MetricCard("击球次数")
        self._m_duration = MetricCard("时长")
        self._m_speed_avg = MetricCard("平均球速(像素/秒)")
        self._m_speed_max = MetricCard("最大球速(像素/秒)")
        self._m_visible = MetricCard("可见率")
        self._m_output = MetricCard("输出目录")

        self._p1_map = TerritoryScatterPlot()
        self._p2_map = TerritoryScatterPlot()
        self._p_dist_plot = SimpleLinePlot()
        self._p1_speed_plot = SimpleLinePlot()
        self._p2_speed_plot = SimpleLinePlot()

        self._build_ui()
        self._build_actions()
        self._connect()

    def _build_actions(self):
        open_video = QAction("导入视频", self)
        open_video.triggered.connect(self._choose_video)
        self.menuBar().addAction(open_video)

        open_out = QAction("打开输出目录", self)
        open_out.triggered.connect(self._open_output_dir)
        self.menuBar().addAction(open_out)

        review_action = QAction("比赛复盘数据", self)
        review_action.triggered.connect(self._open_match_review)
        self.menuBar().addAction(review_action)

        help_menu = self.menuBar().addMenu("帮助")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        usage_action = QAction("使用说明", self)
        usage_action.triggered.connect(self._show_usage)
        help_menu.addAction(usage_action)

        export_menu = self.menuBar().addMenu("导出")
        export_overview = QAction("导出概览截图…", self)
        export_overview.triggered.connect(self._export_overview_png)
        export_menu.addAction(export_overview)

        export_events = QAction("导出当前事件表(CSV)…", self)
        export_events.triggered.connect(self._export_events_csv)
        export_menu.addAction(export_events)

        export_csv = QAction("导出当前 CSV 数据(CSV)…", self)
        export_csv.triggered.connect(self._export_csv_csv)
        export_menu.addAction(export_csv)

    def _build_ui(self):
        player_tabs = QTabWidget()
        player_tabs.addTab(self._input_player, "输入视频")
        player_tabs.addTab(self._preview_player, "检测预览")
        player_tabs.addTab(self._output_player, "输出视频")
        player_tabs.addTab(self._build_compare_view(), "对比")

        params = QGroupBox("训练分析参数")
        form = QFormLayout(params)
        form.setLabelAlignment(Qt.AlignRight)
        form.addRow("视频路径", self._video_path_edit)
        form.addRow("输出目录", self._result_dir_edit)
        form.addRow("设备", self._device_combo)
        form.addRow("Pose 模型", self._pose_model_combo)
        form.addRow("", self._use_court_cb)
        form.addRow("TrackNet 权重", self._model_path_edit)
        form.addRow("球场模型", self._court_model_edit)
        form.addRow("球网模型", self._net_model_edit)
        form.addRow("输入帧数", self._num_frames_spin)
        form.addRow("检测阈值", self._threshold_spin)
        form.addRow("轨迹长度", self._traj_len_spin)
        form.addRow("球场检测间隔", self._court_interval_spin)
        form.addRow("预览抽样间隔", self._emit_every_spin)
        form.addRow("合成预览抽样", self._viz_emit_every_spin)

        buttons = QHBoxLayout()
        buttons.addWidget(self._open_video_btn)
        buttons.addWidget(self._run_btn)
        buttons.addWidget(self._stop_btn)
        buttons.addWidget(self._open_output_btn)

        status_box = QGroupBox("运行状态")
        status_layout = QVBoxLayout(status_box)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("当前步骤:"))
        row1.addWidget(self._status_step, 1)
        row1.addWidget(self._overall_label)
        status_layout.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("进度:"))
        row2.addWidget(self._status_progress, 1)
        row2.addWidget(self._elapsed_label)
        row2.addWidget(self._eta_label)
        status_layout.addLayout(row2)
        status_layout.addWidget(self._progress_bar)
        status_layout.addWidget(self._stepper)

        result_box = QGroupBox("结果浏览")
        result_form = QFormLayout(result_box)
        result_form.setLabelAlignment(Qt.AlignRight)
        result_form.addRow("结果集", self._result_combo)
        result_btns = QWidget()
        result_btns_layout = QHBoxLayout(result_btns)
        result_btns_layout.setContentsMargins(0, 0, 0, 0)
        result_btns_layout.addWidget(self._refresh_results_btn)
        result_btns_layout.addWidget(self._load_result_btn)
        result_form.addRow("", result_btns)

        right_tabs = QTabWidget()
        right_tabs.addTab(params, "参数")
        right_tabs.addTab(self._log, "日志")

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(right_tabs, 1)
        right_layout.addWidget(status_box)
        right_layout.addWidget(result_box)
        right_layout.addLayout(buttons)

        data_tabs = QTabWidget()
        data_tabs.addTab(self._build_overview(), "概览")
        data_tabs.addTab(self._events_page, "击球事件")
        data_tabs.addTab(self._csv_view, "CSV 数据")
        data_tabs.addTab(self._speed_plot, "球速曲线")
        data_tabs.addTab(self._ball_y_plot, "球高度(像素)")
        data_tabs.addTab(self._hit_count_plot, "累计击球数")
        data_tabs.addTab(self._build_distributions(), "分布")
        data_tabs.addTab(self._build_players(), "选手")

        left_split = QSplitter(Qt.Vertical)
        left_split.addWidget(player_tabs)
        left_split.addWidget(data_tabs)
        left_split.setStretchFactor(0, 3)
        left_split.setStretchFactor(1, 2)

        root_split = QSplitter(Qt.Horizontal)
        root_split.addWidget(left_split)
        root_split.addWidget(right_panel)
        root_split.setStretchFactor(0, 4)
        root_split.setStretchFactor(1, 2)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._build_header())
        layout.addWidget(root_split, 1)
        self.setCentralWidget(container)

    def _build_distributions(self) -> QWidget:
        root = QWidget()
        layout = QGridLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._speed_hist.set_data([], title="球速分布(Pro)", x_label="ball_speed", color="#3b82f6")
        self._hit_interval_hist.set_data([], title="击球间隔分布(Pro)", x_label="delta_seconds", color="#22c55e")
        self._hit_height_hist.set_data([], title="击球高度分布(Pro)", x_label="hit_y_px", color="#f59e0b")
        self._player_speed_hist.set_data([], title="选手瞬时速度分布(Pro)", x_label="player_speed_px_s", color="#a78bfa")

        layout.addWidget(self._speed_hist, 0, 0)
        layout.addWidget(self._hit_interval_hist, 0, 1)
        layout.addWidget(self._hit_height_hist, 1, 0)
        layout.addWidget(self._player_speed_hist, 1, 1)

        return root

    def _build_players(self) -> QWidget:
        root = QWidget()
        layout = QGridLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._p1_map.set_points([], title="选手0覆盖(凸包)", color="#3b82f6")
        self._p2_map.set_points([], title="选手1覆盖(凸包)", color="#ef4444")
        self._p1_map.set_current_point(None)
        self._p2_map.set_current_point(None)
        self._p_dist_plot.set_series([], [], x_label="time_seconds", y_label="player_distance_px", title="选手间距")
        self._p1_speed_plot.set_series([], [], x_label="time_seconds", y_label="p0_speed_px_s", title="选手0速度")
        self._p2_speed_plot.set_series([], [], x_label="time_seconds", y_label="p1_speed_px_s", title="选手1速度")

        layout.addWidget(self._p1_map, 0, 0)
        layout.addWidget(self._p2_map, 0, 1)
        layout.addWidget(self._p_dist_plot, 1, 0, 2, 1)
        layout.addWidget(self._p1_speed_plot, 1, 1)
        layout.addWidget(self._p2_speed_plot, 2, 1)

        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        
        return root

    def _build_compare_view(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        controls.addWidget(self._compare_sync_cb)
        controls.addWidget(QLabel("跟随模式"))
        controls.addWidget(self._compare_follow_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._compare_input)
        split.addWidget(self._compare_output)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        return root

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(52)
        layout = QHBoxLayout(w)
        layout.setContentsMargins(10, 8, 10, 8)
        title = QLabel("比赛视频训练分析工作台")
        title.setStyleSheet("QLabel{font-size:16px;font-weight:600;color:#e5e7eb;}")
        sub = QLabel("TrackNetV3 + MMPose + Event + BST")
        sub.setStyleSheet("QLabel{color:#9ca3af;}")
        layout.addWidget(title)
        layout.addSpacing(12)
        layout.addWidget(sub, 1)
        file_lab = QLabel()
        file_lab.setStyleSheet("QLabel{color:#93c5fd;}")
        file_lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._header_file = file_lab
        layout.addWidget(file_lab)
        return w

    def _build_overview(self) -> QWidget:
        root = QWidget()
        self._overview_widget = root
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)

        cards = QWidget()
        grid = QGridLayout(cards)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self._m_hits.set_accent(QColor("#f59e0b"))
        self._m_duration.set_accent(QColor("#22c55e"))
        self._m_speed_avg.set_accent(QColor("#3b82f6"))
        self._m_speed_max.set_accent(QColor("#ef4444"))
        self._m_visible.set_accent(QColor("#a78bfa"))
        self._m_output.set_accent(QColor("#93c5fd"))

        grid.addWidget(self._m_hits, 0, 0)
        grid.addWidget(self._m_duration, 0, 1)
        grid.addWidget(self._m_visible, 0, 2)
        grid.addWidget(self._m_speed_avg, 1, 0)
        grid.addWidget(self._m_speed_max, 1, 1)
        grid.addWidget(self._m_output, 1, 2)

        self._timeline.set_markers([], [], title="击球时间轴(点击跳转)")
        self._heatmap.set_points([], title="球位置密度(气泡聚合)")
        self._heatmap.set_show_current_point(True)
        self._stroke_bar.set_data([], [], title="击球类型分布")

        bottom_split = QSplitter(Qt.Horizontal)
        bottom_split.addWidget(self._build_density_panel())
        bottom_split.addWidget(self._stroke_bar)
        bottom_split.setStretchFactor(0, 3)
        bottom_split.setStretchFactor(1, 2)

        layout.addWidget(cards)
        layout.addWidget(self._timeline)
        layout.addWidget(bottom_split, 1)
        return root

    def _build_density_panel(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)

        bar = QWidget()
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.addWidget(QLabel("密度来源"))
        bar_layout.addWidget(self._density_source_combo)
        bar_layout.addSpacing(8)
        bar_layout.addWidget(QLabel("网格"))
        bar_layout.addWidget(self._density_bins_combo)
        bar_layout.addSpacing(8)
        bar_layout.addWidget(self._density_show_current_cb)
        bar_layout.addStretch(1)
        bar_layout.addWidget(self._density_export_btn)

        layout.addWidget(bar)
        layout.addWidget(self._heatmap, 1)
        return root

    def _build_events_page(self) -> QWidget:
        layout = QVBoxLayout(self._events_page)
        layout.setContentsMargins(0, 0, 0, 0)

        bar = QWidget()
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.addWidget(self._event_search, 2)
        bar_layout.addWidget(QLabel("选手"))
        bar_layout.addWidget(self._event_player_filter)
        bar_layout.addWidget(QLabel("击球类型"))
        bar_layout.addWidget(self._event_stroke_filter, 1)
        bar_layout.addWidget(self._event_reset_btn)

        layout.addWidget(bar)
        layout.addWidget(self._events_view, 1)
        return self._events_page

    def _connect(self):
        self._open_video_btn.clicked.connect(self._choose_video)
        self._open_output_btn.clicked.connect(self._open_output_dir)
        self._run_btn.clicked.connect(self._start_pipeline)
        self._stop_btn.clicked.connect(self._stop_pipeline)
        self._output_player.positionChanged.connect(self._on_output_position)
        self._refresh_results_btn.clicked.connect(self._refresh_results_list)
        self._load_result_btn.clicked.connect(self._load_selected_result)
        self._timeline.markerActivated.connect(self._seek_output)
        if self._events_view.selectionModel() is not None:
            self._events_view.selectionModel().selectionChanged.connect(self._on_event_selection_changed)
        self._events_view.doubleClicked.connect(self._on_event_double_clicked)
        self._event_search.textChanged.connect(self._apply_event_filters)
        self._event_player_filter.currentIndexChanged.connect(self._apply_event_filters)
        self._event_stroke_filter.currentIndexChanged.connect(self._apply_event_filters)
        self._event_reset_btn.clicked.connect(self._reset_event_filters)

        self._compare_input.positionChanged.connect(self._on_compare_input_pos)
        self._compare_output.positionChanged.connect(self._on_compare_output_pos)
        self._density_source_combo.currentIndexChanged.connect(self._refresh_density_view)
        self._density_bins_combo.currentIndexChanged.connect(self._refresh_density_view)
        self._density_show_current_cb.toggled.connect(self._on_density_show_current_toggled)
        self._density_export_btn.clicked.connect(self._export_density_png)
        self._apply_window_ready()
        self._refresh_results_list()
        self._build_events_page()

    def _apply_window_ready(self):
        app = QApplication.instance()
        if app is not None:
            _apply_style(app)

    def _choose_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择比赛视频", str(ROOT / "videos"), "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not file_path:
            return
        self._video_path_edit.setText(file_path)
        self._header_file.setText(file_path)
        self._input_player.open(file_path)
        self._compare_input.open(file_path)
        self._preview_player.set_preview_frame(self._get_black_frame(), 0, 0)

    def _open_output_dir(self):
        text = self._result_dir_edit.text().strip()
        if not text:
            return
        p = Path(text).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        p = p.resolve(strict=False)
        p.mkdir(parents=True, exist_ok=True)
        self._result_dir_edit.setText(str(p))
        os.startfile(str(p))
        self._refresh_results_list()

    def _start_pipeline(self):
        if self._worker_thread is not None:
            return
        video_path = self._video_path_edit.text().strip()
        if not video_path:
            QMessageBox.warning(self, "提示", "请先导入视频")
            return
        result_dir_text = self._result_dir_edit.text().strip()
        if not result_dir_text:
            QMessageBox.warning(self, "提示", "请设置输出目录")
            return
        result_dir_path = Path(result_dir_text).expanduser()
        if not result_dir_path.is_absolute():
            result_dir_path = ROOT / result_dir_path
        result_dir_path = result_dir_path.resolve(strict=False)
        result_dir = str(result_dir_path)
        self._result_dir_edit.setText(result_dir)

        config = PipelineConfig(
            model_path=self._model_path_edit.text().strip(),
            num_frames=int(self._num_frames_spin.value()),
            threshold=float(self._threshold_spin.value()),
            traj_len=int(self._traj_len_spin.value()),
            device=self._device_combo.currentText(),
            pose_model=self._pose_model_combo.currentText(),
            use_court_detection=bool(self._use_court_cb.isChecked()),
            court_model_path=self._court_model_edit.text().strip(),
            net_model_path=self._net_model_edit.text().strip(),
            court_detection_interval=int(self._court_interval_spin.value()),
            pose_emit_every_n=int(self._emit_every_spin.value()),
            viz_emit_every_n=int(self._viz_emit_every_spin.value()),
            dataset="shuttleset",
            stroke_seq_len=100,
        )

        self._log.clear()
        self._status_step.setText("启动中…")
        self._status_progress.setText("0%")
        self._overall_label.setText("总体进度: 0%")
        self._elapsed_label.setText("耗时: 0s")
        self._eta_label.setText("预计剩余: --")
        self._progress_bar.setValue(0)
        self._stepper.reset()
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._run_started_ts = time.time()
        self._last_progress_ts = self._run_started_ts
        self._last_progress = 0

        self._worker = PipelineWorker(video_path, result_dir, config)
        self._worker_thread = WorkerThread(self._worker)
        self._worker.logLine.connect(self._append_log)
        self._worker.stepChanged.connect(self._on_step)
        self._worker.overallProgressChanged.connect(self._on_overall_progress)
        self._worker.previewFrame.connect(self._on_preview_frame)
        self._worker.outputsReady.connect(self._on_outputs_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker_thread.start()

    def _stop_pipeline(self):
        if self._worker is not None:
            self._worker.request_stop()
            self._append_log("已请求停止…")

    def _append_log(self, line: str):
        self._log.append(line)

    def _on_step(self, step_name: str):
        self._status_step.setText(step_name)
        self._stepper.set_current(step_name)

    def _on_overall_progress(self, p: int):
        self._overall_label.setText(f"总体进度: {p}%")
        self._status_progress.setText(f"{p}%")
        self._progress_bar.setValue(int(p))
        now = time.time()
        start = getattr(self, "_run_started_ts", None)
        if start is not None:
            elapsed = max(0.0, now - float(start))
            self._elapsed_label.setText(f"耗时: {int(elapsed)}s")
        if p > 0 and start is not None:
            elapsed = max(0.001, now - float(start))
            total_est = elapsed / (p / 100.0)
            remain = max(0.0, total_est - elapsed)
            self._eta_label.setText(f"预计剩余: {int(remain)}s")

    def _on_preview_frame(self, frame_bgr: Any):
        if frame_bgr is None:
            return
        self._preview_player.set_preview_frame(frame_bgr)

    def _on_outputs_ready(self, outputs: Dict[str, Any]):
        self._current_outputs = outputs
        self._refresh_results_list(select_dir=outputs.get("video_result_dir"))
        self._load_outputs(outputs)

    def _load_outputs(self, outputs: Dict[str, Any]):
        combined_video = outputs.get("combined_video_path")
        if combined_video and Path(combined_video).exists():
            self._output_player.open(combined_video)
            self._compare_output.open(combined_video)

        csv_path = outputs.get("csv_path")
        df = pd.DataFrame()
        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
        self._csv_model.set_dataframe(df)
        self._current_df = df.copy()

        if not df.empty:
            xs = df["time_seconds"].astype(float).tolist() if "time_seconds" in df.columns else list(range(len(df)))
            speed = df["ball_speed"].astype(float).tolist() if "ball_speed" in df.columns else [0.0] * len(xs)
            self._speed_plot.set_series(xs, speed, x_label="time_seconds", y_label="ball_speed")

            by = (
                df["ball_denoise_y"].astype(float).tolist()
                if "ball_denoise_y" in df.columns
                else (df["ball_y"].astype(float).tolist() if "ball_y" in df.columns else [0.0] * len(xs))
            )
            self._ball_y_plot.set_series(xs, by, x_label="time_seconds", y_label="ball_y")

            hits = (
                df["cumulative_hit_count"].astype(float).tolist()
                if "cumulative_hit_count" in df.columns
                else [0.0] * len(xs)
            )
            self._hit_count_plot.set_series(xs, hits, x_label="time_seconds", y_label="cumulative_hit_count")
        else:
            self._speed_plot.set_series([], [])
            self._ball_y_plot.set_series([], [])
            self._hit_count_plot.set_series([], [])

        events_df = self._load_events(outputs)
        self._events_raw_df = events_df.copy()
        self._rebuild_event_filters(events_df)
        self._apply_event_filters()
        self._update_overview(df, events_df, outputs)
        self._update_players(df)
        self._update_distributions(df, events_df)

    def _load_events(self, outputs: Dict[str, Any]) -> pd.DataFrame:
        hit_path = outputs.get("hit_events_path")
        if not hit_path or not Path(hit_path).exists():
            return pd.DataFrame()
        with open(hit_path, "r", encoding="utf-8") as f:
            hits = json.load(f)
        df = pd.DataFrame(hits)

        stroke_path = outputs.get("stroke_results_path")
        if stroke_path and Path(stroke_path).exists():
            with open(stroke_path, "r", encoding="utf-8") as f:
                strokes = json.load(f)
            sdf = pd.DataFrame(strokes)
            if "frame" in df.columns and "frame" in sdf.columns:
                df = df.merge(sdf, on="frame", how="left", suffixes=("", "_stroke"))
        return df

    def _update_overview(self, df: pd.DataFrame, events_df: pd.DataFrame, outputs: Dict[str, Any]):
        total_hits = 0
        if "is_hit" in df.columns:
            total_hits = int(df["is_hit"].astype(int).sum())
        elif "frame" in events_df.columns:
            total_hits = int(len(events_df))
        self._m_hits.set_content(value=str(total_hits), subtitle="hit frames")

        duration = 0.0
        if "time_seconds" in df.columns and len(df) > 0:
            duration = float(df["time_seconds"].iat[-1])
        self._m_duration.set_content(value=f"{duration:.1f}s", subtitle="from video fps")

        if "ball_speed" in df.columns and len(df) > 0:
            sp = pd.to_numeric(df["ball_speed"], errors="coerce").fillna(0.0)
            self._m_speed_avg.set_content(value=f"{float(sp.mean()):.1f}", subtitle="avg over frames")
            self._m_speed_max.set_content(value=f"{float(sp.max()):.1f}", subtitle="max over frames")
        else:
            self._m_speed_avg.set_content(value="--", subtitle="")
            self._m_speed_max.set_content(value="--", subtitle="")

        visible_ratio = None
        vis_col = "ball_denoise_visible" if "ball_denoise_visible" in df.columns else ("ball_visible" if "ball_visible" in df.columns else None)
        if vis_col is not None and len(df) > 0:
            v = pd.to_numeric(df[vis_col], errors="coerce").fillna(0).astype(int)
            visible_ratio = float((v > 0).mean())
        self._m_visible.set_content(value=f"{(visible_ratio * 100):.1f}%" if visible_ratio is not None else "--", subtitle=vis_col or "")

        out_dir = outputs.get("video_result_dir", "")
        self._m_output.set_content(value=Path(out_dir).name if out_dir else "--", subtitle=out_dir)

        if not events_df.empty and "frame" in events_df.columns:
            frames = events_df["frame"].astype(int).tolist()
            xs = []
            if "time_seconds" in df.columns and len(df) > 0:
                ts = df["time_seconds"].astype(float).tolist()
                for f in frames:
                    if 0 <= f < len(ts):
                        xs.append(float(ts[f]))
                    else:
                        xs.append(float(f))
            else:
                xs = [float(f) for f in frames]
            colors: List[QColor] = []
            if "player" in events_df.columns:
                ps = events_df["player"].astype(int).tolist()
                for p in ps:
                    colors.append(QColor("#3b82f6") if p == 0 else QColor("#ef4444"))
            self._timeline.set_markers(xs, frames, colors=colors, title="击球时间轴(点击跳转)")
        else:
            self._timeline.set_markers([], [], title="击球时间轴(点击跳转)")

        self._refresh_density_view()
        self._update_current_ball_marker(self._output_player.current_frame())

        if not events_df.empty:
            col = None
            for c in ["stroke_type_name", "stroke_type_name_en", "stroke_type"]:
                if c in events_df.columns:
                    col = c
                    break
            if col is not None:
                s = events_df[col].fillna("").astype(str)
                s = s[s != ""]
                vc = s.value_counts().head(10)
                self._stroke_bar.set_data(vc.index.tolist(), vc.astype(float).tolist(), title="击球类型分布(Top10)")
            else:
                self._stroke_bar.set_data([], [], title="击球类型分布")
        else:
            self._stroke_bar.set_data([], [], title="击球类型分布")

    def _update_distributions(self, df: pd.DataFrame, events_df: pd.DataFrame):
        speed_values: List[float] = []
        if df is not None and not df.empty and "ball_speed" in df.columns:
            sp = pd.to_numeric(df["ball_speed"], errors="coerce").fillna(0.0).astype(float)
            vis_col = self._visible_column(df)
            if vis_col is not None:
                vis = pd.to_numeric(df[vis_col], errors="coerce").fillna(0).astype(int)
                sp = sp[vis > 0]
            speed_values = sp.tolist()
        self._speed_hist.set_data(speed_values, title="球速分布(Pro)", x_label="ball_speed", color="#3b82f6")

        intervals: List[float] = []
        fps = 30.0
        if events_df is not None and not events_df.empty and "frame" in events_df.columns:
            frames = pd.to_numeric(events_df["frame"], errors="coerce").dropna().astype(int).sort_values().tolist()
            if len(frames) >= 2:
                if df is not None and not df.empty and "time_seconds" in df.columns:
                    ts = pd.to_numeric(df["time_seconds"], errors="coerce").fillna(0.0).astype(float).tolist()
                    try:
                        fps = len(df) / float(df["time_seconds"].iat[-1])
                    except ZeroDivisionError:
                        pass
                    for i in range(1, len(frames)):
                        a = frames[i - 1]
                        b = frames[i]
                        if 0 <= a < len(ts) and 0 <= b < len(ts):
                            intervals.append(max(0.0, float(ts[b] - ts[a])))
                else:
                    info = self._output_player.info()
                    fps = float(info.fps) if info is not None else 25.0
                    for i in range(1, len(frames)):
                        intervals.append(max(0.0, float(frames[i] - frames[i - 1]) / max(1.0, fps)))
        self._hit_interval_hist.set_data(intervals, title="击球间隔分布(Pro)", x_label="delta_seconds", color="#22c55e")

        # Hit Height
        heights: List[float] = []
        if events_df is not None and not events_df.empty:
            if "y" in events_df.columns:
                heights = pd.to_numeric(events_df["y"], errors="coerce").dropna().astype(float).tolist()
            elif "frame" in events_df.columns and df is not None and not df.empty:
                frames = pd.to_numeric(events_df["frame"], errors="coerce").dropna().astype(int).tolist()
                y_col = "ball_denoise_y" if "ball_denoise_y" in df.columns else ("ball_y" if "ball_y" in df.columns else None)
                if y_col:
                    y_vals = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0).astype(float).tolist()
                    for f in frames:
                        if 0 <= f < len(y_vals):
                            heights.append(y_vals[f])
        self._hit_height_hist.set_data(heights, title="击球高度分布(Pro)", x_label="hit_y_px", color="#f59e0b")

        # Player Speed
        p_speeds: List[float] = []
        if df is not None and not df.empty:
            p1 = getattr(self, "_p1_centroids", [])
            p2 = getattr(self, "_p2_centroids", [])
            t = df["time_seconds"].astype(float).tolist() if "time_seconds" in df.columns else []
            
            def calc_speeds(centroids):
                s = []
                if not centroids or len(centroids) < 2 or not t:
                    return s
                for i in range(1, len(centroids)):
                    if i >= len(t): break
                    c1, c2 = centroids[i-1], centroids[i]
                    dt = t[i] - t[i-1]
                    if c1 and c2 and dt > 0.001:
                        dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
                        s.append(dist / dt)
                return s

            p_speeds.extend(calc_speeds(p1))
            p_speeds.extend(calc_speeds(p2))
        
        self._player_speed_hist.set_data(p_speeds, title="选手瞬时速度分布(Pro)", x_label="player_speed_px_s", color="#a78bfa")

    def _player_joint_columns(self, df: pd.DataFrame, player_prefix: str):
        xs = []
        ys = []
        for j in range(17):
            x = f"{player_prefix}_joint{j}_x"
            y = f"{player_prefix}_joint{j}_y"
            if x in df.columns and y in df.columns:
                xs.append(x)
                ys.append(y)
        return xs, ys

    def _compute_player_centroids(self, df: pd.DataFrame, player_prefix: str) -> List[Optional[tuple[float, float]]]:
        if df is None or df.empty:
            return []
        xs_cols, ys_cols = self._player_joint_columns(df, player_prefix)
        if not xs_cols or not ys_cols:
            return [None for _ in range(len(df))]

        xs = df[xs_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        ys = df[ys_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        out: List[Optional[tuple[float, float]]] = []
        for i in range(len(df)):
            x_row = xs.iloc[i].to_numpy()
            y_row = ys.iloc[i].to_numpy()
            mask = (x_row > 1.0) & (y_row > 1.0)
            if mask.sum() < 3:
                out.append(None)
                continue
            out.append((float(x_row[mask].mean()), float(y_row[mask].mean())))
        return out

    def _update_players(self, df: pd.DataFrame):
        if df is None or df.empty:
            self._p1_map.set_points([], title="选手0覆盖(凸包)", color="#3b82f6")
            self._p2_map.set_points([], title="选手1覆盖(凸包)", color="#ef4444")
            self._p_dist_plot.set_series([], [], x_label="time_seconds", y_label="player_distance_px", title="选手间距")
            self._p1_speed_plot.set_series([], [], x_label="time_seconds", y_label="p0_speed_px_s", title="选手0速度")
            self._p2_speed_plot.set_series([], [], x_label="time_seconds", y_label="p1_speed_px_s", title="选手1速度")
            return

        t = df["time_seconds"].astype(float).tolist() if "time_seconds" in df.columns else list(range(len(df)))
        p1 = self._compute_player_centroids(df, "player1")
        p2 = self._compute_player_centroids(df, "player2")
        self._p1_centroids = p1
        self._p2_centroids = p2

        p1_pts = [pt for pt in p1 if pt is not None]
        p2_pts = [pt for pt in p2 if pt is not None]
        self._p1_map.set_points(p1_pts, title="选手0覆盖(凸包)", color="#3b82f6")
        self._p2_map.set_points(p2_pts, title="选手1覆盖(凸包)", color="#ef4444")

        info = self._output_player.info()
        fps = float(info.fps) if info is not None else 25.0

        dist_x: List[float] = []
        dist_y: List[float] = []
        for i in range(len(df)):
            if i >= len(t):
                break
            if p1[i] is None or p2[i] is None:
                continue
            dx = float(p1[i][0] - p2[i][0])
            dy = float(p1[i][1] - p2[i][1])
            dist_x.append(float(t[i]))
            dist_y.append((dx * dx + dy * dy) ** 0.5)
        self._p_dist_plot.set_series(dist_x, dist_y, x_label="time_seconds", y_label="player_distance_px", title="选手间距")

        s1_x: List[float] = []
        s1_y: List[float] = []
        s2_x: List[float] = []
        s2_y: List[float] = []
        for i in range(1, len(df)):
            if i >= len(t):
                break
            if p1[i - 1] is not None and p1[i] is not None:
                dx = float(p1[i][0] - p1[i - 1][0])
                dy = float(p1[i][1] - p1[i - 1][1])
                s1_x.append(float(t[i]))
                s1_y.append(((dx * dx + dy * dy) ** 0.5) * fps)
            if p2[i - 1] is not None and p2[i] is not None:
                dx = float(p2[i][0] - p2[i - 1][0])
                dy = float(p2[i][1] - p2[i - 1][1])
                s2_x.append(float(t[i]))
                s2_y.append(((dx * dx + dy * dy) ** 0.5) * fps)
        self._p1_speed_plot.set_series(s1_x, s1_y, x_label="time_seconds", y_label="p0_speed_px_s", title="选手0速度")
        self._p2_speed_plot.set_series(s2_x, s2_y, x_label="time_seconds", y_label="p1_speed_px_s", title="选手1速度")

        self._update_current_player_markers(self._output_player.current_frame())

    def _update_current_player_markers(self, frame_index: int):
        p1 = getattr(self, "_p1_centroids", None)
        p2 = getattr(self, "_p2_centroids", None)
        if p1 is None or p2 is None:
            self._p1_map.set_current_point(None)
            self._p2_map.set_current_point(None)
            return
        i = int(frame_index)
        if not (0 <= i < len(p1)) or not (0 <= i < len(p2)):
            self._p1_map.set_current_point(None)
            self._p2_map.set_current_point(None)
            return
        self._p1_map.set_current_point(p1[i] if p1[i] is not None else None)
        self._p2_map.set_current_point(p2[i] if p2[i] is not None else None)

    def _on_density_show_current_toggled(self, checked: bool):
        self._heatmap.set_show_current_point(bool(checked))
        self._update_current_ball_marker(self._output_player.current_frame())

    def _density_bins(self):
        text = self._density_bins_combo.currentText()
        if "24x14" in text:
            return (24, 14)
        if "72x40" in text:
            return (72, 40)
        return (44, 24)

    def _ball_xy_columns(self, df: pd.DataFrame):
        x_col = "ball_denoise_x" if "ball_denoise_x" in df.columns else ("ball_x" if "ball_x" in df.columns else None)
        y_col = "ball_denoise_y" if "ball_denoise_y" in df.columns else ("ball_y" if "ball_y" in df.columns else None)
        return x_col, y_col

    def _visible_column(self, df: pd.DataFrame):
        return "ball_denoise_visible" if "ball_denoise_visible" in df.columns else ("ball_visible" if "ball_visible" in df.columns else None)

    def _frames_for_density_mode(self) -> Optional[List[int]]:
        mode = self._density_source_combo.currentText()
        df = self._current_df
        events_df = self._events_raw_df
        if df is None or df.empty:
            return []

        if mode == "全部帧":
            return list(range(len(df)))

        if mode == "可见帧":
            vis_col = self._visible_column(df)
            if vis_col is None:
                return list(range(len(df)))
            vis = pd.to_numeric(df[vis_col], errors="coerce").fillna(0).astype(int)
            return vis.index[vis > 0].astype(int).tolist()

        if mode == "仅击球帧":
            if "is_hit" in df.columns:
                hit = pd.to_numeric(df["is_hit"], errors="coerce").fillna(0).astype(int)
                return hit.index[hit > 0].astype(int).tolist()
            if events_df is not None and not events_df.empty and "frame" in events_df.columns:
                return pd.to_numeric(events_df["frame"], errors="coerce").dropna().astype(int).tolist()
            return []

        if mode in ["仅选手0击球帧", "仅选手1击球帧"]:
            if events_df is None or events_df.empty or "frame" not in events_df.columns or "player" not in events_df.columns:
                return []
            want = 0 if "0" in mode else 1
            e = events_df.copy()
            pl = pd.to_numeric(e["player"], errors="coerce").fillna(-999).astype(int)
            e = e[pl == want]
            return pd.to_numeric(e["frame"], errors="coerce").dropna().astype(int).tolist()

        return None

    def _refresh_density_view(self, checked: bool = False):
        df = self._current_df
        if df is None or df.empty:
            self._heatmap.set_bins(self._density_bins())
            self._heatmap.set_points([], title="球位置密度(气泡聚合)")
            return

        x_col, y_col = self._ball_xy_columns(df)
        if x_col is None or y_col is None:
            self._heatmap.set_bins(self._density_bins())
            self._heatmap.set_points([], title="球位置密度(气泡聚合)")
            return

        frames = self._frames_for_density_mode()
        if frames is None:
            frames = list(range(len(df)))

        frames = [f for f in frames if 0 <= int(f) < len(df)]
        xs = pd.to_numeric(df.loc[frames, x_col], errors="coerce").fillna(0.0).astype(float)
        ys = pd.to_numeric(df.loc[frames, y_col], errors="coerce").fillna(0.0).astype(float)

        vis_col = self._visible_column(df)
        if self._density_source_combo.currentText() != "全部帧" and vis_col is not None:
            vis = pd.to_numeric(df.loc[frames, vis_col], errors="coerce").fillna(0).astype(int)
            mask = vis > 0
            xs = xs[mask]
            ys = ys[mask]

        pts = list(zip(xs.tolist(), ys.tolist()))
        title = f"球位置密度(气泡聚合) · {self._density_source_combo.currentText()}"
        self._heatmap.set_bins(self._density_bins())
        self._heatmap.set_points(pts, title=title)
        self._update_current_ball_marker(self._output_player.current_frame())

    def _update_current_ball_marker(self, frame_index: int):
        if not self._density_show_current_cb.isChecked():
            self._heatmap.set_current_point(None)
            return
        df = self._current_df
        if df is None or df.empty:
            self._heatmap.set_current_point(None)
            return
        if not (0 <= int(frame_index) < len(df)):
            self._heatmap.set_current_point(None)
            return
        x_col, y_col = self._ball_xy_columns(df)
        if x_col is None or y_col is None:
            self._heatmap.set_current_point(None)
            return
        x = pd.to_numeric(df[x_col].iat[int(frame_index)], errors="coerce")
        y = pd.to_numeric(df[y_col].iat[int(frame_index)], errors="coerce")
        if pd.isna(x) or pd.isna(y):
            self._heatmap.set_current_point(None)
            return
        self._heatmap.set_current_point((float(x), float(y)))

    def _export_density_png(self):
        default_dir = Path(self._result_dir_edit.text().strip()) if self._result_dir_edit.text().strip() else ROOT
        default_dir.mkdir(parents=True, exist_ok=True)
        file_path, _ = QFileDialog.getSaveFileName(self, "导出密度图", str(default_dir / "ball_density.png"), "PNG Image (*.png)")
        if not file_path:
            return
        pix = self._heatmap.grab()
        pix.save(file_path, "PNG")
        self._append_log(f"已导出密度图: {file_path}")

    def _on_finished(self, ok: bool, message: str):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        self._worker_thread = None
        self._worker = None
        self._stepper.set_finished(ok)
        if ok:
            self._status_step.setText("完成")
            self._append_log("分析完成")
        else:
            self._status_step.setText("停止" if message == "已停止" else "失败")
            self._append_log(message)

    def _on_output_position(self, frame_index: int):
        self._highlight_from_frame(frame_index)

    def _highlight_from_frame(self, frame_index: int):
        if self._csv_model.rowCount() <= 0:
            return
        df = self._csv_model._df
        if "time_seconds" not in df.columns:
            return
        if not (0 <= frame_index < len(df)):
            return
        t = float(df["time_seconds"].iat[frame_index])
        self._speed_plot.set_highlight_x(t)
        self._ball_y_plot.set_highlight_x(t)
        self._hit_count_plot.set_highlight_x(t)
        self._timeline.set_selected_by_frame(int(frame_index))
        self._update_current_ball_marker(int(frame_index))
        self._update_current_player_markers(int(frame_index))

    def _seek_output(self, frame_index: int):
        self._output_player.seek(int(frame_index))
        self._highlight_from_frame(int(frame_index))
        self._sync_compare_from_output(int(frame_index))

    def _on_event_selection_changed(self, selected: QItemSelection, deselected: QItemSelection):
        if selected.indexes():
            row = selected.indexes()[0].row()
            df = self._events_model._df
            if "frame" not in df.columns:
                return
            try:
                frame = int(df["frame"].iat[row])
            except Exception:
                return
            self._seek_output(frame)

    def _on_event_double_clicked(self, index):
        if not index.isValid():
            return
        df = self._events_model._df
        if df is None or df.empty or "frame" not in df.columns:
            return
        try:
            frame = int(df["frame"].iat[index.row()])
        except Exception:
            return
        self._seek_output(frame)

    def _reset_event_filters(self):
        self._event_search.setText("")
        self._event_player_filter.blockSignals(True)
        self._event_stroke_filter.blockSignals(True)
        self._event_player_filter.setCurrentIndex(0)
        self._event_stroke_filter.setCurrentIndex(0)
        self._event_player_filter.blockSignals(False)
        self._event_stroke_filter.blockSignals(False)
        self._apply_event_filters()

    def _rebuild_event_filters(self, events_df: pd.DataFrame):
        self._event_player_filter.blockSignals(True)
        self._event_stroke_filter.blockSignals(True)
        self._event_player_filter.clear()
        self._event_stroke_filter.clear()
        self._event_player_filter.addItem("全部")
        self._event_stroke_filter.addItem("全部")

        if not events_df.empty:
            if "player" in events_df.columns:
                for v in sorted(set(pd.to_numeric(events_df["player"], errors="coerce").dropna().astype(int).tolist())):
                    self._event_player_filter.addItem(str(v))
            stroke_col = None
            for c in ["stroke_type_name", "stroke_type_name_en", "stroke_type"]:
                if c in events_df.columns:
                    stroke_col = c
                    break
            if stroke_col is not None:
                vals = events_df[stroke_col].fillna("").astype(str)
                vals = sorted(set([s for s in vals.tolist() if s.strip()]))
                for s in vals[:200]:
                    self._event_stroke_filter.addItem(s)

        self._event_player_filter.blockSignals(False)
        self._event_stroke_filter.blockSignals(False)

    def _apply_event_filters(self):
        base = getattr(self, "_events_raw_df", pd.DataFrame()).copy()
        if base.empty:
            self._events_model.set_dataframe(base)
            return

        player_text = self._event_player_filter.currentText().strip()
        if player_text and player_text != "全部" and "player" in base.columns:
            try:
                p = int(player_text)
                base = base[pd.to_numeric(base["player"], errors="coerce").fillna(-999).astype(int) == p]
            except Exception:
                pass

        stroke_text = self._event_stroke_filter.currentText().strip()
        if stroke_text and stroke_text != "全部":
            stroke_col = None
            for c in ["stroke_type_name", "stroke_type_name_en", "stroke_type"]:
                if c in base.columns:
                    stroke_col = c
                    break
            if stroke_col is not None:
                base = base[base[stroke_col].fillna("").astype(str) == stroke_text]

        q = self._event_search.text().strip()
        if q:
            q_lower = q.lower()
            cols = [c for c in ["frame", "player", "stroke_type_name", "stroke_type_name_en", "stroke_type"] if c in base.columns]
            if not cols:
                cols = list(base.columns)[:20]
            mask = None
            for c in cols:
                s = base[c].fillna("").astype(str).str.lower().str.contains(q_lower, regex=False)
                mask = s if mask is None else (mask | s)
            if mask is not None:
                base = base[mask]

        self._events_model.set_dataframe(base.reset_index(drop=True))

    def _on_compare_input_pos(self, frame_index: int):
        if not self._compare_sync_cb.isChecked():
            return
        if self._compare_follow_combo.currentText() != "输入驱动输出":
            return
        self._sync_compare_from_input(frame_index)

    def _on_compare_output_pos(self, frame_index: int):
        if not self._compare_sync_cb.isChecked():
            return
        if self._compare_follow_combo.currentText() != "输出驱动输入":
            return
        self._sync_compare_from_output(frame_index)

    def _sync_compare_from_input(self, input_frame: int):
        if self._compare_guard:
            return
        in_info = self._compare_input.info()
        out_info = self._compare_output.info()
        if in_info is None or out_info is None:
            return
        t = float(input_frame) / max(1.0, float(in_info.fps))
        out_frame = int(t * float(out_info.fps))
        out_frame = max(0, min(out_frame, max(0, out_info.total_frames - 1)))
        if abs(out_frame - self._compare_output.current_frame()) <= 1:
            return
        self._compare_guard = True
        try:
            self._compare_output.seek(out_frame)
        finally:
            self._compare_guard = False

    def _sync_compare_from_output(self, output_frame: int):
        if self._compare_guard:
            return
        in_info = self._compare_input.info()
        out_info = self._compare_output.info()
        if in_info is None or out_info is None:
            return
        t = float(output_frame) / max(1.0, float(out_info.fps))
        in_frame = int(t * float(in_info.fps))
        in_frame = max(0, min(in_frame, max(0, in_info.total_frames - 1)))
        if abs(in_frame - self._compare_input.current_frame()) <= 1:
            return
        self._compare_guard = True
        try:
            self._compare_input.seek(in_frame)
        finally:
            self._compare_guard = False

    def _export_overview_png(self):
        widget = getattr(self, "_overview_widget", None)
        if widget is None:
            return
        default_dir = Path(self._result_dir_edit.text().strip()) if self._result_dir_edit.text().strip() else ROOT
        default_dir.mkdir(parents=True, exist_ok=True)
        file_path, _ = QFileDialog.getSaveFileName(self, "导出概览截图", str(default_dir / "overview.png"), "PNG Image (*.png)")
        if not file_path:
            return
        pix = widget.grab()
        pix.save(file_path, "PNG")
        self._append_log(f"已导出概览截图: {file_path}")

    def _export_events_csv(self):
        df = self._events_model._df
        if df is None or df.empty:
            QMessageBox.information(self, "提示", "当前没有事件数据可导出")
            return
        default_dir = Path(self._result_dir_edit.text().strip()) if self._result_dir_edit.text().strip() else ROOT
        default_dir.mkdir(parents=True, exist_ok=True)
        file_path, _ = QFileDialog.getSaveFileName(self, "导出事件表 CSV", str(default_dir / "events_filtered.csv"), "CSV Files (*.csv)")
        if not file_path:
            return
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        self._append_log(f"已导出事件 CSV: {file_path}")

    def _export_csv_csv(self):
        df = self._csv_model._df
        if df is None or df.empty:
            QMessageBox.information(self, "提示", "当前没有 CSV 数据可导出")
            return
        default_dir = Path(self._result_dir_edit.text().strip()) if self._result_dir_edit.text().strip() else ROOT
        default_dir.mkdir(parents=True, exist_ok=True)
        file_path, _ = QFileDialog.getSaveFileName(self, "导出 CSV 数据", str(default_dir / "track_data.csv"), "CSV Files (*.csv)")
        if not file_path:
            return
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        self._append_log(f"已导出 CSV 数据: {file_path}")

    def _scan_results(self) -> List[Dict[str, Any]]:
        text = self._result_dir_edit.text().strip()
        if not text:
            return []
        root = Path(text).expanduser()
        if not root.is_absolute():
            root = ROOT / root
        root = root.resolve(strict=False)
        if not root.exists():
            return []
        results: List[Dict[str, Any]] = []
        for sub in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            combined = next(iter(sorted(sub.glob("*_combined.mp4"))), None)
            csv = next(iter(sorted(sub.glob("*_data.csv"))), None)
            hit = next(iter(sorted(sub.glob("*_hit_events.json"))), None)
            stroke = next(iter(sorted(sub.glob("*_stroke_types.json"))), None)
            if combined is None and csv is None and hit is None:
                continue
            results.append(
                {
                    "video_result_dir": str(sub),
                    "combined_video_path": str(combined) if combined else "",
                    "csv_path": str(csv) if csv else "",
                    "hit_events_path": str(hit) if hit else "",
                    "stroke_results_path": str(stroke) if stroke else "",
                }
            )
        return results

    def _refresh_results_list(self, checked: bool = False, select_dir: Optional[str] = None):
        current_dir = select_dir
        if current_dir is None:
            cur_data = self._result_combo.currentData()
            if isinstance(cur_data, dict):
                current_dir = cur_data.get("video_result_dir")

        items = self._scan_results()
        self._result_combo.blockSignals(True)
        self._result_combo.clear()
        select_index = -1
        for idx, item in enumerate(items):
            name = Path(item["video_result_dir"]).name
            self._result_combo.addItem(name, item)
            if current_dir and Path(item["video_result_dir"]) == Path(current_dir):
                select_index = idx
        if select_index >= 0:
            self._result_combo.setCurrentIndex(select_index)
        self._result_combo.blockSignals(False)

    def _load_selected_result(self, checked: bool = False):
        data = self._result_combo.currentData()
        if not isinstance(data, dict):
            return
        self._current_outputs = data
        self._load_outputs(data)

    def _get_black_frame(self):
        import numpy as np
        return np.zeros((360, 640, 3), dtype=np.uint8)

    def _open_match_review(self):
        if not self._review_window:
            self._review_window = MatchReviewWindow(self)
        self._review_window.show()

    def _show_about(self):
        about_text = """
        <h2>TrackNetV3_Attention</h2>
        <p><b>版本:</b> 1.1</p>
        <p><b>定位:</b> 端到端的羽毛球视频智能分析与专业复盘系统</p>
        <hr>
        <h3>系统概览</h3>
        <p>本系统面向教练与运动员，提供从视频输入到数据洞察与战术复盘的完整流程：目标检测、姿态估计、事件识别、击球类型识别、结果合成与多维可视化。</p>
        <h3>核心功能</h3>
        <ul>
            <li><b>检测与识别:</b> TrackNetV3 羽毛球检测、MMPose 姿态检测、击球事件检测、BST 击球类型识别、球场/球网检测与透视映射</li>
            <li><b>结果管理:</b> 历史结果自动索引与加载，支持数据集下拉选择与一键刷新（results/目录）</li>
            <li><b>专业复盘:</b> 
                <ul>
                    <li>概览：六维能力雷达 + 三维战术图（滚轮缩放、按球速/类型着色切换、连线/包络开关、透明背景）</li>
                    <li>球员表现：KPI 指标（均速/95%分位/峰值等）、速度直方图+KDE、加速度时间序列（冲刺标记）、覆盖分位等值线、站位质心与稳定性椭圆</li>
                    <li>技术统计：双侧击球类型分布与速度–高度散点</li>
                    <li>深度战术：战术类型流图（ThemeRiver，时间维度的击球战术结构）、转移热力图、空间控制(Voronoi)</li>
                    <li>体能负荷：累计距离与三维击球分析</li>
                </ul>
            </li>
            <li><b>交互增强:</b> 悬停放大预览（所有图表）、复盘窗口右上角数据集刷新保留当前选择</li>
            <li><b>导出:</b> CSV / 截图 / 可视化视频</li>
        </ul>
        <h3>项目结构 (e:\\learn\\TrackNetV3_Attention)</h3>
        <ul>
            <li><b>core/</b> 数据管线与算法模块（ball_detect、pose_detect、event_detect、stroke_classify、visualize_combined、export_to_csv 等）</li>
            <li><b>models/</b> 模型权重（TrackNet、BST、球场/球网）</li>
            <li><b>videos/</b> 示例输入视频（test2.mp4、test6.mp4）</li>
            <li><b>results/</b> 输出目录（每次分析生成 <i>视频名/</i> 子目录，含 *_data.csv、*_hit_events.json、*_stroke_types.json、*_poses.npy、*_combined.mp4、loca_info/ 等）</li>
            <li><b>ui_pyside6/</b> 图形界面与复盘实现（main.py、match_review/*）</li>
            <li><b>run_combined.py</b> 可视化合成运行脚本</li>
            <li><b>docs/README.md</b> 项目文档</li>
        </ul>
        <h3>数据规范</h3>
        <ul>
            <li><b>CSV(*_data.csv):</b> 主要字段包括 time_seconds、ball_x/ball_y/ball_speed、p1_speed/p2_speed、player关节坐标等</li>
            <li><b>事件JSON(*_hit_events.json):</b> 每次击球的帧索引与选手</li>
            <li><b>类型JSON(*_stroke_types.json):</b> 每次击球的类型与帧索引</li>
            <li><b>姿态(*_poses.npy):</b> 姿态关键点数组</li>
            <li><b>视频(*_combined.mp4):</b> 合成可视化输出</li>
        </ul>
        <h3>技术栈</h3>
        <p>PySide6 + PyTorch + MMPose + Matplotlib/Seaborn</p>
        """
        QMessageBox.about(self, "关于", about_text)

    def _show_usage(self):
        usage_text = """
        <h2>使用说明</h2>
        <h3>1. 准备与参数</h3>
        <ul>
            <li>将待分析视频放置到 <b>videos/</b> 或任意路径</li>
            <li>在“参数”页选择 <b>设备</b>(cuda/cpu)、<b>姿态模型</b>(rtmpose-t/s/m/l)、是否启用<b>球场/球网检测</b></li>
            <li>设置 <b>TrackNet 权重</b>、<b>输入帧数</b>(1–9)、<b>检测阈值</b>(0–1)、<b>轨迹长度</b>、<b>检测/预览间隔</b> 与 <b>输出目录</b></li>
        </ul>
        <h3>2. 执行分析</h3>
        <p>点击“开始训练分析”，系统自动执行：球场/球网检测 → 羽毛球检测 → 姿态检测 → 击球事件 → 击球类型 → 合成可视化 → 数据导出。过程中可随时点击“停止”。</p>
        <h3>3. 复盘窗口（比赛复盘数据）</h3>
        <ul>
            <li>右上角<b>数据集下拉</b>列出 <b>results/</b> 下历史结果；旁侧<b>刷新</b>用于新增数据后快速更新列表（保留当前选择）</li>
            <li><b>概览 (Dashboard):</b> 六维雷达 + 三维战术图。三维图支持滚轮缩放、按球速/类型着色切换、开/关包络与连线、透明背景；悬停可放大预览</li>
            <li><b>战术复盘 (Tactical):</b> 回合列表选择 → 三维球路沙盘与详情信息（时长、拍数、选手跑动与均速等）</li>
            <li><b>球员表现 (Physical):</b> 
                <ul>
                    <li>KPI：平均速度、95%分位、最大速度、加速度峰值/均值、前/后场占比、左/右占比、近网攻势指数</li>
                    <li>速度分布：直方图+KDE；加速度时间序列：高加速度点标注；覆盖分位等值线：50/80/95%；站位质心与稳定性：质心+椭圆</li>
                </ul>
            </li>
            <li><b>技术统计 (Technical):</b> 双侧击球类型饼图、速度–高度散点</li>
            <li><b>深度战术 (Deep Tactics):</b> 战术类型流图（ThemeRiver，展示各类型随时间的占比趋势）、战术转移热力图、空间控制 (Voronoi)</li>
            <li><b>体能负荷 (Load):</b> 累计距离曲线与三维击球分析</li>
        </ul>
        <h3>4. 输出目录结构</h3>
        <ul>
            <li>路径：<b>e:\\learn\\TrackNetV3_Attention\\results\\&lt;视频名&gt;</b></li>
            <li><b>*_data.csv:</b> 时间戳、球坐标/球速、选手速度与关键点等帧级数据</li>
            <li><b>*_hit_events.json:</b> 每次击球的帧与选手</li>
            <li><b>*_stroke_types.json:</b> 每次击球的类型与帧</li>
            <li><b>*_poses.npy:</b> 姿态关键点数组</li>
            <li><b>*_combined.mp4:</b> 合成可视化视频</li>
            <li><b>loca_info/、loca_info_denoise/:</b> 球场位置信息与去噪版本</li>
        </ul>
        <h3>5. 模型与示例</h3>
        <ul>
            <li><b>models/:</b> TrackNet/球场/球网/BST 权重</li>
            <li><b>videos/:</b> 示例 test2.mp4、test6.mp4</li>
            <li><b>run_combined.py:</b> 合成可视化运行脚本</li>
        </ul>
        <h3>6. 常见问题</h3>
        <ul>
            <li>复盘列表未出现新数据：点击复盘窗口右上角“刷新”按钮</li>
            <li>三维图过小或偏移：使用滚轮缩放与视角按钮，或点击“重置缩放”</li>
            <li>中文显示异常：确保系统已安装中文字体（微软雅黑/黑体）</li>
        </ul>
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("使用说明")
        dlg.resize(1050, 780)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        view = QTextBrowser(dlg)
        view.setOpenExternalLinks(True)
        view.setHtml(usage_text)
        layout.addWidget(view, 1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        close_btn = QPushButton("关闭", dlg)
        close_btn.clicked.connect(dlg.close)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        dlg.exec()


def main():
    app = QApplication(sys.argv)
    _apply_style(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
