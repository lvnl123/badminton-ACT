from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def _bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    return QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


@dataclass
class VideoInfo:
    fps: float
    total_frames: int
    width: int
    height: int


class VideoPlayer(QWidget):
    positionChanged = Signal(int)
    videoOpened = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._cap: Optional[cv2.VideoCapture] = None
        self._path: Optional[str] = None
        self._info: Optional[VideoInfo] = None
        self._playing = False
        self._frame_index = 0
        self._updating_slider = False

        self._frame_label = QLabel()
        self._frame_label.setAlignment(Qt.AlignCenter)
        self._frame_label.setMinimumHeight(260)
        self._frame_label.setStyleSheet("QLabel{background:#0f1216;border:1px solid #2a2f3a;}")

        self._play_btn = QPushButton("播放")
        self._pause_btn = QPushButton("暂停")
        self._pause_btn.setEnabled(False)
        self._time_label = QLabel("-- / --")
        self._time_label.setMinimumWidth(140)
        self._time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setEnabled(False)
        self._slider.setMinimum(0)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.valueChanged.connect(self._on_slider_changed)

        controls = QHBoxLayout()
        controls.addWidget(self._play_btn)
        controls.addWidget(self._pause_btn)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._time_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._frame_label, 1)
        layout.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._play_btn.clicked.connect(self.play)
        self._pause_btn.clicked.connect(self.pause)

    def open(self, path: str) -> bool:
        self.close_video()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self._cap = cap
        self._path = path
        self._info = VideoInfo(fps=fps, total_frames=total_frames, width=width, height=height)
        self._frame_index = 0

        self._slider.setEnabled(total_frames > 0)
        self._slider.setMaximum(max(0, total_frames - 1))
        self._play_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)

        self._render_frame_at(0)
        self.videoOpened.emit(path)
        return True

    def seek(self, frame_index: int):
        if self._cap is None or self._info is None:
            return
        self.pause()
        self._render_frame_at(int(frame_index))

    def info(self) -> Optional[VideoInfo]:
        return self._info

    def current_frame(self) -> int:
        return int(self._frame_index)

    def close_video(self):
        self.pause()
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._path = None
        self._info = None
        self._frame_index = 0
        self._slider.setEnabled(False)
        self._time_label.setText("-- / --")
        self._frame_label.setPixmap(QPixmap())

    def play(self):
        if self._cap is None or self._info is None:
            return
        if self._playing:
            return
        self._playing = True
        self._play_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        interval_ms = int(1000 / max(1.0, float(self._info.fps)))
        self._timer.start(max(1, interval_ms))

    def pause(self):
        self._playing = False
        self._timer.stop()
        self._play_btn.setEnabled(self._cap is not None)
        self._pause_btn.setEnabled(False)

    def set_preview_frame(self, frame_bgr: np.ndarray, frame_index: Optional[int] = None, total_frames: Optional[int] = None):
        self.pause()
        qimg = _bgr_to_qimage(frame_bgr)
        pix = QPixmap.fromImage(qimg)
        self._frame_label.setPixmap(pix.scaled(self._frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if frame_index is not None and total_frames is not None and total_frames > 0:
            self._time_label.setText(f"{frame_index+1} / {total_frames}")

    def resizeEvent(self, event):
        if self._frame_label.pixmap() is not None and not self._frame_label.pixmap().isNull():
            pix = self._frame_label.pixmap()
            self._frame_label.setPixmap(pix.scaled(self._frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)

    def _tick(self):
        if self._cap is None or self._info is None:
            self.pause()
            return
        if self._frame_index >= max(0, self._info.total_frames):
            self.pause()
            return
        self._render_frame_at(self._frame_index + 1)

    def _render_frame_at(self, frame_index: int):
        if self._cap is None or self._info is None:
            return
        frame_index = int(max(0, min(frame_index, max(0, self._info.total_frames - 1))))
        if frame_index != self._frame_index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self.pause()
            return
        self._frame_index = frame_index

        qimg = _bgr_to_qimage(frame)
        pix = QPixmap.fromImage(qimg)
        self._frame_label.setPixmap(pix.scaled(self._frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._update_time_label()

        if self._slider.isEnabled():
            self._updating_slider = True
            self._slider.setValue(self._frame_index)
            self._updating_slider = False
        self.positionChanged.emit(self._frame_index)

    def _update_time_label(self):
        if self._info is None:
            self._time_label.setText("-- / --")
            return
        self._time_label.setText(f"{self._frame_index+1} / {self._info.total_frames}")

    def _on_slider_pressed(self):
        self.pause()

    def _on_slider_released(self):
        self._render_frame_at(self._slider.value())

    def _on_slider_changed(self, value: int):
        if self._updating_slider:
            return
        if self._cap is None:
            return
        self._time_label.setText(f"{value+1} / {self._slider.maximum()+1}")
