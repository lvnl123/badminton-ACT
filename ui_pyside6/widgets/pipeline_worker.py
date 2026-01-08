from __future__ import annotations

import traceback
from dataclasses import asdict
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from ui_pyside6.widgets.pipeline_runner import PipelineConfig, PipelineOutputs, run_pipeline


class PipelineWorker(QObject):
    logLine = Signal(str)
    stepChanged = Signal(str)
    overallProgressChanged = Signal(int)
    previewFrame = Signal(object)
    outputsReady = Signal(object)
    finished = Signal(bool, str)

    def __init__(self, video_path: str, result_dir: str, config: PipelineConfig):
        super().__init__()
        self._video_path = video_path
        self._result_dir = result_dir
        self._config = config
        self._stop = False

    def request_stop(self):
        self._stop = True

    def _is_stopped(self) -> bool:
        return self._stop

    def run(self):
        try:
            outputs: PipelineOutputs = run_pipeline(
                self._video_path,
                self._result_dir,
                self._config,
                log=self.logLine.emit,
                step=self.stepChanged.emit,
                overall_progress=self.overallProgressChanged.emit,
                preview_frame=self._emit_preview,
                stop_requested=self._is_stopped,
            )
            self.outputsReady.emit(asdict(outputs))
            self.finished.emit(True, "OK")
        except Exception as e:
            msg = str(e)
            if msg == "STOP_REQUESTED":
                self.finished.emit(False, "已停止")
                return
            detail = traceback.format_exc()
            self.logLine.emit(detail)
            self.finished.emit(False, msg)

    def _emit_preview(self, frame_bgr: np.ndarray):
        self.previewFrame.emit(frame_bgr)


class WorkerThread(QThread):
    def __init__(self, worker: PipelineWorker):
        super().__init__()
        self.worker = worker

    def run(self):
        self.worker.run()
