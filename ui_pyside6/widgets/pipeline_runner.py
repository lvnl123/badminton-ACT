from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from core.ball_detect import ball_detect
from core.court_detect import CourtDetector
from core.event_detect import EventDetector
from core.export_to_csv import export_to_csv
from core.net_detect import NetDetector
from core.pose_detect import PoseDetector
from core.stroke_classify import create_classifier
from core.utils import read_json
from core.visualize_combined import load_ball_positions, visualize_combined


@dataclass(frozen=True)
class PipelineConfig:
    model_path: str
    num_frames: int
    threshold: float
    traj_len: int
    device: str
    pose_model: str
    use_court_detection: bool
    court_model_path: str
    net_model_path: str
    court_detection_interval: int
    pose_emit_every_n: int
    viz_emit_every_n: int
    dataset: str
    stroke_seq_len: int


@dataclass(frozen=True)
class PipelineOutputs:
    video_name: str
    video_result_dir: str
    ball_json_path: str
    ball_denoise_json_path: str
    poses_path: str
    hit_events_path: str
    stroke_results_path: Optional[str]
    combined_video_path: str
    csv_path: str


def run_pipeline(
    video_path: str,
    result_dir: str,
    config: PipelineConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
    step: Optional[Callable[[str], None]] = None,
    overall_progress: Optional[Callable[[int], None]] = None,
    preview_frame: Optional[Callable[[np.ndarray], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> PipelineOutputs:
    def _log(message: str):
        if log is not None:
            log(message)

    def _step(name: str):
        if step is not None:
            step(name)

    def _set_overall(p: int):
        if overall_progress is not None:
            overall_progress(int(max(0, min(100, p))))

    def _stopping() -> bool:
        return bool(stop_requested and stop_requested())

    result_dir_path = Path(result_dir).expanduser()
    if not result_dir_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        result_dir_path = project_root / result_dir_path
    result_dir_path = result_dir_path.resolve(strict=False)
    result_dir_path.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem
    video_result_dir = result_dir_path / video_name
    video_result_dir.mkdir(parents=True, exist_ok=True)

    ball_json_path = video_result_dir / "loca_info" / f"{video_name}.json"
    ball_denoise_json_path = video_result_dir / "loca_info_denoise" / f"{video_name}.json"
    poses_path = video_result_dir / f"{video_name}_poses.npy"
    combined_video_path = video_result_dir / f"{video_name}_combined.mp4"
    hit_events_path = video_result_dir / f"{video_name}_hit_events.json"
    stroke_results_path = video_result_dir / f"{video_name}_stroke_types.json"
    csv_path = video_result_dir / f"{video_name}_data.csv"

    court_info = None
    extended_court_points = None
    court_boundary_params = None
    partitioned_keypoints = None
    net_keypoints = None
    per_frame_court_keypoints = []
    per_frame_net_keypoints = []

    if config.use_court_detection:
        _step("球场/球网检测")
        _log("开始逐帧球场/球网检测…")

        court_detector = CourtDetector(model_path=config.court_model_path, device=config.device)
        net_detector = NetDetector(model_path=config.net_model_path, device=config.device)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _log(f"总帧数: {total_frames}，间隔: {config.court_detection_interval} 帧")

        current_court = None
        current_net = None

        for frame_idx in range(total_frames):
            if _stopping():
                cap.release()
                raise RuntimeError("STOP_REQUESTED")

            ret, frame = cap.read()
            if not ret:
                per_frame_court_keypoints.append(None)
                per_frame_net_keypoints.append(None)
                continue

            if frame_idx % max(1, config.court_detection_interval) == 0:
                court_detector.reset()
                court_info_result, have_court = court_detector.get_court_info(frame)
                if have_court:
                    court_info = court_info_result
                    extended_court_points = court_detector._CourtDetector__extended_court_points
                    court_boundary_params = court_detector.get_court_boundary_params()
                    partitioned_keypoints = court_detector.get_partitioned_keypoints()
                    current_court = partitioned_keypoints
                else:
                    current_court = None
                    partitioned_keypoints = None

                net_info_result, have_net = net_detector.get_net_info(frame)
                if have_net:
                    net_keypoints = net_detector.get_partitioned_keypoints()
                    current_net = net_keypoints
                else:
                    current_net = None
                    net_keypoints = None

            per_frame_court_keypoints.append(current_court)
            per_frame_net_keypoints.append(current_net)

            if preview_frame is not None and config.pose_emit_every_n > 0 and frame_idx % config.pose_emit_every_n == 0:
                preview = frame.copy()
                if current_court is not None:
                    preview = court_detector.draw_court(preview, mode="frame_select")
                if current_net is not None:
                    preview = net_detector.draw_net(preview, mode="frame_select")
                preview_frame(preview)

            _set_overall(int(frame_idx / max(1, total_frames) * 10))

        cap.release()
        _log("球场/球网检测完成")

    _step("羽毛球检测")
    _log("开始 TrackNetV3 羽毛球检测…")

    def _ball_frame_cb(frame_idx: int, frame_bgr: np.ndarray, ball_pos, visible: int):
        if preview_frame is None:
            return
        if config.pose_emit_every_n <= 0 or frame_idx % config.pose_emit_every_n != 0:
            return
        img = frame_bgr.copy()
        if visible == 1 and ball_pos is not None:
            cv2.circle(img, (int(ball_pos[0]), int(ball_pos[1])), 10, (0, 255, 255), -1)
            cv2.circle(img, (int(ball_pos[0]), int(ball_pos[1])), 16, (255, 255, 255), 2)
        preview_frame(img)

    def _ball_progress(processed: int, total: int):
        base = 10
        span = 35
        _set_overall(base + int(processed / max(1, total) * span))

    ball_detect(
        video_path,
        str(video_result_dir),
        config.model_path,
        config.num_frames,
        config.threshold,
        frame_callback=_ball_frame_cb,
        progress_callback=_ball_progress,
    )

    if _stopping():
        raise RuntimeError("STOP_REQUESTED")

    _log("羽毛球检测完成")

    _step("姿态检测")
    _log("开始 MMPose 姿态检测…")

    detector = PoseDetector(device=config.device, model=config.pose_model, use_court_based=config.use_court_detection)
    if config.use_court_detection and court_boundary_params is not None:
        detector.set_court_info(court_boundary_params, extended_court_points)

    def _pose_preview(frame_idx: int, frame_bgr: Optional[np.ndarray], frame_poses):
        if preview_frame is None:
            return
        if frame_bgr is None:
            return
        preview_frame(frame_bgr)

    def _pose_progress(processed: int, total: int):
        base = 45
        span = 35
        _set_overall(base + int(processed / max(1, total) * span))

    poses, video_info = detector.detect_video(
        video_path,
        frame_callback=_pose_preview,
        progress_callback=_pose_progress,
        emit_every_n_frames=max(1, config.pose_emit_every_n),
    )
    detector.save_poses(poses, str(poses_path))
    _log(f"姿态数组: {poses.shape}")

    if _stopping():
        raise RuntimeError("STOP_REQUESTED")

    _step("事件检测")
    _log("开始击球事件检测…")

    ball_data = read_json(str(ball_denoise_json_path))
    trajectory_data = []
    for frame_idx in range(len(ball_data)):
        frame_key = str(frame_idx)
        frame_data = ball_data.get(frame_key, None)
        if frame_data is None:
            trajectory_data.append(None)
            continue
        x = frame_data.get("x", 0)
        y = frame_data.get("y", 0)
        visible = frame_data.get("visible", 0)
        if visible == 1 and x > 0 and y > 0:
            trajectory_data.append([x, y])
        else:
            trajectory_data.append(None)

    event_detector = EventDetector(trajectory_data, poses)
    hit_frames, hit_players = event_detector.detect_hits(
        fps=video_info["fps"],
        prominence=1.0,
        angle_threshold=15,
        min_frame_gap=5,
        min_continuation_frames=2,
        min_movement_threshold=5,
    )
    event_detector.save_hit_events(str(hit_events_path))
    _log(f"击球次数: {len(hit_frames)}")
    _set_overall(82)

    if _stopping():
        raise RuntimeError("STOP_REQUESTED")

    _step("击球类型识别")
    _log("开始击球类型识别…")
    stroke_type_names = None
    stroke_results_written = False
    try:
        classifier = create_classifier(dataset=config.dataset, seq_len=config.stroke_seq_len)
        stroke_types = classifier.classify_hits(trajectory_data, poses, hit_frames)
        classifier.save_stroke_results(hit_frames, hit_players, stroke_types, str(stroke_results_path))
        stroke_type_names = [classifier.get_stroke_type_name(st) for st in stroke_types]
        stroke_results_written = True
        _log("击球类型识别完成")
    except Exception as e:
        _log(f"击球类型识别失败: {e}")
        stroke_type_names = None
        stroke_results_written = False
    _set_overall(86)

    if _stopping():
        raise RuntimeError("STOP_REQUESTED")

    _step("合成可视化")
    _log("开始生成合成可视化视频…")

    ball_positions = load_ball_positions(str(ball_denoise_json_path))

    def _viz_frame_cb(frame_idx: int, frame_bgr: np.ndarray):
        if preview_frame is None:
            return
        preview_frame(frame_bgr)

    def _viz_progress(processed: int, total: int):
        base = 86
        span = 12
        _set_overall(base + int(processed / max(1, total) * span))

    visualize_combined(
        video_path=video_path,
        ball_positions=ball_positions,
        poses=poses,
        output_path=str(combined_video_path),
        traj_len=config.traj_len,
        court_keypoints=court_info,
        partitioned_keypoints=partitioned_keypoints,
        net_keypoints=net_keypoints,
        hit_frames=hit_frames,
        per_frame_court_keypoints=per_frame_court_keypoints if config.use_court_detection else None,
        per_frame_net_keypoints=per_frame_net_keypoints if config.use_court_detection else None,
        stroke_types=stroke_type_names,
        frame_callback=_viz_frame_cb,
        progress_callback=_viz_progress,
        emit_every_n_frames=max(1, config.viz_emit_every_n),
    )

    if _stopping():
        raise RuntimeError("STOP_REQUESTED")

    _step("导出数据")
    _log("开始导出 CSV…")
    export_to_csv(
        hit_events_path=str(hit_events_path),
        poses_path=str(poses_path),
        ball_json_path=str(ball_json_path),
        ball_denoise_json_path=str(ball_denoise_json_path),
        output_csv_path=str(csv_path),
        fps=video_info["fps"],
        stroke_types_path=str(stroke_results_path) if stroke_results_written else None,
    )
    _set_overall(100)

    return PipelineOutputs(
        video_name=video_name,
        video_result_dir=str(video_result_dir),
        ball_json_path=str(ball_json_path),
        ball_denoise_json_path=str(ball_denoise_json_path),
        poses_path=str(poses_path),
        hit_events_path=str(hit_events_path),
        stroke_results_path=str(stroke_results_path) if stroke_results_written else None,
        combined_video_path=str(combined_video_path),
        csv_path=str(csv_path),
    )
