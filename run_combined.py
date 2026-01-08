import os
import sys
import argparse
import cv2
from pathlib import Path

from core.ball_detect import ball_detect
from core.pose_detect import PoseDetector
from core.visualize_combined import create_combined_visualization
from core.utils import read_json
from core.court_based_assigner import assign_players_court_based
from core.court_detect import CourtDetector
from core.net_detect import NetDetector
from core.event_detect import EventDetector
from core.export_to_csv import export_to_csv
from core.stroke_classify import create_classifier


def run_combined_pipeline(
    video_path: str,
    result_dir: str,
    model_path: str = "e:\\learn\\TrackNetV3_migrated\\model_best.pth",
    num_frames: int = 3,
    threshold: float = 0.5,
    traj_len: int = 10,
    device: str = 'cuda',
    pose_model: str = 'rtmpose-m',
    use_court_detection: bool = True,
    court_model_path: str = "models/court_kpRCNN.pth",
    court_detection_interval: int = 30
):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    
    print("=" * 60)
    print("TrackNetV3 + MMPose Combined Pipeline")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Result directory: {result_dir}")
    print(f"Model: {model_path}")
    print(f"Number of frames: {num_frames}")
    print(f"Detection threshold: {threshold}")
    print(f"Trajectory length: {traj_len}")
    print(f"Device: {device}")
    print(f"Pose model: {pose_model}")
    print(f"Use court detection: {use_court_detection}")
    print("=" * 60)

    video_result_dir = result_dir / video_name
    video_result_dir.mkdir(parents=True, exist_ok=True)
    
    ball_json_path = video_result_dir / "loca_info" / f"{video_name}.json"
    ball_denoise_json_path = video_result_dir / "loca_info_denoise" / f"{video_name}.json"
    
    poses_path = video_result_dir / f"{video_name}_poses.npy"
    
    ball_video_path = video_result_dir / f"{video_name}_with_trajectory_attention.mp4"
    poses_video_path = video_result_dir / f"{video_name}_with_poses.mp4"
    combined_video_path = video_result_dir / f"{video_name}_combined.mp4"

    court_info = None
    extended_court_points = None
    court_boundary_params = None
    partitioned_keypoints = None
    net_keypoints = None
    
    per_frame_court_keypoints = []
    per_frame_net_keypoints = []
    
    if use_court_detection:
        print("\nStep 0: Court and net detection (per-frame)...")
        court_detector = CourtDetector(model_path=court_model_path, device=device)
        net_detector = NetDetector(model_path="models/net_kpRCNN.pth", device=device)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Total frames: {total_frames}")
        print(f"Detection interval: every {court_detection_interval} frames")
        
        for frame_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                per_frame_court_keypoints.append(None)
                per_frame_net_keypoints.append(None)
                continue
            
            if frame_idx % court_detection_interval == 0:
                court_detector.reset()
                court_info_result, have_court = court_detector.get_court_info(frame)
                
                if have_court:
                    court_info = court_info_result
                    extended_court_points = court_detector._CourtDetector__extended_court_points
                    court_boundary_params = court_detector.get_court_boundary_params()
                    partitioned_keypoints = court_detector.get_partitioned_keypoints()
                else:
                    partitioned_keypoints = None
                
                net_info_result, have_net = net_detector.get_net_info(frame)
                if have_net:
                    net_keypoints = net_detector.get_partitioned_keypoints()
                else:
                    net_keypoints = None
            
            per_frame_court_keypoints.append(partitioned_keypoints)
            per_frame_net_keypoints.append(net_keypoints)
            
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")
        
        cap.release()
        
        successful_detections = sum(1 for kp in per_frame_court_keypoints if kp is not None)
        print(f"Court detection completed!")
        print(f"  - Successful detections: {successful_detections}/{total_frames} frames")
        print(f"  - Detection rate: {successful_detections/total_frames*100:.1f}%")

    print("\nStep 1: Ball detection with TrackNetV3...")
    ball_detect(video_path, str(video_result_dir), model_path, num_frames, threshold)
    print(f"Ball detection completed!")
    print(f"  - Ball JSON: {ball_json_path}")
    print(f"  - Ball video: {ball_video_path}")

    print("\nStep 2: Pose detection with MMPose...")
    detector = PoseDetector(device=device, model=pose_model, use_court_based=use_court_detection)
    
    if use_court_detection and court_boundary_params is not None:
        detector.set_court_info(court_boundary_params, extended_court_points)
    
    poses, video_info = detector.detect_video(video_path)
    detector.save_poses(poses, str(poses_path))
    print(f"Pose detection completed!")
    print(f"  - Poses array: {poses_path}")
    print(f"  - Poses shape: {poses.shape}")

    print("\nStep 2.5: Event detection (hitting frame capture)...")
    ball_data = read_json(str(ball_denoise_json_path))
    trajectory_data = []
    for frame_idx in range(len(ball_data)):
        frame_key = str(frame_idx)
        if frame_key in ball_data:
            frame_data = ball_data[frame_key]
            x = frame_data.get('x', 0)
            y = frame_data.get('y', 0)
            visible = frame_data.get('visible', 0)
            if visible == 1 and x > 0 and y > 0:
                trajectory_data.append([x, y])
            else:
                trajectory_data.append(None)
        else:
            trajectory_data.append(None)

    event_detector = EventDetector(trajectory_data, poses)
    hit_frames, hit_players = event_detector.detect_hits(
        fps=video_info['fps'],
        prominence=1.0,
        angle_threshold=15,
        min_frame_gap=5,
        min_continuation_frames=2,
        min_movement_threshold=5
    )
    
    hit_events_path = video_result_dir / f"{video_name}_hit_events.json"
    event_detector.save_hit_events(str(hit_events_path))
    
    print(f"Event detection completed!")
    print(f"  - Hit frames detected: {len(hit_frames)}")
    print(f"  - Hit events JSON: {hit_events_path}")
    if len(hit_frames) > 0:
        print(f"  - First 5 hit frames: {hit_frames[:5]}")
        print(f"  - First 5 hit players: {hit_players[:5]}")

    print("\nStep 2.6: Stroke type classification...")
    try:
        classifier = create_classifier(dataset='shuttleset', seq_len=100)
        stroke_types = classifier.classify_hits(trajectory_data, poses, hit_frames)
        
        stroke_results_path = video_result_dir / f"{video_name}_stroke_types.json"
        classifier.save_stroke_results(hit_frames, hit_players, stroke_types, str(stroke_results_path))
        
        print(f"Stroke classification completed!")
        print(f"  - Stroke types classified: {len(stroke_types)}")
        print(f"  - Stroke results JSON: {stroke_results_path}")
        if len(stroke_types) > 0:
            print(f"  - First 5 stroke types: {[classifier.get_stroke_type_name(st) for st in stroke_types[:5]]}")
    except Exception as e:
        print(f"Stroke classification failed: {e}")
        print("Continuing without stroke classification...")
        stroke_types = [-1] * len(hit_frames)

    print("\nStep 3: Creating combined visualization...")
    stroke_type_names = None
    if 'stroke_types' in locals() and len(stroke_types) > 0:
        stroke_type_names = [classifier.get_stroke_type_name(st) for st in stroke_types]
    
    create_combined_visualization(
        video_path=video_path,
        ball_json_path=str(ball_denoise_json_path),
        poses_path=str(poses_path),
        output_path=str(combined_video_path),
        traj_len=traj_len,
        court_keypoints=court_info,
        partitioned_keypoints=partitioned_keypoints,
        net_keypoints=net_keypoints,
        hit_frames=hit_frames,
        per_frame_court_keypoints=per_frame_court_keypoints,
        per_frame_net_keypoints=per_frame_net_keypoints,
        stroke_types=stroke_type_names
    )
    print(f"Combined visualization completed!")
    print(f"  - Combined video: {combined_video_path}")

    print("\nStep 4: Exporting data to CSV...")
    csv_path = video_result_dir / f"{video_name}_data.csv"
    stroke_types_path = video_result_dir / f"{video_name}_stroke_types.json"
    if not stroke_types_path.exists():
        stroke_types_path = None
    export_to_csv(
        hit_events_path=str(hit_events_path),
        poses_path=str(poses_path),
        ball_json_path=str(ball_json_path),
        ball_denoise_json_path=str(ball_denoise_json_path),
        output_csv_path=str(csv_path),
        fps=video_info['fps'],
        stroke_types_path=str(stroke_types_path) if stroke_types_path else None
    )
    print(f"CSV export completed!")
    print(f"  - CSV file: {csv_path}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print("\nOutput files:")
    print(f"1. Ball trajectory video: {ball_video_path}")
    print(f"2. Poses array: {poses_path}")
    print(f"3. Hit events JSON: {hit_events_path}")
    print(f"4. Stroke types JSON: {stroke_results_path if 'stroke_results_path' in locals() else 'N/A'}")
    print(f"5. Combined video (ball + poses): {combined_video_path}")
    print(f"6. Data CSV: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run combined TrackNetV3 and MMPose pipeline for badminton video analysis')
    
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--result_dir', type=str, default='./results', help='Result directory')
    parser.add_argument('--model', type=str, default='e:\\learn\\TrackNetV3_migrated\\model_best.pth', help='Model path')
    parser.add_argument('--num_frames', type=int, default=3, help='Number of frames for TrackNetV3')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold (0.0-1.0)')
    parser.add_argument('--traj_len', type=int, default=10, help='Trajectory length to display')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--pose_model', type=str, default='rtmpose-m', 
                        choices=['rtmpose-t', 'rtmpose-s', 'rtmpose-m', 'rtmpose-l', 'rtmpose-x'],
                        help='MMPose model size (t=tiny, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--use_court_detection', action='store_true', default=True,
                        help='Use court detection for player assignment')
    parser.add_argument('--court_model', type=str, default='models/court_kpRCNN.pth',
                        help='Court detection model path')
    parser.add_argument('--court_detection_interval', type=int, default=30,
                        help='Court detection interval in frames (default: 30)')
    
    args = parser.parse_args()
    
    run_combined_pipeline(
        video_path=args.video,
        result_dir=args.result_dir,
        model_path=args.model,
        num_frames=args.num_frames,
        threshold=args.threshold,
        traj_len=args.traj_len,
        device=args.device,
        pose_model=args.pose_model,
        use_court_detection=args.use_court_detection,
        court_model_path=args.court_model,
        court_detection_interval=args.court_detection_interval
    )
