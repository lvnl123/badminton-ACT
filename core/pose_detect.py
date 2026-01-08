import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from .person_tracker import track_poses
from .court_based_assigner import CourtBasedPlayerAssigner


class PoseDetector:
    def __init__(self, device='cuda', model='rtmpose-m', use_court_based=False):
        self.device = device
        self.model = model
        self.use_court_based = use_court_based
        self.inferencer = None
        self.court_assigner = None
        self._init_inferencer()

    def _init_inferencer(self):
        try:
            from mmpose.apis import MMPoseInferencer
            
            model_map = {
                'rtmpose-t': 'rtmpose-t_8xb256-420e_coco-256x192',
                'rtmpose-s': 'rtmpose-s_8xb256-420e_coco-256x192',
                'rtmpose-m': 'rtmpose-m_8xb256-420e_coco-256x192',
                'rtmpose-l': 'rtmpose-l_8xb256-420e_coco-256x192',
            }
            
            pose2d_model = model_map.get(self.model, 'rtmpose-m_8xb256-420e_coco-256x192')
            
            self.inferencer = MMPoseInferencer(
                pose2d=pose2d_model,
                device=self.device,
                show_progress=False
            )
            print(f"MMPose inferencer initialized successfully with model: {pose2d_model}")
        except Exception as e:
            print(f"Failed to initialize MMPose inferencer: {e}")
            self.inferencer = None

    def set_court_info(self, court_info: List[float], extended_court_points: Optional[np.ndarray] = None):
        if self.use_court_based:
            if self.court_assigner is None:
                self.court_assigner = CourtBasedPlayerAssigner(720, 1280)
            self.court_assigner.set_court_info(court_info, extended_court_points)
            print(f"Court info set for court-based assignment: net_y={court_info[4] if len(court_info) >= 5 else 'N/A'}")

    def detect_video(
        self,
        video_path: str,
        frame_callback=None,
        progress_callback=None,
        emit_every_n_frames: int = 1
    ) -> Tuple[np.ndarray, List[Dict]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        all_poses = []
        frame_indices = []

        if self.inferencer is not None:
            try:
                results_generator = self.inferencer(video_path, show=False)
                cap_preview = None
                if frame_callback is not None:
                    cap_preview = cv2.VideoCapture(video_path)
                
                skeleton_pairs = [
                    (0, 1), (0, 2), (1, 3), (2, 4),
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16)
                ]
                
                for frame_idx, result in enumerate(tqdm(results_generator, total=total_frames, desc="Detecting poses")):
                    predictions = result.get('predictions', [])
                    frame_poses = None
                    if predictions and len(predictions) > 0:
                        frame_poses = predictions[0]
                        all_poses.append(frame_poses)
                        frame_indices.append(frame_idx)
                    
                    if cap_preview is not None:
                        ret_preview, frame_preview = cap_preview.read()
                        if ret_preview and emit_every_n_frames > 0 and (frame_idx % emit_every_n_frames == 0):
                            persons = frame_poses if isinstance(frame_poses, list) else ([frame_poses] if frame_poses is not None else [])
                            for person_idx, person in enumerate(persons[:2]):
                                keypoints = person.get('keypoints', None) if isinstance(person, dict) else None
                                if keypoints is None:
                                    continue
                                kp = np.array(keypoints)[:17, :2]
                                for start_idx, end_idx in skeleton_pairs:
                                    pt1 = kp[start_idx]
                                    pt2 = kp[end_idx]
                                    if np.any(pt1) and np.any(pt2):
                                        color = (255, 0, 0) if person_idx == 0 else (0, 0, 255)
                                        cv2.line(frame_preview, tuple(map(int, pt1)), tuple(map(int, pt2)), color, 3)
                                for joint in kp:
                                    if np.any(joint):
                                        color = (255, 0, 0) if person_idx == 0 else (0, 0, 255)
                                        cv2.circle(frame_preview, tuple(map(int, joint)), 4, color, -1)
                            frame_callback(frame_idx, frame_preview, frame_poses)
                    elif frame_callback is not None and emit_every_n_frames > 0 and (frame_idx % emit_every_n_frames == 0):
                        frame_callback(frame_idx, None, frame_poses)

                    if progress_callback is not None:
                        progress_callback(frame_idx + 1, total_frames)
                
                if cap_preview is not None:
                    cap_preview.release()
            except Exception as e:
                print(f"Error during pose detection: {e}")
                all_poses = []
                frame_indices = []
        else:
            print("MMPose inferencer not available, skipping pose detection")

        cap.release()

        poses_array = self._poses_to_array(all_poses, total_frames)
        
        return poses_array, {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'frame_indices': frame_indices
        }

    def _poses_to_array(self, poses_list: List[Dict], total_frames: int) -> np.ndarray:
        all_frame_poses = []

        for frame_poses in poses_list:
            persons = frame_poses if isinstance(frame_poses, list) else [frame_poses]
            
            valid_persons = []
            for person in persons:
                keypoints = person.get('keypoints', None)
                if keypoints is not None and len(keypoints) >= 17:
                    valid_persons.append(np.array(keypoints)[:17, :2])
            
            if self.use_court_based and self.court_assigner is not None and len(valid_persons) >= 2:
                top_player, bottom_player = self.court_assigner.assign_players(valid_persons)
                if top_player is not None and bottom_player is not None:
                    all_frame_poses.append([top_player, bottom_player])
                else:
                    court_persons = []
                    for person in valid_persons:
                        if self.court_assigner.is_in_court(person):
                            court_persons.append(person)
                    all_frame_poses.append(court_persons)
            elif self.use_court_based and self.court_assigner is not None:
                court_persons = []
                for person in valid_persons:
                    if self.court_assigner.is_in_court(person):
                        court_persons.append(person)
                all_frame_poses.append(court_persons)
            else:
                all_frame_poses.append(valid_persons)

        poses_array = track_poses(all_frame_poses, max_persons=2)

        return poses_array

    def save_poses(self, poses: np.ndarray, output_path: str):
        np.save(output_path, poses)
        print(f"Poses saved to {output_path}")

    def load_poses(self, poses_path: str) -> np.ndarray:
        poses = np.load(poses_path)
        print(f"Poses loaded from {poses_path}")
        return poses

    def visualize_poses(self, video_path: str, poses: np.ndarray, output_path: str, skeleton_pairs=None):
        if skeleton_pairs is None:
            skeleton_pairs = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        total_frames = poses.shape[0]

        with tqdm(total=total_frames, desc="Visualizing poses") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx < total_frames:
                    for person_idx in range(2):
                        person_poses = poses[frame_idx, person_idx]
                        
                        for start_idx, end_idx in skeleton_pairs:
                            pt1 = person_poses[start_idx]
                            pt2 = person_poses[end_idx]
                            
                            if np.any(pt1) and np.any(pt2):
                                color = (0, 255, 0) if person_idx == 0 else (0, 0, 255)
                                pt1 = tuple(map(int, pt1))
                                pt2 = tuple(map(int, pt2))
                                cv2.line(frame, pt1, pt2, color, 2)

                        for joint_idx in range(17):
                            joint = person_poses[joint_idx]
                            if np.any(joint):
                                color = (0, 255, 0) if person_idx == 0 else (0, 0, 255)
                                joint = tuple(map(int, joint))
                                cv2.circle(frame, joint, 4, color, -1)

                out.write(frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()
        print(f"Pose visualization saved to {output_path}")


def detect_poses_video(video_path: str, output_dir: str, device='cuda'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    poses_path = output_dir / f"{video_name}_poses.npy"
    vis_path = output_dir / f"{video_name}_with_poses.mp4"

    detector = PoseDetector(device=device)
    
    print(f"Detecting poses in {video_path}...")
    poses, video_info = detector.detect_video(video_path)
    
    detector.save_poses(poses, str(poses_path))
    
    print(f"Visualizing poses...")
    detector.visualize_poses(video_path, poses, str(vis_path))
    
    print(f"Pose detection complete!")
    print(f"Video info: {video_info}")
    print(f"Poses shape: {poses.shape}")
    
    return poses, video_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect poses in badminton video')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    detect_poses_video(args.video, args.output_dir, args.device)
