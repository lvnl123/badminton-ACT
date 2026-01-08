import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def _put_chinese_text(frame: np.ndarray, text: str, position: Tuple[int, int], 
                      font_size: int = 40, color: Tuple[int, int, int] = (0, 255, 0), 
                      font_path: str = None) -> np.ndarray:
    if font_path is None:
        font_path = r"C:\Windows\Fonts\msyh.ttc"
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text(position, text, font=font, fill=color)
        
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame_bgr
    except Exception as e:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, position, font, font_size / 30, color, 3, cv2.LINE_AA)
        return frame


def visualize_combined(
    video_path: str,
    ball_positions: List[Tuple[int, int]],
    poses: np.ndarray,
    output_path: str,
    traj_len: int = 10,
    skeleton_pairs=None,
    court_keypoints=None,
    partitioned_keypoints=None,
    net_keypoints: Optional[List[List[int]]] = None,
    hit_frames: Optional[List[int]] = None,
    per_frame_court_keypoints: Optional[List[Optional[List[List[int]]]]] = None,
    per_frame_net_keypoints: Optional[List[Optional[List[List[int]]]]] = None,
    stroke_types: Optional[List[str]] = None,
    frame_callback=None,
    progress_callback=None,
    emit_every_n_frames: int = 1
):
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

    court_zones = None
    
    current_partitioned_keypoints = partitioned_keypoints
    current_net_keypoints = net_keypoints
    
    if per_frame_court_keypoints is not None and len(per_frame_court_keypoints) > 0:
        if per_frame_court_keypoints[0] is not None:
            court_zones = _extract_court_zones(per_frame_court_keypoints[0])
    elif partitioned_keypoints is not None:
        court_zones = _extract_court_zones(partitioned_keypoints)
    
    ball_speeds = _calculate_ball_speeds(ball_positions, fps)
    speed_thresholds = _calculate_speed_thresholds(ball_speeds)

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Creating combined visualization") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if per_frame_court_keypoints is not None and frame_idx < len(per_frame_court_keypoints):
                current_partitioned_keypoints = per_frame_court_keypoints[frame_idx]
            else:
                current_partitioned_keypoints = partitioned_keypoints
            
            if per_frame_net_keypoints is not None and frame_idx < len(per_frame_net_keypoints):
                current_net_keypoints = per_frame_net_keypoints[frame_idx]
            else:
                current_net_keypoints = net_keypoints
            
            if current_partitioned_keypoints is not None:
                court_zones = _extract_court_zones(current_partitioned_keypoints)
            else:
                court_zones = None

            if current_partitioned_keypoints is not None or current_net_keypoints is not None:
                if court_zones is not None and frame_idx < poses.shape[0]:
                    _highlight_player_zones(frame, poses[frame_idx], court_zones)

                if current_partitioned_keypoints is not None:
                    c_edges = [[0, 1], [0, 5], [1, 2], [1, 6], [2, 3], [2, 7], [3, 4],
                               [3, 8], [4, 9], [5, 6], [5, 10], [6, 7], [6, 11], [7, 8],
                               [7, 12], [8, 9], [8, 13], [9, 14], [10, 11], [10, 15],
                               [11, 12], [11, 16], [12, 13], [12, 17], [13, 14], [13, 18],
                               [14, 19], [15, 16], [15, 20], [16, 17], [16, 21], [17, 18],
                               [17, 22], [18, 19], [18, 23], [19, 24], [20, 21], [20, 25],
                               [21, 22], [21, 26], [22, 23], [22, 27], [23, 24], [23, 28],
                               [24, 29], [25, 26], [25, 30], [26, 27], [26, 31], [27, 28],
                               [27, 32], [28, 29], [28, 33], [29, 34], [30, 31], [31, 32],
                               [32, 33], [33, 34]]
                    court_color_edge = (53, 195, 242)
                    court_color_kps = (5, 135, 242)

                    for e in c_edges:
                        cv2.line(frame, (int(current_partitioned_keypoints[e[0]][0]),
                                         int(current_partitioned_keypoints[e[0]][1])),
                                 (int(current_partitioned_keypoints[e[1]][0]),
                                  int(current_partitioned_keypoints[e[1]][1])),
                                 court_color_edge, 2, lineType=cv2.LINE_AA)

                    for kp in current_partitioned_keypoints:
                        cv2.circle(frame, tuple(kp), 1, court_color_kps, 5)

                if current_net_keypoints is not None:
                    net_edges = [[0, 1], [2, 3], [0, 4], [1, 5]]
                    net_color_edge = (255, 165, 0)
                    net_color_kps = (255, 140, 0)

                    for e in net_edges:
                        cv2.line(frame, (int(current_net_keypoints[e[0]][0]),
                                         int(current_net_keypoints[e[0]][1])),
                                 (int(current_net_keypoints[e[1]][0]),
                                  int(current_net_keypoints[e[1]][1])),
                                 net_color_edge, 2, lineType=cv2.LINE_AA)

                    for kp in current_net_keypoints:
                        cv2.circle(frame, tuple(kp), 1, net_color_kps, 5)

            if frame_idx < len(ball_positions) and frame_idx < poses.shape[0]:
                ball_pos = ball_positions[frame_idx]
                
                if ball_pos is not None and ball_pos[0] > 0 and ball_pos[1] > 0:
                    start_idx = max(0, frame_idx - traj_len)
                    for j in range(start_idx, frame_idx):
                        prev_ball = ball_positions[j]
                        if prev_ball is not None and prev_ball[0] > 0 and prev_ball[1] > 0:
                            alpha = 1.0 - ((frame_idx - j) / traj_len)
                            speed = ball_speeds[j] if j < len(ball_speeds) else 0
                            color = _get_speed_color(speed, speed_thresholds, alpha)
                            cv2.circle(frame, tuple(map(int, prev_ball)), 4, color, -1)
                    
                    current_speed = ball_speeds[frame_idx] if frame_idx < len(ball_speeds) else 0
                    current_color = _get_speed_color(current_speed, speed_thresholds, 1.0)
                    cv2.circle(frame, tuple(map(int, ball_pos)), 8, current_color, -1)
                    cv2.circle(frame, tuple(map(int, ball_pos)), 12, (255, 255, 255), 2)

                for person_idx in range(2):
                    person_poses = poses[frame_idx, person_idx]
                    
                    for start_idx, end_idx in skeleton_pairs:
                        pt1 = person_poses[start_idx]
                        pt2 = person_poses[end_idx]
                        
                        if np.any(pt1) and np.any(pt2):
                            color = (255, 0, 0) if person_idx == 0 else (0, 0, 255)
                            pt1 = tuple(map(int, pt1))
                            pt2 = tuple(map(int, pt2))
                            cv2.line(frame, pt1, pt2, color, 4)

                    for joint_idx in range(17):
                        joint = person_poses[joint_idx]
                        if np.any(joint):
                            color = (255, 0, 0) if person_idx == 0 else (0, 0, 255)
                            joint = tuple(map(int, joint))
                            cv2.circle(frame, joint, 5, color, -1)

            if hit_frames is not None:
                current_hit_count = sum(1 for hit_frame in hit_frames if hit_frame <= frame_idx)
                text = f"Hits: {current_hit_count}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = width - text_size[0] - 20
                text_y = 60
                
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

            if stroke_types is not None and hit_frames is not None:
                current_stroke_type = None
                for hit_frame, stroke_type in zip(hit_frames, stroke_types):
                    if hit_frame <= frame_idx:
                        current_stroke_type = stroke_type
                
                if current_stroke_type is not None:
                    stroke_text = f"击球类型: {current_stroke_type}"
                    frame = _put_chinese_text(frame, stroke_text, (20, 20), font_size=40, color=(0, 255, 0))

            if frame_callback is not None and emit_every_n_frames > 0 and (frame_idx % emit_every_n_frames == 0):
                frame_callback(frame_idx, frame)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)

    cap.release()
    out.release()
    print(f"Combined visualization saved to {output_path}")


def _extract_court_zones(partitioned_keypoints: List[List[int]]) -> List[List[Tuple[int, int]]]:
    c_edges = [[0, 1], [0, 5], [1, 2], [1, 6], [2, 3], [2, 7], [3, 4],
               [3, 8], [4, 9], [5, 6], [5, 10], [6, 7], [6, 11], [7, 8],
               [7, 12], [8, 9], [8, 13], [9, 14], [10, 11], [10, 15],
               [11, 12], [11, 16], [12, 13], [12, 17], [13, 14], [13, 18],
               [14, 19], [15, 16], [15, 20], [16, 17], [16, 21], [17, 18],
               [17, 22], [18, 19], [18, 23], [19, 24], [20, 21], [20, 25],
               [21, 22], [21, 26], [22, 23], [22, 27], [23, 24], [23, 28],
               [24, 29], [25, 26], [25, 30], [26, 27], [26, 31], [27, 28],
               [27, 32], [28, 29], [28, 33], [29, 34], [30, 31], [31, 32],
               [32, 33], [33, 34]]

    zones = []
    zone_edges = [
        [0, 5, 6, 1],
        [5, 10, 11, 6],
        [10, 15, 16, 11],
        [15, 20, 21, 16],
        [20, 25, 26, 21],
        [25, 30, 31, 26],
        [30, 31, 32, 33],
        [31, 26, 27, 32],
        [26, 21, 22, 27],
        [21, 16, 17, 22],
        [16, 11, 12, 17],
        [11, 6, 7, 12],
        [6, 1, 2, 7],
        [1, 2, 3, 8],
        [2, 3, 4, 9],
        [3, 8, 9, 4],
        [8, 13, 14, 9],
        [13, 18, 19, 14],
        [18, 23, 24, 19],
        [23, 28, 29, 24],
        [28, 33, 34, 29],
        [33, 32, 27, 28],
        [32, 27, 22, 23],
        [27, 22, 17, 18],
        [22, 17, 12, 13],
        [17, 12, 7, 8]
    ]

    for zone in zone_edges:
        zone_points = []
        for idx in zone:
            zone_points.append((partitioned_keypoints[idx][0], partitioned_keypoints[idx][1]))
        zones.append(zone_points)

    return zones


def _point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def _highlight_player_zones(frame: np.ndarray, poses: np.ndarray, court_zones: List[List[Tuple[int, int]]]):
    for person_idx in range(2):
        person_poses = poses[person_idx]
        
        ankle_left = person_poses[15]
        ankle_right = person_poses[16]
        
        color = (0, 255, 0) if person_idx == 0 else (255, 165, 0)
        
        if np.any(ankle_left):
            ankle_left_point = (int(ankle_left[0]), int(ankle_left[1]))
            
            for zone in court_zones:
                if _point_in_polygon(ankle_left_point, zone):
                    overlay = frame.copy()
                    pts = np.array(zone, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        if np.any(ankle_right):
            ankle_right_point = (int(ankle_right[0]), int(ankle_right[1]))
            
            for zone in court_zones:
                if _point_in_polygon(ankle_right_point, zone):
                    overlay = frame.copy()
                    pts = np.array(zone, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


def _calculate_ball_speeds(ball_positions: List[Tuple[int, int]], fps: float) -> List[float]:
    speeds = []
    
    for i in range(len(ball_positions)):
        if i == 0:
            speeds.append(0)
            continue
        
        curr_pos = ball_positions[i]
        prev_pos = ball_positions[i - 1]
        
        if curr_pos is None or prev_pos is None:
            speeds.append(0)
            continue
        
        if curr_pos[0] <= 0 or curr_pos[1] <= 0 or prev_pos[0] <= 0 or prev_pos[1] <= 0:
            speeds.append(0)
            continue
        
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        speed = distance * fps
        speeds.append(speed)
    
    return speeds


def _calculate_speed_thresholds(speeds: List[float]) -> Tuple[float, float, float]:
    valid_speeds = [s for s in speeds if s > 0]
    
    if len(valid_speeds) == 0:
        return (100, 300, 600)
    
    q25 = np.percentile(valid_speeds, 25)
    q50 = np.percentile(valid_speeds, 50)
    q75 = np.percentile(valid_speeds, 75)
    
    slow_threshold = q25
    medium_threshold = q50
    fast_threshold = q75
    
    return (slow_threshold, medium_threshold, fast_threshold)


def _get_speed_color(speed: float, thresholds: Tuple[float, float, float], alpha: float = 1.0) -> Tuple[int, int, int]:
    slow_threshold, medium_threshold, fast_threshold = thresholds
    
    if speed < slow_threshold:
        base_color = (0, 255, 0)
    elif speed < medium_threshold:
        base_color = (0, 255, 255)
    elif speed < fast_threshold:
        base_color = (0, 0, 255)
    else:
        base_color = (255, 0, 255)
    
    color = (int(base_color[0] * alpha), int(base_color[1] * alpha), int(base_color[2] * alpha))
    
    return color


def load_ball_positions(json_path: str) -> List[Tuple[int, int]]:
    import json
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    ball_positions = []
    
    for frame_idx in sorted(data.keys(), key=lambda x: int(x)):
        frame_data = data[frame_idx]
        x = frame_data.get('x', None)
        y = frame_data.get('y', None)
        visible = frame_data.get('visible', 0)
        
        if x is not None and y is not None and visible == 1:
            ball_positions.append((x, y))
        else:
            ball_positions.append(None)
    
    return ball_positions


def create_combined_visualization(
    video_path: str,
    ball_json_path: str,
    poses_path: str,
    output_path: str,
    traj_len: int = 10,
    court_keypoints=None,
    partitioned_keypoints=None,
    net_keypoints: Optional[List[List[int]]] = None,
    hit_frames: Optional[List[int]] = None,
    per_frame_court_keypoints: Optional[List[Optional[List[List[int]]]]] = None,
    per_frame_net_keypoints: Optional[List[Optional[List[List[int]]]]] = None,
    stroke_types: Optional[List[str]] = None
):
    ball_positions = load_ball_positions(ball_json_path)
    poses = np.load(poses_path)
    
    print(f"球网关键点数量: {len(net_keypoints) if net_keypoints is not None else 0}")
    print(f"羽毛球场关键点数量: {len(partitioned_keypoints) if partitioned_keypoints is not None else 0}")
    print(f"每帧球场关键点: {len(per_frame_court_keypoints) if per_frame_court_keypoints is not None else 0} 帧")
    print(f"每帧球网关键点: {len(per_frame_net_keypoints) if per_frame_net_keypoints is not None else 0} 帧")
    
    visualize_combined(
        video_path=video_path,
        ball_positions=ball_positions,
        poses=poses,
        output_path=output_path,
        traj_len=traj_len,
        court_keypoints=court_keypoints,
        partitioned_keypoints=partitioned_keypoints,
        net_keypoints=net_keypoints,
        hit_frames=hit_frames,
        per_frame_court_keypoints=per_frame_court_keypoints,
        per_frame_net_keypoints=per_frame_net_keypoints,
        stroke_types=stroke_types
    )


def load_court_keypoints(json_path: str) -> Optional[List[List[int]]]:
    import json
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'court_keypoints' in data:
            return data['court_keypoints']
        elif 'partitioned_keypoints' in data:
            return data['partitioned_keypoints']
        return None
    except Exception as e:
        print(f"Warning: Could not load court keypoints from {json_path}: {e}")
        return None


def load_net_keypoints(json_path: str) -> Optional[List[List[int]]]:
    import json
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'net_keypoints' in data:
            return data['net_keypoints']
        return None
    except Exception as e:
        print(f"Warning: Could not load net keypoints from {json_path}: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create combined visualization of ball trajectory and player poses')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--ball_json', type=str, required=True, help='Ball detection JSON path')
    parser.add_argument('--poses', type=str, required=True, help='Poses numpy array path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--traj_len', type=int, default=10, help='Trajectory length to display')
    parser.add_argument('--court_json', type=str, default=None, help='Court keypoints JSON path')
    parser.add_argument('--net_json', type=str, default=None, help='Net keypoints JSON path')
    
    args = parser.parse_args()
    
    court_keypoints = None
    partitioned_keypoints = None
    net_keypoints = None
    
    if args.court_json:
        partitioned_keypoints = load_court_keypoints(args.court_json)
    
    if args.net_json:
        net_keypoints = load_net_keypoints(args.net_json)
    
    create_combined_visualization(
        args.video,
        args.ball_json,
        args.poses,
        args.output,
        args.traj_len,
        court_keypoints,
        partitioned_keypoints,
        net_keypoints
    )
