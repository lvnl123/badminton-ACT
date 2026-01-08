import numpy as np
from scipy.signal import find_peaks
import json
import os


class EventDetector:
    def __init__(self, trajectory_data, poses=None):
        self.trajectory_data = trajectory_data
        self.poses = poses
        self.hit_frames = []
        self.hit_players = []

    def detect_hits(self, fps=25, prominence=2, angle_threshold=30, velocity_threshold=3, min_frame_gap=13, min_continuation_frames=5, min_movement_threshold=20):
        frames = []
        realx = []
        realy = []

        for frame_idx, data in enumerate(self.trajectory_data):
            if data is not None and len(data) >= 2:
                x, y = data
                if x > 0 and y > 0:
                    frames.append(frame_idx)
                    realx.append(x)
                    realy.append(y)

        if len(frames) == 0:
            print("No valid trajectory points found!")
            return [], []

        frames = np.array(frames)
        realx = np.array(realx)
        realy = np.array(realy)

        points = np.column_stack([realx, realy, frames])
        x, y, z = points.T

        hit_indices = []

        peaks, properties = find_peaks(y, prominence=prominence)
        valleys, _ = find_peaks(-y, prominence=prominence)
        
        print(f"Detected {len(peaks)} peaks and {len(valleys)} valleys with prominence={prominence}")

        for peak_idx in peaks:
            if peak_idx < 2 or peak_idx >= len(y) - 2:
                continue

            prev_slope = y[peak_idx] - y[peak_idx - 1]
            next_slope = y[peak_idx + 1] - y[peak_idx]
            
            if prev_slope > 0 and next_slope < 0:
                angle = self._calculate_angle(
                    [x[peak_idx - 1], y[peak_idx - 1], x[peak_idx], y[peak_idx]],
                    [x[peak_idx], y[peak_idx], x[peak_idx + 1], y[peak_idx + 1]]
                )
                
                if angle > angle_threshold:
                    hit_indices.append(peak_idx)

        for valley_idx in valleys:
            if valley_idx < 2 or valley_idx >= len(y) - 2:
                continue

            prev_slope = y[valley_idx] - y[valley_idx - 1]
            next_slope = y[valley_idx + 1] - y[valley_idx]
            
            if prev_slope < 0 and next_slope > 0:
                angle = self._calculate_angle(
                    [x[valley_idx - 1], y[valley_idx - 1], x[valley_idx], y[valley_idx]],
                    [x[valley_idx], y[valley_idx], x[valley_idx + 1], y[valley_idx + 1]]
                )
                
                if angle > angle_threshold:
                    hit_indices.append(valley_idx)

        hit_indices = sorted(list(set(hit_indices)))
        hit_frames = [int(frames[i]) for i in hit_indices]

        hit_frames = self._merge_consecutive_hits(hit_frames, min_frame_gap=min_frame_gap)

        hit_frames = self._validate_hit_continuation(hit_frames, min_continuation_frames=min_continuation_frames, min_movement_threshold=min_movement_threshold)

        landing_frame = self._detect_landing_frame()
        if landing_frame is not None:
            hit_frames = [f for f in hit_frames if f < landing_frame]
            print(f"Filtered out hits after landing frame {landing_frame}")

        if self.poses is not None:
            hit_players = self._filter_hits_by_pose(hit_frames)
        else:
            hit_players = [1] * len(hit_frames)

        self.hit_frames = hit_frames
        self.hit_players = hit_players

        return hit_frames, hit_players

    def _calculate_angle(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        vec1 = np.array([x2 - x1, y2 - y1])
        vec2 = np.array([x4 - x3, y4 - y3])

        unit_vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        unit_vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

        dot_product = np.dot(unit_vec1, unit_vec2)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        angle = np.degrees(np.arccos(dot_product))
        return angle

    def _filter_hits_by_pose(self, hit_frames):
        hit_players = []

        for frame_idx in hit_frames:
            if frame_idx >= len(self.trajectory_data):
                hit_players.append(0)
                continue

            trajectory_point = self.trajectory_data[frame_idx]
            if trajectory_point is None or len(trajectory_point) < 2:
                hit_players.append(0)
                continue

            ball_pos = np.array(trajectory_point[:2])

            reached_by = 0
            dist_reached = 1e99

            if self.poses is not None and frame_idx < len(self.poses):
                for player_idx in range(min(2, self.poses.shape[1])):
                    pose_data = self.poses[frame_idx, player_idx]

                    if pose_data is None:
                        continue

                    pose_centroid = self._get_pose_centroid(pose_data)

                    if pose_centroid is not None:
                        dist = np.linalg.norm(ball_pos - pose_centroid)
                        if dist < dist_reached:
                            dist_reached = dist
                            reached_by = player_idx + 1

            hit_players.append(reached_by)

        return hit_players

    def _get_pose_centroid(self, pose_data):
        valid_points = []

        for i in range(pose_data.shape[0]):
            x, y = pose_data[i, 0], pose_data[i, 1]
            if x > 0 and y > 0:
                valid_points.append([x, y])

        if len(valid_points) > 0:
            return np.mean(valid_points, axis=0)
        return None

    def _validate_hit_continuation(self, hit_frames, min_continuation_frames=5, min_movement_threshold=20):
        validated_hits = []
        
        for hit_frame in hit_frames:
            if hit_frame >= len(self.trajectory_data):
                continue
            
            hit_data = self.trajectory_data[hit_frame]
            if hit_data is None or len(hit_data) < 2:
                continue
            
            hit_x = hit_data[0]
            hit_y = hit_data[1]
            
            if hit_x <= 0 or hit_y <= 0:
                continue
            
            movement_count = 0
            
            for i in range(1, min_continuation_frames + 1):
                if hit_frame + i >= len(self.trajectory_data):
                    break
                
                next_data = self.trajectory_data[hit_frame + i]
                if next_data is None or len(next_data) < 2:
                    continue
                
                next_x = next_data[0]
                next_y = next_data[1]
                
                if next_x <= 0 or next_y <= 0:
                    continue
                
                distance = np.sqrt((next_x - hit_x)**2 + (next_y - hit_y)**2)
                
                if distance >= min_movement_threshold:
                    movement_count += 1
            
            if movement_count >= 1:
                validated_hits.append(hit_frame)
            else:
                print(f"  Frame {hit_frame}: Invalid hit - ball does not continue moving (movement_count={movement_count})")
        
        print(f"Validated {len(validated_hits)}/{len(hit_frames)} hits after continuation check")
        
        return validated_hits

    def _merge_consecutive_hits(self, hit_frames, min_frame_gap=10):
        if len(hit_frames) == 0:
            return hit_frames

        merged_hits = [hit_frames[0]]

        for i in range(1, len(hit_frames)):
            if hit_frames[i] - merged_hits[-1] >= min_frame_gap:
                merged_hits.append(hit_frames[i])

        return merged_hits

    def _detect_landing_frame(self):
        if len(self.trajectory_data) == 0:
            return None
        
        valid_frames = []
        valid_y = []
        
        for frame_idx, data in enumerate(self.trajectory_data):
            if data is not None and len(data) >= 2 and data[0] > 0 and data[1] > 0:
                valid_frames.append(frame_idx)
                valid_y.append(data[1])
        
        if len(valid_y) == 0:
            return None
        
        valid_y = np.array(valid_y)
        
        if len(valid_y) < 10:
            return None
        
        ground_y = np.percentile(valid_y, 90)
        
        for i in range(len(valid_frames) - 1, max(0, len(valid_frames) - 50), -1):
            frame_idx = valid_frames[i]
            y = valid_y[i]
            
            if y >= ground_y - 20:
                return frame_idx
        
        return None

    def save_hit_events(self, output_path):
        hit_events = []

        for frame_idx, player_idx in zip(self.hit_frames, self.hit_players):
            hit_events.append({
                'frame': frame_idx,
                'player': player_idx
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(hit_events, f, indent=2)

        print(f"Hit events saved to {output_path}")
        return hit_events

    @staticmethod
    def load_hit_events(json_path):
        with open(json_path, 'r') as f:
            hit_events = json.load(f)

        hit_frames = [event['frame'] for event in hit_events]
        hit_players = [event['player'] for event in hit_events]

        return hit_frames, hit_players
