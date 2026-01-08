import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class KalmanFilter:
    def __init__(self, initial_state: np.ndarray, process_noise: float = 1.0, measurement_noise: float = 1.0):
        self.state = initial_state.copy()
        self.covariance = np.eye(4) * 10.0
        
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state[:2]

    def update(self, measurement: np.ndarray):
        z = measurement
        y = z - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance


class PersonTracker:
    def __init__(self, max_persons: int = 2):
        self.max_persons = max_persons
        self.kalman_filters = [None] * max_persons
        self.position_history = [deque(maxlen=10) for _ in range(max_persons)]
        self.velocity_history = [deque(maxlen=5) for _ in range(max_persons)]
        self.confidence_scores = [0.0] * max_persons
        self.track_lengths = [0] * max_persons
        self.last_seen_frames = [-1] * max_persons
        self.current_frame = 0

    def get_person_center(self, keypoints: np.ndarray) -> Tuple[float, float]:
        valid_keypoints = keypoints[keypoints[:, 0] > 0]
        if len(valid_keypoints) > 0:
            return np.mean(valid_keypoints, axis=0)
        return (0, 0)

    def get_head_position(self, keypoints: np.ndarray) -> Tuple[float, float]:
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        
        head_points = []
        if nose[0] > 0 and nose[1] > 0:
            head_points.append(nose)
        if left_eye[0] > 0 and left_eye[1] > 0:
            head_points.append(left_eye)
        if right_eye[0] > 0 and right_eye[1] > 0:
            head_points.append(right_eye)
        
        if len(head_points) > 0:
            return np.mean(head_points, axis=0)
        return (0, 0)

    def get_foot_position(self, keypoints: np.ndarray) -> Tuple[float, float]:
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        foot_points = []
        if left_ankle[0] > 0 and left_ankle[1] > 0:
            foot_points.append(left_ankle)
        if right_ankle[0] > 0 and right_ankle[1] > 0:
            foot_points.append(right_ankle)
        
        if len(foot_points) > 0:
            return np.mean(foot_points, axis=0)
        return (0, 0)

    def get_shoulder_position(self, keypoints: np.ndarray) -> Tuple[float, float]:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        shoulder_points = []
        if left_shoulder[0] > 0 and left_shoulder[1] > 0:
            shoulder_points.append(left_shoulder)
        if right_shoulder[0] > 0 and right_shoulder[1] > 0:
            shoulder_points.append(right_shoulder)
        
        if len(shoulder_points) > 0:
            return np.mean(shoulder_points, axis=0)
        return (0, 0)

    def get_keypoint_confidence(self, keypoints: np.ndarray) -> float:
        valid_count = np.sum(keypoints[:, 0] > 0)
        return valid_count / 17.0

    def calculate_velocity(self, current_pos: Tuple[float, float], track_idx: int) -> Tuple[float, float]:
        if len(self.position_history[track_idx]) > 0:
            last_pos = self.position_history[track_idx][-1]
            velocity = (current_pos[0] - last_pos[0], current_pos[1] - last_pos[1])
            return velocity
        return (0, 0)

    def predict_position(self, track_idx: int) -> Optional[Tuple[float, float]]:
        if self.kalman_filters[track_idx] is not None:
            predicted = self.kalman_filters[track_idx].predict()
            return (predicted[0], predicted[1])
        return None

    def update_kalman_filter(self, track_idx: int, position: Tuple[float, float]):
        if self.kalman_filters[track_idx] is None:
            state = np.array([position[0], position[1], 0, 0])
            self.kalman_filters[track_idx] = KalmanFilter(state)
        else:
            self.kalman_filters[track_idx].update(np.array(position))

    def calculate_match_cost(self, current_keypoints: np.ndarray, track_idx: int, 
                            predicted_pos: Optional[Tuple[float, float]] = None) -> float:
        current_head = self.get_head_position(current_keypoints)
        current_foot = self.get_foot_position(current_keypoints)
        current_shoulder = self.get_shoulder_position(current_keypoints)
        current_center = self.get_person_center(current_keypoints)
        
        if len(self.position_history[track_idx]) == 0:
            return 0.0

        last_head = self.position_history[track_idx][-1].get('head', (0, 0))
        last_foot = self.position_history[track_idx][-1].get('foot', (0, 0))
        last_shoulder = self.position_history[track_idx][-1].get('shoulder', (0, 0))
        last_center = self.position_history[track_idx][-1].get('center', (0, 0))
        
        head_dist = np.linalg.norm(np.array(current_head) - np.array(last_head))
        foot_dist = np.linalg.norm(np.array(current_foot) - np.array(last_foot))
        shoulder_dist = np.linalg.norm(np.array(current_shoulder) - np.array(last_shoulder))
        center_dist = np.linalg.norm(np.array(current_center) - np.array(last_center))
        
        cost = 0.4 * head_dist + 0.3 * shoulder_dist + 0.2 * foot_dist + 0.1 * center_dist
        
        if predicted_pos is not None:
            pred_dist = np.linalg.norm(np.array(current_center) - np.array(predicted_pos))
            cost = 0.7 * cost + 0.3 * pred_dist
        
        confidence = self.get_keypoint_confidence(current_keypoints)
        cost = cost / (confidence + 0.01)
        
        return cost

    def match_persons(self, current_keypoints: List[np.ndarray]) -> List[int]:
        self.current_frame += 1
        
        if len(current_keypoints) == 0:
            for i in range(self.max_persons):
                self.last_seen_frames[i] = self.current_frame
            return []

        if self.current_frame == 1 or all(kf is None for kf in self.kalman_filters):
            for i, kp in enumerate(current_keypoints[:self.max_persons]):
                center = self.get_person_center(kp)
                self.update_kalman_filter(i, center)
                self.position_history[i].append({
                    'head': self.get_head_position(kp),
                    'foot': self.get_foot_position(kp),
                    'shoulder': self.get_shoulder_position(kp),
                    'center': center
                })
                self.confidence_scores[i] = self.get_keypoint_confidence(kp)
                self.track_lengths[i] = 1
                self.last_seen_frames[i] = self.current_frame
            return list(range(min(len(current_keypoints), self.max_persons)))

        num_current = len(current_keypoints)
        
        if num_current == 1:
            current_center = self.get_person_center(current_keypoints[0])
            current_head = self.get_head_position(current_keypoints[0])
            
            costs = []
            for i in range(self.max_persons):
                if self.kalman_filters[i] is not None:
                    predicted = self.predict_position(i)
                    cost = self.calculate_match_cost(current_keypoints[0], i, predicted)
                    
                    last_center = self.position_history[i][-1].get('center', (0, 0))
                    x_distance = abs(current_center[0] - last_center[0])
                    
                    if x_distance > 100:
                        cost *= 2.0
                    
                    costs.append((cost, i))
                else:
                    costs.append((float('inf'), i))
            
            costs.sort(key=lambda x: x[0])
            best_track = costs[0][1]
            
            self.update_kalman_filter(best_track, current_center)
            self.position_history[best_track].append({
                'head': current_head,
                'foot': self.get_foot_position(current_keypoints[0]),
                'shoulder': self.get_shoulder_position(current_keypoints[0]),
                'center': current_center
            })
            self.confidence_scores[best_track] = self.get_keypoint_confidence(current_keypoints[0])
            self.track_lengths[best_track] += 1
            self.last_seen_frames[best_track] = self.current_frame
            
            return [best_track]

        if num_current == 2:
            current_centers = [self.get_person_center(kp) for kp in current_keypoints]
            current_heads = [self.get_head_position(kp) for kp in current_keypoints]
            
            cost_matrix = np.full((2, self.max_persons), float('inf'))
            
            for i in range(2):
                for j in range(self.max_persons):
                    if self.kalman_filters[j] is not None:
                        predicted = self.predict_position(j)
                        cost = self.calculate_match_cost(current_keypoints[i], j, predicted)
                        
                        last_center = self.position_history[j][-1].get('center', (0, 0))
                        x_distance = abs(current_centers[i][0] - last_center[0])
                        
                        if x_distance > 100:
                            cost *= 2.0
                        
                        cost_matrix[i, j] = cost
            
            if self.max_persons == 2:
                cost00 = cost_matrix[0, 0]
                cost01 = cost_matrix[0, 1]
                cost10 = cost_matrix[1, 0]
                cost11 = cost_matrix[1, 1]
                
                if cost00 + cost11 < cost01 + cost10:
                    assignment = [0, 1]
                else:
                    assignment = [1, 0]
            else:
                assignment = [0, 1]
            
            for i, track_idx in enumerate(assignment):
                if track_idx < self.max_persons:
                    self.update_kalman_filter(track_idx, current_centers[i])
                    self.position_history[track_idx].append({
                        'head': current_heads[i],
                        'foot': self.get_foot_position(current_keypoints[i]),
                        'shoulder': self.get_shoulder_position(current_keypoints[i]),
                        'center': current_centers[i]
                    })
                    self.confidence_scores[track_idx] = self.get_keypoint_confidence(current_keypoints[i])
                    self.track_lengths[track_idx] += 1
                    self.last_seen_frames[track_idx] = self.current_frame
            
            return assignment

        return list(range(min(num_current, self.max_persons)))


def track_poses(poses_list: List[List[np.ndarray]], max_persons: int = 2) -> np.ndarray:
    tracker = PersonTracker(max_persons=max_persons)
    total_frames = len(poses_list)
    poses_array = np.zeros((total_frames, max_persons, 17, 2), dtype=np.float32)

    for frame_idx, frame_poses in enumerate(poses_list):
        if len(frame_poses) == 0:
            continue

        assignment = tracker.match_persons(frame_poses)

        for person_idx, original_idx in enumerate(assignment):
            if person_idx < max_persons and original_idx < len(frame_poses):
                poses_array[frame_idx, person_idx] = frame_poses[original_idx][:17, :2]

    return poses_array
