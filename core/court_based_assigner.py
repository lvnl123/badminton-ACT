import numpy as np
from typing import List, Tuple, Optional


class CourtBasedPlayerAssigner:
    def __init__(self, frame_height: int, frame_width: int, court_info: Optional[List[float]] = None, 
                 extended_court_points: Optional[np.ndarray] = None):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.court_info = court_info
        self.extended_court_points = extended_court_points
        
        if court_info is not None and len(court_info) >= 5:
            self.net_y = court_info[4]
        else:
            self.net_y = frame_height / 2

    def set_court_info(self, court_info: List[float], extended_court_points: Optional[np.ndarray] = None):
        self.court_info = court_info
        if len(court_info) >= 5:
            self.net_y = court_info[4]
        if extended_court_points is not None:
            self.extended_court_points = extended_court_points

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

    def get_person_center(self, keypoints: np.ndarray) -> Tuple[float, float]:
        valid_keypoints = keypoints[keypoints[:, 0] > 0]
        if len(valid_keypoints) > 0:
            return np.mean(valid_keypoints, axis=0)
        return (0, 0)

    def is_in_top_half(self, keypoints: np.ndarray) -> bool:
        foot_pos = self.get_foot_position(keypoints)
        if foot_pos[0] == 0 and foot_pos[1] == 0:
            center_pos = self.get_person_center(keypoints)
            if center_pos[0] == 0 and center_pos[1] == 0:
                return False
            return center_pos[1] < self.net_y
        return foot_pos[1] < self.net_y

    def is_in_bottom_half(self, keypoints: np.ndarray) -> bool:
        foot_pos = self.get_foot_position(keypoints)
        if foot_pos[0] == 0 and foot_pos[1] == 0:
            center_pos = self.get_person_center(keypoints)
            if center_pos[0] == 0 and center_pos[1] == 0:
                return False
            return center_pos[1] > self.net_y
        return foot_pos[1] > self.net_y

    def is_in_court(self, keypoints: np.ndarray) -> bool:
        if self.court_info is None or self.extended_court_points is None:
            return True

        l_a = self.court_info[0]
        l_b = self.court_info[1]
        r_a = self.court_info[2]
        r_b = self.court_info[3]

        ankle_x = (keypoints[15][0] + keypoints[16][0]) / 2
        ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2

        top = ankle_y > self.extended_court_points[0][1]
        bottom = ankle_y < self.extended_court_points[5][1]

        lmp_x = (ankle_y - l_b) / l_a
        rmp_x = (ankle_y - r_b) / r_a
        left = ankle_x > lmp_x
        right = ankle_x < rmp_x

        if left and right and top and bottom:
            return True
        else:
            return False

    def assign_players(self, keypoints_list: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if len(keypoints_list) < 2:
            return None, None

        valid_indices = []
        for i, keypoints in enumerate(keypoints_list):
            if self.is_in_court(keypoints):
                valid_indices.append(i)

        if len(valid_indices) < 2:
            return None, None

        valid_keypoints = [keypoints_list[i] for i in valid_indices]

        top_player = None
        bottom_player = None

        for keypoints in valid_keypoints:
            if self.is_in_top_half(keypoints):
                if top_player is None:
                    top_player = keypoints
            elif self.is_in_bottom_half(keypoints):
                if bottom_player is None:
                    bottom_player = keypoints

        if top_player is None and bottom_player is None:
            centers = [self.get_person_center(kp) for kp in valid_keypoints]
            centers = [(i, c) for i, c in enumerate(centers) if c[0] > 0 or c[1] > 0]
            centers.sort(key=lambda x: x[1][1])
            
            if len(centers) >= 2:
                top_player = valid_keypoints[centers[0][0]]
                bottom_player = valid_keypoints[centers[1][0]]

        return top_player, bottom_player

    def assign_players_with_indices(self, keypoints_list: List[np.ndarray]) -> Tuple[Optional[int], Optional[int]]:
        if len(keypoints_list) < 2:
            return None, None

        top_idx = None
        bottom_idx = None

        for i, keypoints in enumerate(keypoints_list):
            if self.is_in_top_half(keypoints):
                if top_idx is None:
                    top_idx = i
            elif self.is_in_bottom_half(keypoints):
                if bottom_idx is None:
                    bottom_idx = i

        if top_idx is None and bottom_idx is None:
            centers = [(i, self.get_person_center(kp)) for i, kp in enumerate(keypoints_list)]
            centers = [(i, c) for i, c in centers if c[0] > 0 or c[1] > 0]
            centers.sort(key=lambda x: x[1][1])
            
            if len(centers) >= 2:
                top_idx = centers[0][0]
                bottom_idx = centers[1][0]

        return top_idx, bottom_idx


def assign_players_court_based(poses_list: List[List[np.ndarray]], 
                                frame_height: int, 
                                frame_width: int) -> np.ndarray:
    assigner = CourtBasedPlayerAssigner(frame_height, frame_width)
    total_frames = len(poses_list)
    poses_array = np.zeros((total_frames, 2, 17, 2), dtype=np.float32)

    for frame_idx, frame_poses in enumerate(poses_list):
        if len(frame_poses) < 2:
            continue

        top_player, bottom_player = assigner.assign_players(frame_poses)

        if top_player is not None:
            poses_array[frame_idx, 0] = top_player[:17, :2]
        if bottom_player is not None:
            poses_array[frame_idx, 1] = bottom_player[:17, :2]

    return poses_array
