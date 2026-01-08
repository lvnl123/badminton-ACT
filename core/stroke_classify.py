import torch
from torch import Tensor, nn
import numpy as np
from pathlib import Path
import json

from .bst import BST, BST_CG, BST_AP, BST_CG_AP


def get_bone_pairs(skeleton_format='coco'):
    if skeleton_format == 'coco':
        pairs = [
            (0,1),(0,2),(1,2),(1,3),(2,4),   # head
            (3,5),(4,6),                     # ears to shoulders
            (5,7),(7,9),(6,8),(8,10),        # arms
            (5,6),(5,11),(6,12),(11,12),     # torso
            (11,13),(13,15),(12,14),(14,16)  # legs
        ]
    else:
        raise NotImplementedError
    return pairs


def create_bones(joints: np.ndarray, pairs) -> np.ndarray:
    bones = []
    for start, end in pairs:
        start_j = joints[:, :, start, :]
        end_j = joints[:, :, end, :]
        bone = np.where((start_j != 0.0) & (end_j != 0.0), end_j - start_j, 0.0)
        bones.append(bone)
    return np.stack(bones, axis=-2)


class StrokeClassifier:
    def __init__(self, model_path, model_type='BST_CG_AP', seq_len=100, n_classes=35, n_joints=17):
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.n_joints = n_joints
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.model_type = model_type

        self.net = self._load_model(model_path, model_type, seq_len, n_classes)
        self.net.eval()

    def _load_model(self, model_path, model_type, seq_len, n_classes):
        n_joints = 17
        n_bones = 19
        extra = 1
        in_channels = 2
        in_dim = (n_joints + n_bones * extra) * in_channels

        if model_type == 'BST':
            net = BST(
                in_dim=in_dim,
                n_class=n_classes,
                seq_len=seq_len,
                depth_tem=2,
                depth_inter=1
            )
        elif model_type == 'BST_CG':
            net = BST_CG(
                in_dim=in_dim,
                n_class=n_classes,
                seq_len=seq_len,
                depth_tem=2,
                depth_inter=1
            )
        elif model_type == 'BST_AP':
            net = BST_AP(
                in_dim=in_dim,
                n_class=n_classes,
                seq_len=seq_len,
                depth_tem=2,
                depth_inter=1
            )
        elif model_type == 'BST_CG_AP':
            net = BST_CG_AP(
                in_dim=in_dim,
                n_class=n_classes,
                seq_len=seq_len,
                depth_tem=2,
                depth_inter=1
            )
        else:
            raise NotImplementedError(f"Model type {model_type} not supported")

        net.load_state_dict(torch.load(str(model_path), map_location=self.device, weights_only=True))
        return net.to(self.device)

    def prepare_hit_segment(self, trajectory_data, poses, hit_frame, seq_len=100):
        if hit_frame < seq_len // 2:
            start_frame = 0
            end_frame = min(seq_len, len(trajectory_data))
        else:
            start_frame = hit_frame - seq_len // 2
            end_frame = min(hit_frame + seq_len // 2, len(trajectory_data))

        segment_length = end_frame - start_frame
        if segment_length < seq_len:
            pad_before = (seq_len - segment_length) // 2
            pad_after = seq_len - segment_length - pad_before
        else:
            pad_before = 0
            pad_after = 0

        n_joints = 17
        human_pose = np.zeros((seq_len, 2, n_joints, 2))
        shuttle = np.zeros((seq_len, 2))
        pos = np.zeros((seq_len, 2, 2))

        for i in range(segment_length):
            frame_idx = start_frame + i
            output_idx = pad_before + i

            if frame_idx < len(trajectory_data):
                traj = trajectory_data[frame_idx]
                if traj is not None and len(traj) >= 2:
                    shuttle[output_idx] = [traj[0], traj[1]]

            if poses is not None and frame_idx < len(poses):
                for player_idx in range(min(2, poses.shape[1])):
                    pose_data = poses[frame_idx, player_idx]
                    if pose_data is not None:
                        for joint_idx in range(min(n_joints, pose_data.shape[0])):
                            x, y = pose_data[joint_idx, 0], pose_data[joint_idx, 1]
                            if x > 0 and y > 0:
                                human_pose[output_idx, player_idx, joint_idx] = [x, y]
                                pos[output_idx, player_idx] = [x, y]

        pairs = get_bone_pairs('coco')
        bones = create_bones(human_pose, pairs)
        
        mid_joints = []
        for start, end in pairs:
            start_j = human_pose[:, :, start, :]
            end_j = human_pose[:, :, end, :]
            mid_j = np.where((start_j != 0.0) & (end_j != 0.0), (start_j + end_j) / 2, 0.0)
            mid_joints.append(mid_j)
        bones_center = np.stack(mid_joints, axis=-2)
        
        human_pose = np.concatenate((human_pose, bones_center), axis=-2)

        return human_pose, shuttle, pos

    def classify_hit(self, trajectory_data, poses, hit_frame):
        human_pose, shuttle, pos = self.prepare_hit_segment(
            trajectory_data, poses, hit_frame, self.seq_len
        )

        human_pose_tensor = torch.from_numpy(human_pose).float().unsqueeze(0).to(self.device)
        shuttle_tensor = torch.from_numpy(shuttle).float().unsqueeze(0).to(self.device)
        pos_tensor = torch.from_numpy(pos).float().unsqueeze(0).to(self.device)
        video_len_tensor = torch.tensor([self.seq_len]).to(self.device)

        with torch.no_grad():
            b, t, n, j, d = human_pose_tensor.shape
            human_pose_tensor = human_pose_tensor.reshape(b, t, n, -1)
            logits = self.net(human_pose_tensor, shuttle_tensor, pos_tensor, video_len_tensor)
            pred = torch.argmax(logits, dim=1).cpu().item()

        return pred

    def classify_hits(self, trajectory_data, poses, hit_frames):
        stroke_types = []

        for hit_frame in hit_frames:
            stroke_type = self.classify_hit(trajectory_data, poses, hit_frame)
            stroke_types.append(stroke_type)

        return stroke_types

    def get_stroke_type_name(self, class_id, dataset='shuttleset'):
        stroke_types = self._get_stroke_types(dataset)
        if 0 <= class_id < len(stroke_types):
            return stroke_types[class_id]
        return f"Unknown_{class_id}"

    def get_stroke_type_name_en(self, class_id, dataset='shuttleset'):
        stroke_types_en = self._get_stroke_types_en(dataset)
        if 0 <= class_id < len(stroke_types_en):
            return stroke_types_en[class_id]
        return f"Unknown_{class_id}"

    def _get_stroke_types(self, dataset='shuttleset'):
        if dataset == 'shuttleset':
            return [
                '正手高远球', '反手高远球', '正手吊球', '反手吊球',
                '正手杀球', '反手杀球', '正手平抽', '反手平抽',
                '正手网前球', '反手网前球', '正手挑球', '反手挑球',
                '正手推球', '反手推球', '正手扑球', '反手扑球',
                '正手切球', '反手切球', '正手旋转球', '反手旋转球',
                '正手短发球', '正手长发球', '反手短发球', '反手长发球',
                '正手防守', '反手防守', '正手斜线球', '反手斜线球',
                '正手直线球', '反手直线球', '正手挑高球', '反手挑高球',
                '正手半杀球', '反手半杀球', '正手重杀', '反手重杀'
            ]
        elif dataset == 'badDB' or dataset == 'tenniSet':
            return [
                '正手高远球', '反手高远球', '正手吊球', '反手吊球',
                '正手杀球', '反手杀球'
            ]
        else:
            return [f"Class_{i}" for i in range(self.n_classes)]

    def _get_stroke_types_en(self, dataset='shuttleset'):
        if dataset == 'shuttleset':
            return [
                'forehand_clear', 'backhand_clear', 'forehand_drop', 'backhand_drop',
                'forehand_smash', 'backhand_smash', 'forehand_drive', 'backhand_drive',
                'forehand_net_shot', 'backhand_net_shot', 'forehand_lift', 'backhand_lift',
                'forehand_push', 'backhand_push', 'forehand_flick', 'backhand_flick',
                'forehand_slice', 'backhand_slice', 'forehand_spin', 'backhand_spin',
                'serve_forehand_short', 'serve_forehand_long', 'serve_backhand_short', 'serve_backhand_long',
                'forehand_defensive', 'backhand_defensive', 'forehand_cross_court', 'backhand_cross_court',
                'forehand_straight', 'backhand_straight', 'forehand_lob', 'backhand_lob',
                'forehand_half_smash', 'backhand_half_smash', 'forehand_kill', 'backhand_kill'
            ]
        elif dataset == 'badDB' or dataset == 'tenniSet':
            return [
                'forehand_clear', 'backhand_clear', 'forehand_drop', 'backhand_drop',
                'forehand_smash', 'backhand_smash'
            ]
        else:
            return [f"Class_{i}" for i in range(self.n_classes)]

    def save_stroke_results(self, hit_frames, hit_players, stroke_types, output_path):
        stroke_results = []

        for frame, player, stroke_type in zip(hit_frames, hit_players, stroke_types):
            stroke_results.append({
                'frame': frame,
                'player': player,
                'stroke_type_id': int(stroke_type),
                'stroke_type_name': self.get_stroke_type_name(int(stroke_type)),
                'stroke_type_name_en': self.get_stroke_type_name_en(int(stroke_type))
            })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stroke_results, f, indent=2)

        print(f"Stroke classification results saved to {output_path}")
        return stroke_results

    @staticmethod
    def load_stroke_results(json_path):
        with open(json_path, 'r') as f:
            stroke_results = json.load(f)

        hit_frames = [result['frame'] for result in stroke_results]
        hit_players = [result['player'] for result in stroke_results]
        stroke_types = [result['stroke_type_id'] for result in stroke_results]

        return hit_frames, hit_players, stroke_types


def create_classifier(dataset='shuttleset', seq_len=100):
    models_dir = Path(__file__).parent.parent / 'models' / 'bst'

    if dataset == 'shuttleset':
        model_dir = models_dir / 'shuttleset_35classes'
        model_files = list(model_dir.glob('*.pt'))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")

        model_path = model_files[0]
        n_classes = 35
        model_type = 'BST_CG_AP'
    elif dataset == 'badDB':
        model_dir = models_dir / 'badDB_6classes'
        model_files = list(model_dir.glob('*.pt'))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")

        model_path = model_files[0]
        n_classes = 6
        model_type = 'BST'
    elif dataset == 'tenniSet':
        model_dir = models_dir / 'tenniSet_6classes'
        model_files = list(model_dir.glob('*.pt'))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")

        model_path = model_files[0]
        n_classes = 6
        model_type = 'BST'
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    classifier = StrokeClassifier(
        model_path=model_path,
        model_type=model_type,
        seq_len=seq_len,
        n_classes=n_classes
    )

    return classifier
