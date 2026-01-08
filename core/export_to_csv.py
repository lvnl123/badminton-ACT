import json
import numpy as np
import pandas as pd
from pathlib import Path


def export_to_csv(
    hit_events_path: str,
    poses_path: str,
    ball_json_path: str,
    ball_denoise_json_path: str,
    output_csv_path: str,
    fps: float = 25.0,
    stroke_types_path: str = None
):
    hit_events = []
    with open(hit_events_path, 'r') as f:
        hit_events = json.load(f)
    
    hit_frames_set = set(event['frame'] for event in hit_events)
    hit_frame_to_player = {event['frame']: event['player'] for event in hit_events}
    
    stroke_types = {}
    if stroke_types_path and Path(stroke_types_path).exists():
        with open(stroke_types_path, 'r') as f:
            stroke_data = json.load(f)
            stroke_types = {event['frame']: event for event in stroke_data}
    
    with open(ball_json_path, 'r') as f:
        ball_data = json.load(f)
    
    with open(ball_denoise_json_path, 'r') as f:
        ball_denoise_data = json.load(f)
    
    poses = np.load(poses_path)
    total_frames = poses.shape[0]
    
    rows = []
    cumulative_hit_count = 0
    
    for frame_idx in range(total_frames):
        frame_str = str(frame_idx)
        
        ball_info = ball_data.get(frame_str, {'visible': 0, 'x': 0, 'y': 0})
        ball_denoise_info = ball_denoise_data.get(frame_str, {'visible': 0, 'x': 0, 'y': 0})
        
        ball_x = ball_info.get('x', 0)
        ball_y = ball_info.get('y', 0)
        ball_visible = ball_info.get('visible', 0)
        
        ball_denoise_x = ball_denoise_info.get('x', 0)
        ball_denoise_y = ball_denoise_info.get('y', 0)
        ball_denoise_visible = ball_denoise_info.get('visible', 0)
        
        is_hit = frame_idx in hit_frames_set
        hit_player = hit_frame_to_player.get(frame_idx, 0)
        
        stroke_type_id = -1
        stroke_type_name = ''
        stroke_type_name_en = ''
        if frame_idx in stroke_types:
            stroke_info = stroke_types[frame_idx]
            stroke_type_id = stroke_info.get('stroke_type_id', -1)
            stroke_type_name = stroke_info.get('stroke_type_name', '')
            stroke_type_name_en = stroke_info.get('stroke_type_name_en', '')
        
        if is_hit:
            cumulative_hit_count += 1
        
        ball_speed = 0
        if frame_idx > 0:
            prev_frame_str = str(frame_idx - 1)
            prev_ball = ball_denoise_data.get(prev_frame_str, {'visible': 0, 'x': 0, 'y': 0})
            if prev_ball['visible'] == 1 and ball_denoise_visible == 1:
                dx = ball_denoise_x - prev_ball['x']
                dy = ball_denoise_y - prev_ball['y']
                distance = np.sqrt(dx**2 + dy**2)
                ball_speed = distance * fps
        
        row = {
            'frame': frame_idx,
            'time_seconds': frame_idx / fps,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'ball_visible': ball_visible,
            'ball_denoise_x': ball_denoise_x,
            'ball_denoise_y': ball_denoise_y,
            'ball_denoise_visible': ball_denoise_visible,
            'ball_speed': ball_speed,
            'is_hit': 1 if is_hit else 0,
            'hit_player': hit_player,
            'cumulative_hit_count': cumulative_hit_count,
            'stroke_type_id': stroke_type_id,
            'stroke_type_name': stroke_type_name,
            'stroke_type_name_en': stroke_type_name_en
        }
        
        for player_idx in range(2):
            for joint_idx in range(17):
                joint = poses[frame_idx, player_idx, joint_idx]
                row[f'player{player_idx+1}_joint{joint_idx}_x'] = joint[0]
                row[f'player{player_idx+1}_joint{joint_idx}_y'] = joint[1]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file exported to: {output_csv_path}")
    print(f"Total frames: {len(df)}")
    print(f"Total hits: {cumulative_hit_count}")
    print(f"Columns: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export video analysis data to CSV')
    parser.add_argument('--hit_events', type=str, required=True, help='Hit events JSON path')
    parser.add_argument('--poses', type=str, required=True, help='Poses numpy array path')
    parser.add_argument('--ball_json', type=str, required=True, help='Ball detection JSON path')
    parser.add_argument('--ball_denoise_json', type=str, required=True, help='Ball denoised JSON path')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV path')
    parser.add_argument('--fps', type=float, default=25.0, help='Video FPS')
    
    args = parser.parse_args()
    
    export_to_csv(
        args.hit_events,
        args.poses,
        args.ball_json,
        args.ball_denoise_json,
        args.output_csv,
        args.fps
    )
