import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.spatial import ConvexHull
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Rally:
    id: int
    start_frame: int
    end_frame: int
    duration_sec: float
    hit_count: int
    strokes: List[dict]
    # Trajectory data slice for this rally
    trajectory: pd.DataFrame 
    # Player stats in this rally: {player_id: {'dist': float, 'avg_speed': float, 'max_speed': float}}
    player_stats: Dict[int, dict] 

class MatchEngine:
    def __init__(self, match_dir: str):
        self.match_dir = Path(match_dir)
        self.match_name = self.match_dir.name
        self.df = None
        self.hits = []
        self.strokes = []
        self.rallies: List[Rally] = []
        # Global stats
        self.global_player_stats = {
            1: {'dist': 0.0, 'speed_dist': [], 'coverage': 0.0, 'type_counts': {}},
            2: {'dist': 0.0, 'speed_dist': [], 'coverage': 0.0, 'type_counts': {}}
        }
        self.poses = None # npy array

    def load_data(self):
        """Load all data files from the directory."""
        try:
            # 1. Load CSV
            csv_path = self.match_dir / f"{self.match_name}_data.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            self.df = pd.read_csv(csv_path)
            
            # Ensure frame is index for faster lookup
            # self.df.set_index('frame', drop=False, inplace=True)

            # 2. Load Hits
            hits_path = self.match_dir / f"{self.match_name}_hit_events.json"
            if hits_path.exists():
                with open(hits_path, 'r', encoding='utf-8') as f:
                    self.hits = json.load(f)
            
            # 3. Load Strokes
            strokes_path = self.match_dir / f"{self.match_name}_stroke_types.json"
            if strokes_path.exists():
                with open(strokes_path, 'r', encoding='utf-8') as f:
                    self.strokes = json.load(f)

            # 4. Load Poses (optional)
            poses_path = self.match_dir / f"{self.match_name}_poses.npy"
            if poses_path.exists():
                self.poses = np.load(poses_path)

            logger.info(f"Loaded match data: {len(self.df)} frames, {len(self.hits)} hits")
            
            self._process_data()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _process_data(self):
        """Process raw data into rallies and advanced stats."""
        self._calculate_velocities()
        self._segment_rallies()
        self._calculate_global_stats()

    def _calculate_velocities(self):
        """Calculate ball and player velocities if not present."""
        # Calculate player speed (px/frame -> px/sec assuming 30fps if not specified, 
        # but better use time_seconds diff)
        
        dt = self.df['time_seconds'].diff().fillna(0.04) # Default to 0.04s (25fps) if diff is 0
        dt[dt == 0] = 0.04
        
        # Player 1 (Bottom?) - usually player1_joint0_x/y is root (hips)
        p1_dx = self.df['player1_joint0_x'].diff().fillna(0)
        p1_dy = self.df['player1_joint0_y'].diff().fillna(0)
        self.df['p1_speed'] = np.sqrt(p1_dx**2 + p1_dy**2) / dt
        
        # Player 2 (Top?)
        p2_dx = self.df['player2_joint0_x'].diff().fillna(0)
        p2_dy = self.df['player2_joint0_y'].diff().fillna(0)
        self.df['p2_speed'] = np.sqrt(p2_dx**2 + p2_dy**2) / dt
        
        # Smooth speeds
        self.df['p1_speed'] = self.df['p1_speed'].rolling(window=5, min_periods=1).mean()
        self.df['p2_speed'] = self.df['p2_speed'].rolling(window=5, min_periods=1).mean()

    def _segment_rallies(self):
        """Segment the match into rallies based on hit events and time gaps."""
        if not self.hits:
            # Fallback: treat whole match as one rally if no hits
            return

        # Sort hits by frame
        sorted_hits = sorted(self.hits, key=lambda x: x['frame'])
        
        current_rally_hits = []
        rally_id = 1
        
        # Threshold for new rally: > 4 seconds gap between hits
        NEW_RALLY_THRESHOLD_FRAMES = 30 * 4 
        
        last_hit_frame = -999
        
        for hit in sorted_hits:
            frame = hit['frame']
            
            # Check if this hit starts a new rally
            if frame - last_hit_frame > NEW_RALLY_THRESHOLD_FRAMES and current_rally_hits:
                # Finish previous rally
                self._create_rally(rally_id, current_rally_hits)
                rally_id += 1
                current_rally_hits = []
            
            # Add stroke info to hit if available
            stroke_info = next((s for s in self.strokes if s['frame'] == frame), None)
            if stroke_info:
                hit.update(stroke_info)
                
            current_rally_hits.append(hit)
            last_hit_frame = frame
            
        # Add last rally
        if current_rally_hits:
            self._create_rally(rally_id, current_rally_hits)

    def _create_rally(self, rally_id: int, hits: List[dict]):
        """Create a Rally object from a list of hits."""
        start_frame = max(0, hits[0]['frame'] - 30) # Start 1 sec before first hit
        end_frame = min(len(self.df)-1, hits[-1]['frame'] + 60) # End 2 sec after last hit
        
        rally_df = self.df.iloc[start_frame:end_frame+1].copy()
        duration = rally_df['time_seconds'].max() - rally_df['time_seconds'].min()
        
        # Calculate stats for this rally
        p1_dist = (rally_df['p1_speed'] * rally_df['time_seconds'].diff().fillna(0)).sum()
        p2_dist = (rally_df['p2_speed'] * rally_df['time_seconds'].diff().fillna(0)).sum()
        
        r = Rally(
            id=rally_id,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_sec=duration,
            hit_count=len(hits),
            strokes=hits,
            trajectory=rally_df,
            player_stats={
                1: {'dist': p1_dist, 'avg_speed': rally_df['p1_speed'].mean(), 'max_speed': rally_df['p1_speed'].max()},
                2: {'dist': p2_dist, 'avg_speed': rally_df['p2_speed'].mean(), 'max_speed': rally_df['p2_speed'].max()}
            }
        )
        self.rallies.append(r)

    def _calculate_global_stats(self):
        """Calculate aggregate stats for the whole match."""
        # Distance
        # We sum distances from all rallies to avoid counting dead time walking
        self.global_player_stats[1]['dist'] = sum(r.player_stats[1]['dist'] for r in self.rallies)
        self.global_player_stats[2]['dist'] = sum(r.player_stats[2]['dist'] for r in self.rallies)
        
        # Speed Distribution (sample from all rally frames)
        all_p1_speeds = []
        all_p2_speeds = []
        for r in self.rallies:
            all_p1_speeds.extend(r.trajectory['p1_speed'].dropna().tolist())
            all_p2_speeds.extend(r.trajectory['p2_speed'].dropna().tolist())
            
        self.global_player_stats[1]['speed_dist'] = all_p1_speeds
        self.global_player_stats[2]['speed_dist'] = all_p2_speeds
        
        # Stroke Counts
        for r in self.rallies:
            for s in r.strokes:
                p = s.get('player', 0)
                st_name = s.get('stroke_type_name', 'Unknown')
                if p in [1, 2]:
                    self.global_player_stats[p]['type_counts'][st_name] = \
                        self.global_player_stats[p]['type_counts'].get(st_name, 0) + 1

        # Court Coverage (Convex Hull Area)
        # Filter valid positions (not 0,0)
        for p in [1, 2]:
            x_col = f'player{p}_joint0_x'
            y_col = f'player{p}_joint0_y'
            points = self.df[[x_col, y_col]].values
            # Filter out (0,0) or NaN
            mask = (points[:,0] > 10) & (points[:,1] > 10) & ~np.isnan(points[:,0])
            valid_points = points[mask]
            
            if len(valid_points) > 3:
                try:
                    hull = ConvexHull(valid_points)
                    self.global_player_stats[p]['coverage'] = hull.area
                except:
                    self.global_player_stats[p]['coverage'] = 0.0

    def get_transition_matrix(self, player_id: int) -> pd.DataFrame:
        """
        Calculate the stroke type transition matrix for a player.
        Rows: Previous Stroke Type (by Opponent or Self?)
        Let's do: My Previous Stroke -> My Current Stroke (Chain) OR Opponent Stroke -> My Response
        Professional analysis usually looks at: Opponent Shot -> My Response (Tactical Response)
        """
        transitions = {}
        
        for r in self.rallies:
            strokes = r.strokes
            for i in range(1, len(strokes)):
                curr = strokes[i]
                prev = strokes[i-1]
                
                if curr.get('player') == player_id and prev.get('player') != player_id:
                    # Opponent shot -> My response
                    prev_type = prev.get('stroke_type_name', 'Unknown')
                    curr_type = curr.get('stroke_type_name', 'Unknown')
                    
                    if prev_type not in transitions: transitions[prev_type] = {}
                    transitions[prev_type][curr_type] = transitions[prev_type].get(curr_type, 0) + 1

        # Convert to DataFrame
        if not transitions:
            return pd.DataFrame()
            
        df = pd.DataFrame(transitions).fillna(0).T # Rows: Opponent Shot, Cols: My Response
        # Normalize by row (probability)
        df_norm = df.div(df.sum(axis=1), axis=0)
        return df_norm

    def get_sankey_data(self) -> dict:
        """
        Generate data for Sankey diagram: Serve -> ... -> End Reason
        Simplified to 3 steps: Service -> 3rd Shot -> Outcome
        """
        flows = [] # (source, target, value)
        
        for r in self.rallies:
            if len(r.strokes) < 1: continue
            
            # 1. Service
            first_stroke = r.strokes[0]
            service_type = first_stroke.get('stroke_type_name', 'Serve')
            
            # 2. Outcome (Last Shot)
            last_stroke = r.strokes[-1]
            last_type = last_stroke.get('stroke_type_name', 'End')
            winner = "Rally End" # We don't have score info in json usually, unless inferred
            
            # Simple flow: Service -> Last Shot Type
            flows.append((service_type, last_type))
            
        # Aggregate
        from collections import Counter
        counts = Counter(flows)
        
        return {
            "sources": [k[0] for k in counts.keys()],
            "targets": [k[1] for k in counts.keys()],
            "values": list(counts.values())
        }

    def get_player_radar_data(self, player_id: int) -> Dict[str, float]:
        stats = self.global_player_stats[player_id]
        counts = stats['type_counts']
        total_shots = sum(counts.values()) if counts else 1
        
        # 1. Attack: Smashes / Drives
        attack_keywords = ['杀', 'smash', 'drive', '抽']
        attack_count = sum(v for k, v in counts.items() if any(x in k.lower() for x in attack_keywords))
        attack_score = min(100, (attack_count / total_shots) * 300) # Heuristic
        
        # 2. Defense: Lifts / Clears (assuming defensive)
        def_keywords = ['挑', 'lift', 'clear', '高远']
        def_count = sum(v for k, v in counts.items() if any(x in k.lower() for x in def_keywords))
        def_score = min(100, (def_count / total_shots) * 300)
        
        # 3. Speed: Avg speed in rallies
        avg_speed = np.mean(stats['speed_dist']) if stats['speed_dist'] else 0
        speed_score = min(100, avg_speed * 0.5) # px/frame factor
        
        # 4. Stamina: Total Distance / Rallies
        dist_score = min(100, stats['dist'] / 1000) # Normalize
        
        # 5. Control: Net shots / Drops
        ctrl_keywords = ['网', 'net', 'drop', '吊', '放']
        ctrl_count = sum(v for k, v in counts.items() if any(x in k.lower() for x in ctrl_keywords))
        ctrl_score = min(100, (ctrl_count / total_shots) * 400)
        
        diversity_types = len(counts.keys())
        diversity_score = min(100, diversity_types * 12.5)
        return {
            "进攻": float(attack_score),
            "防守": float(def_score),
            "速度": float(speed_score),
            "体能": float(dist_score),
            "控制": float(ctrl_score),
            "多样性": float(diversity_score)
        }

    def get_speed_series(self, player_id: int) -> pd.DataFrame:
        col = 'p1_speed' if player_id == 1 else 'p2_speed'
        return self.df[['time_seconds', col]].rename(columns={col: 'speed'}).dropna()

    def get_accel_series(self, player_id: int) -> pd.DataFrame:
        s = self.get_speed_series(player_id).copy()
        dt = s['time_seconds'].diff().fillna(0.04)
        dv = s['speed'].diff().fillna(0.0)
        accel = (dv / dt).rolling(window=3, min_periods=1).mean()
        s['accel'] = accel
        return s

    def get_player_zone_ratios(self, player_id: int) -> Dict[str, float]:
        x_col = f'player{player_id}_joint0_x'
        y_col = f'player{player_id}_joint0_y'
        d = self.df[[x_col, y_col, 'time_seconds']].dropna()
        if d.empty:
            return {'front': 0.0, 'back': 0.0, 'left': 0.0, 'right': 0.0, 'net_aggr': 0.0}
        W = float(self.df['ball_x'].max() if 'ball_x' in self.df else d[x_col].max())
        H = float(self.df['ball_y'].max() if 'ball_y' in self.df else d[y_col].max())
        dt = d['time_seconds'].diff().fillna(0.04)
        left_mask = d[x_col] < W * 0.5
        right_mask = ~left_mask
        front_mask = d[y_col] < H * 0.5
        back_mask = ~front_mask
        total = dt.sum()
        left = dt[left_mask].sum() / total if total > 0 else 0.0
        right = dt[right_mask].sum() / total if total > 0 else 0.0
        front = dt[front_mask].sum() / total if total > 0 else 0.0
        back = dt[back_mask].sum() / total if total > 0 else 0.0
        speed_col = 'p1_speed' if player_id == 1 else 'p2_speed'
        net_speed = self.df.loc[front_mask, speed_col].dropna().mean() if front_mask.any() else 0.0
        net_aggr = front * (net_speed if np.isfinite(net_speed) else 0.0)
        return {'front': float(front), 'back': float(back), 'left': float(left), 'right': float(right), 'net_aggr': float(net_aggr)}

    def get_barycenter_cov(self, player_id: int) -> Dict[str, float]:
        x_col = f'player{player_id}_joint0_x'
        y_col = f'player{player_id}_joint0_y'
        pts = self.df[[x_col, y_col]].dropna().values
        if len(pts) < 5:
            return {'cx': 0.0, 'cy': 0.0, 'var_x': 0.0, 'var_y': 0.0, 'cov_xy': 0.0}
        cx = float(np.mean(pts[:,0]))
        cy = float(np.mean(pts[:,1]))
        cov = np.cov(pts.T)
        return {'cx': cx, 'cy': cy, 'var_x': float(cov[0,0]), 'var_y': float(cov[1,1]), 'cov_xy': float(cov[0,1])}

    def get_physical_kpis(self, player_id: int) -> Dict[str, float]:
        s = self.get_speed_series(player_id)
        if s.empty:
            return {'avg_speed': 0.0, 'p95_speed': 0.0, 'max_speed': 0.0, 'accel_peak': 0.0, 'accel_mean': 0.0}
        avg_speed = float(s['speed'].mean())
        p95_speed = float(np.quantile(s['speed'], 0.95))
        max_speed = float(s['speed'].max())
        a = self.get_accel_series(player_id)
        accel_peak = float(np.nanmax(np.abs(a['accel']))) if not a.empty else 0.0
        accel_mean = float(np.nanmean(np.abs(a['accel']))) if not a.empty else 0.0
        return {'avg_speed': avg_speed, 'p95_speed': p95_speed, 'max_speed': max_speed, 'accel_peak': accel_peak, 'accel_mean': accel_mean}

    def get_chord_data(self, player_id: Optional[int] = None, min_count: int = 2) -> Dict[str, List[dict]]:
        nodes = []
        links = []
        label_norm = {
            '杀': '杀球', 'smash': '杀球',
            '抽': '抽球', 'drive': '抽球',
            '吊': '吊球', 'drop': '吊球',
            '网': '网前', 'net': '网前',
            '挑': '挑球', 'lift': '挑球',
            '高': '高远', 'clear': '高远'
        }
        def normalize(t: str) -> str:
            xl = (t or '').lower()
            for k, v in label_norm.items():
                if k in xl:
                    return v
            return t or '未知'
        from collections import Counter
        pair_counter = Counter()
        type_set = set()
        for r in self.rallies:
            seq = []
            for s in r.strokes:
                if player_id is not None and s.get('player') != player_id:
                    continue
                st = normalize(s.get('stroke_type_name', ''))
                seq.append(st)
            for i in range(len(seq) - 1):
                a = seq[i]; b = seq[i+1]
                if a and b:
                    pair_counter[(a, b)] += 1
                    type_set.add(a); type_set.add(b)
        nodes = [{'name': t} for t in sorted(type_set)]
        if not nodes:
            return {'nodes': [], 'links': []}
        for (a, b), v in pair_counter.items():
            if v >= max(1, min_count):
                links.append({'source': a, 'target': b, 'value': int(v)})
        return {'nodes': nodes, 'links': links}

    def get_theme_river_data(self, window_sec: float = 2.0, player_id: Optional[int] = None) -> Dict[str, object]:
        label_norm = {
            '杀': '杀球', 'smash': '杀球',
            '抽': '抽球', 'drive': '抽球',
            '吊': '吊球', 'drop': '吊球',
            '网': '网前', 'net': '网前',
            '挑': '挑球', 'lift': '挑球',
            '高': '高远', 'clear': '高远'
        }
        def normalize(t: str) -> str:
            xl = (t or '').lower()
            for k, v in label_norm.items():
                if k in xl:
                    return v
            return t or '未知'
        strokes = []
        for r in self.rallies:
            for s in r.strokes:
                if player_id is not None and s.get('player') != player_id:
                    continue
                frame = s.get('frame', None)
                if frame is None or frame >= len(self.df):
                    continue
                t = float(self.df.iloc[frame]['time_seconds']) if 'time_seconds' in self.df.columns else float(frame) * 0.04
                st = normalize(s.get('stroke_type_name', ''))
                strokes.append((t, st))
        if not strokes:
            return {'times': [], 'series': {}}
        strokes.sort(key=lambda x: x[0])
        t_min = strokes[0][0]
        t_max = strokes[-1][0]
        step = max(0.2, window_sec / 5.0)
        times = np.arange(t_min, t_max + step, step)
        types = sorted(list(set(st for _, st in strokes)))
        series = {tp: [] for tp in types}
        half = window_sec / 2.0
        ts_arr = np.array([t for t, _ in strokes])
        st_arr = np.array([st for _, st in strokes])
        for t in times:
            mask = (ts_arr >= t - half) & (ts_arr <= t + half)
            total = int(mask.sum())
            if total == 0:
                for tp in types:
                    series[tp].append(0.0)
            else:
                for tp in types:
                    series[tp].append(float(np.sum(st_arr[mask] == tp)) / float(total))
        return {'times': times.tolist(), 'series': series, 'window_sec': window_sec}
