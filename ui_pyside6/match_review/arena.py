import numpy as np
from .panels import MplCanvas
from mpl_toolkits.mplot3d import Axes3D

class Arena3D(MplCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, width=6, height=5)
        self.axes.remove()
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_facecolor('#1e1e1e')
        # Remove axis backgrounds
        self.axes.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.axes.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.axes.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        
    def plot_rally(self, rally_df, strokes):
        self.axes.cla()
        
        # 1. Estimate Z (Height)
        # Heuristic: Start/End of flight (hits) have specific heights based on stroke type
        # For now, just a simple arc between hits?
        # Better: use simple parabola based on flight time and distance
        
        # Extract segments between hits
        hit_indices = rally_df[rally_df['is_hit'] == 1].index.tolist()
        
        # Add start and end of rally to indices if not present
        if not hit_indices:
            indices = [rally_df.index[0], rally_df.index[-1]]
        else:
            indices = hit_indices
            if indices[0] != rally_df.index[0]: indices.insert(0, rally_df.index[0])
            if indices[-1] != rally_df.index[-1]: indices.append(rally_df.index[-1])
            
        xs = rally_df['ball_x'].values
        ys = rally_df['ball_y'].values
        # Invert Y for visualization (0 at bottom)
        ys = 720 - ys
        
        zs = np.zeros_like(xs, dtype=float)
        
        # Simple gravity model simulation or interpolation
        for i in range(len(indices) - 1):
            start = indices[i] - rally_df.index[0]
            end = indices[i+1] - rally_df.index[0]
            if end <= start: continue
            
            segment_len = end - start
            # Parabola: z = 4 * h * (x)(1-x) where x is 0..1
            # Height depends on distance
            dist = np.sqrt((xs[start]-xs[end])**2 + (ys[start]-ys[end])**2)
            peak_height = 100 + dist * 0.5 # Heuristic px height
            
            t = np.linspace(0, 1, segment_len)
            z_arc = 4 * peak_height * t * (1 - t)
            
            # Add base height (e.g. hit point height)
            base_h = 100 # 1 meter approx
            zs[start:end] = z_arc + base_h
            
        # Plot Trajectory
        self.axes.plot(xs, ys, zs, color='#00ffcc', linewidth=2, label='球路轨迹')
        
        # Plot Projection (Shadow)
        self.axes.plot(xs, ys, np.zeros_like(zs), color='#00ffcc', linewidth=1, alpha=0.3, linestyle='--')
        
        # Plot Hits
        hit_rows = rally_df[rally_df['is_hit'] == 1]
        for _, row in hit_rows.iterrows():
            idx = int(row.name - rally_df.index[0])
            if idx < len(xs):
                self.axes.scatter(xs[idx], ys[idx], zs[idx], color='red', s=50, marker='x')
                
        # Set limits
        self.axes.set_xlim(0, 1280)
        self.axes.set_ylim(0, 720)
        self.axes.set_zlim(0, 600)
        
        self.axes.set_xlabel('宽度 (X)')
        self.axes.set_ylabel('深度 (Y)')
        self.axes.set_zlabel('高度 (Z)')
        
        # View angle
        self.axes.view_init(elev=20, azim=-60)
        
        self.draw()
