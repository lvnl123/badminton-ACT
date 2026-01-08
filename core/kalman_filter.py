import numpy as np
from collections import deque

class KalmanFilter:
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=10.0):
        self.dt = dt
        
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        self.x = None
        self.P = None
        self.F = None
        self.H = None
        self.Q = None
        self.R = None
        
        self.initialized = False
        
    def init(self, x, y):
        self.x = np.array([x, y, 0, 0], dtype=np.float64)
        self.P = np.eye(4) * 1000
        
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        self.Q = np.eye(4) * self.process_noise
        self.R = np.eye(2) * self.measurement_noise
        
        self.initialized = True
        
    def predict(self):
        if not self.initialized:
            return None
            
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]
    
    def update(self, measurement):
        if not self.initialized:
            self.init(measurement[0], measurement[1])
            return measurement
        
        z = np.array(measurement, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x[:2]

class KalmanTrajectorySmoother:
    def __init__(self, max_gap=10, process_noise=1.0, measurement_noise=10.0):
        self.max_gap = max_gap
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
    def smooth(self, x_list, y_list, vis_list):
        if len(x_list) == 0:
            return x_list, y_list, vis_list
        
        smoothed_x = []
        smoothed_y = []
        smoothed_vis = []
        
        kf = KalmanFilter(
            dt=1.0,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )
        
        gap_count = 0
        last_valid_x = None
        last_valid_y = None
        
        for i in range(len(x_list)):
            if vis_list[i] == 1:
                if not kf.initialized:
                    kf.init(x_list[i], y_list[i])
                    smoothed_x.append(x_list[i])
                    smoothed_y.append(y_list[i])
                    smoothed_vis.append(1)
                    last_valid_x = x_list[i]
                    last_valid_y = y_list[i]
                else:
                    predicted = kf.predict()
                    updated = kf.update([x_list[i], y_list[i]])
                    
                    smoothed_x.append(updated[0])
                    smoothed_y.append(updated[1])
                    smoothed_vis.append(1)
                    last_valid_x = x_list[i]
                    last_valid_y = y_list[i]
                
                gap_count = 0
            else:
                gap_count += 1
                
                if gap_count <= self.max_gap and kf.initialized:
                    predicted = kf.predict()
                    
                    if last_valid_x is not None:
                        dx = predicted[0] - last_valid_x
                        dy = predicted[1] - last_valid_y
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist < 200:
                            smoothed_x.append(predicted[0])
                            smoothed_y.append(predicted[1])
                            smoothed_vis.append(1)
                        else:
                            smoothed_x.append(0)
                            smoothed_y.append(0)
                            smoothed_vis.append(0)
                            kf = KalmanFilter(
                                dt=1.0,
                                process_noise=self.process_noise,
                                measurement_noise=self.measurement_noise
                            )
                    else:
                        smoothed_x.append(predicted[0])
                        smoothed_y.append(predicted[1])
                        smoothed_vis.append(1)
                else:
                    smoothed_x.append(0)
                    smoothed_y.append(0)
                    smoothed_vis.append(0)
                    kf = KalmanFilter(
                        dt=1.0,
                        process_noise=self.process_noise,
                        measurement_noise=self.measurement_noise
                    )
        
        return smoothed_x, smoothed_y, smoothed_vis
