import pandas as pd
import numpy as np
import os
import sys

from .utils import read_json, write_json
from .kalman_filter import KalmanTrajectorySmoother

def smooth(json_path, court=None, save_path="./loca_info_denoise"):
    json_name = os.path.splitext(os.path.basename(json_path))[0]

    df_ls = []
    loca_dict = read_json(json_path)

    for frame, vxy_dict in loca_dict.items():
        fvxy_ditc = {}
        fvxy_ditc["frame"] = int(frame)
        for key, value in vxy_dict.items():
            fvxy_ditc[key] = value
        df_ls.append(fvxy_ditc)
    df = pd.DataFrame(df_ls)
    df = df.fillna(0)

    x = df['x'].tolist()
    y = df['y'].tolist()
    vis = df['visible'].tolist()

    pre_dif = []
    for i in range(0, len(x)):
        if i == 0:
            pre_dif.append(0)
        else:
            pre_dif.append(
                ((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)**(1 / 2))

    abnormal = [0] * len(pre_dif)
    X_abn = x
    y_abn = y
    dif_error = 2
    for i in range(len(pre_dif)):
        if i == len(pre_dif):
            abnormal[i] = 0
        elif i == len(pre_dif) - 1:
            abnormal[i] = 0
        elif i == len(pre_dif) - 2:
            abnormal[i] = 0
        elif i == len(pre_dif) - 3:
            abnormal[i] = 0
        elif pre_dif[i] >= 100 and pre_dif[i + 1] >= 100:
            if vis[i:i + 2] == [1, 1]:
                abnormal[i] = 'bias1'
                X_abn[i] = 0
                y_abn[i] = 0
        elif pre_dif[i] >= 100 and pre_dif[i + 2] >= 100:
            if pre_dif[i + 1] < dif_error:
                if vis[i:i + 3] == [1, 1, 1]:
                    abnormal[i:i + 2] = ['bias2', 'bias2']
                    X_abn[i:i + 2] = [0, 0]
                    y_abn[i:i + 2] = [0, 0]
        elif i + 4 < len(pre_dif) and pre_dif[i] >= 100 and pre_dif[i + 3] >= 100:
            if pre_dif[i + 1] < dif_error and pre_dif[i + 2] < dif_error:
                if vis[i:i + 4] == [1, 1, 1, 1]:
                    abnormal[i:i + 3] = ['bias3', 'bias3', 'bias3']
                    X_abn[i:i + 3] = [0, 0, 0]
                    y_abn[i:i + 3] = [0, 0, 0]
        elif i + 5 < len(pre_dif) and pre_dif[i] >= 100 and pre_dif[i + 4] >= 100:
            if pre_dif[i + 1] < dif_error and pre_dif[i + 2] < dif_error and pre_dif[i + 3] < dif_error:
                if vis[i:i + 5] == [1, 1, 1, 1, 1]:
                    abnormal[i:i + 4] = ['bias4', 'bias4', 'bias4', 'bias4']
                    X_abn[i:i + 4] = [0, 0, 0, 0]
                    y_abn[i:i + 4] = [0, 0, 0, 0]

    vis2 = [1] * len(df)
    for i in range(len(df)):
        if X_abn[i] == 0 and y_abn[i] == 0:
            vis2[i] = 0

    smoother = KalmanTrajectorySmoother(max_gap=8, process_noise=5.0, measurement_noise=20.0)
    smoothed_x, smoothed_y, smoothed_vis = smoother.smooth(X_abn, y_abn, vis2)

    df['X'] = smoothed_x
    df['Y'] = smoothed_y

    for index, row in df.iterrows():
        frame = str(int(row["frame"]))
        visible = int(row["visible"])
        x = int(row["X"])
        y = int(row["Y"])

        if x == 0 or y == 0:
            visible = 0
        else:
            visible = 1
        
        ball_dict = {
            frame: {
                "visible": visible,
                "x": x,
                "y": y,
            }
        }

        write_json(ball_dict, json_name, f"{save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, default="loca_info_denoise", help="Output directory path")
    args = parser.parse_args()
    
    smooth(args.input, court=None, save_path=args.output)
