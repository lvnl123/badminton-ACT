import torch
import torchvision
import cv2
import numpy as np
import os
from tqdm import tqdm
from .TrackNetAttention import TrackNetAttention
from .denoise import smooth
from .utils import write_json


def ball_detect(
    video_path,
    result_path,
    model_path="e:\\learn\\TrackNetV3_migrated\\model_best.pth",
    num_frames=3,
    threshold=0.5,
    frame_callback=None,
    progress_callback=None,
):
    """
    Detect shuttlecock trajectory in video using TrackNet with Attention model.
    
    Args:
        video_path: Path to input video file
        result_path: Path to save detection results
        model_path: Path to TrackNet with Attention model weights
        num_frames: Number of frames to process as input sequence (default: 3)
    """
    imgsz = [288, 512]
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    d_save_dir = os.path.join(result_path, "loca_info")
    f_source = str(video_path)

    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = TrackNetAttention().to(device)
    if os.path.exists(model_path):
        pretrained_dict = torch.load(model_path, map_location=device)
        model_dict = model.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded model from {model_path} (partial loading for attention layers)")
    else:
        print(f"Warning: Model file {model_path} not found. Using random weights.")
    model.eval()

    vid_cap = cv2.VideoCapture(f_source)
    video_end = False
    video_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {video_len} frames, {fps} fps, {w}x{h}")

    count = 0
    with tqdm(total=video_len, desc="Processing frames") as pbar:
        while vid_cap.isOpened():
            imgs = []
            for _ in range(num_frames):
                ret, img = vid_cap.read()
                if not ret:
                    video_end = True
                    break
                imgs.append(img)

            if video_end:
                break

            imgs_torch = []
            for img in imgs:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_torch = torchvision.transforms.ToTensor()(img).to(device)
                img_torch = torchvision.transforms.functional.resize(
                    img_torch, imgsz, antialias=True)
                imgs_torch.append(img_torch)

            imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)

            with torch.no_grad():
                preds = model(imgs_torch)
            preds = preds[0].detach().cpu().numpy()

            y_preds = preds > threshold
            y_preds = y_preds.astype('float32')
            y_preds = y_preds * 255
            y_preds = y_preds.astype('uint8')

            for i in range(num_frames):
                if np.amax(y_preds[i]) <= 0:
                    ball_dict = {
                        f"{count}": {
                            "visible": 0,
                            "x": 0,
                            "y": 0,
                        }
                    }
                    write_json(ball_dict, video_name, f"{d_save_dir}")
                    if frame_callback is not None and i < len(imgs):
                        frame_callback(count, imgs[i], None, 0)
                else:
                    pred_img = cv2.resize(y_preds[i], (w, h),
                                          interpolation=cv2.INTER_AREA)

                    (cnts, _) = cv2.findContours(pred_img, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
                    rects = [cv2.boundingRect(ctr) for ctr in cnts]
                    
                    if len(rects) > 0:
                        max_area_idx = 0
                        max_area = rects[max_area_idx][2] * rects[max_area_idx][3]

                        for ii in range(len(rects)):
                            area = rects[ii][2] * rects[ii][3]
                            if area > max_area:
                                max_area_idx = ii
                                max_area = area

                        target = rects[max_area_idx]
                        (cx_pred, cy_pred) = (int((target[0] + target[2] / 2)),
                                              int((target[1] + target[3] / 2)))
                    else:
                        cx_pred, cy_pred = 0, 0

                    ball_dict = {
                        f"{count}": {
                            "visible": 1 if cx_pred > 0 and cy_pred > 0 else 0,
                            "x": cx_pred,
                            "y": cy_pred,
                        }
                    }
                    write_json(ball_dict, video_name, f"{d_save_dir}")
                    if frame_callback is not None and i < len(imgs):
                        visible = 1 if cx_pred > 0 and cy_pred > 0 else 0
                        frame_callback(count, imgs[i], (cx_pred, cy_pred), visible)

                count += 1
                pbar.update(1)
                if progress_callback is not None:
                    progress_callback(count, video_len)

    while count < video_len:
        ball_dict = {
            f"{count}": {
                "visible": 0,
                "x": 0,
                "y": 0,
            }
        }
        write_json(ball_dict, video_name, f"{d_save_dir}")
        count += 1
        pbar.update(1)
        if progress_callback is not None:
            progress_callback(count, video_len)

    vid_cap.release()
    print(f"Detection completed. Results saved to {d_save_dir}")

    dd_save_dir = os.path.join(result_path, "loca_info_denoise")
    os.makedirs(dd_save_dir, exist_ok=True)

    print("Starting trajectory smoothing...")
    json_path = f"{d_save_dir}/{video_name}.json"
    smooth(json_path, court=None, save_path=dd_save_dir)
    print(f"Smoothed trajectory saved to {dd_save_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TrackNetV3 with Attention Shuttlecock Detection')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--result', type=str, default='./results', help='Path to save results')
    parser.add_argument('--model', type=str, default='ball_track_attention.pt', help='Path to model weights')
    parser.add_argument('--num_frames', type=int, default=3, help='Number of frames in input sequence')
    
    args = parser.parse_args()
    
    ball_detect(args.video, args.result, args.model, args.num_frames)
