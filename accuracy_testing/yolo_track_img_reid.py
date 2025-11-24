#!/usr/bin/env python3
import cv2
import re
import torch
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# BoxMOT trackers
from boxmot import BoostTrack
from boxmot import OcSort
from boxmot import StrongSort
from boxmot import HybridSort
from boxmot import DeepOcSort
from boxmot import ByteTrack
from boxmot import BotSort

# ----------------------------
# Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Frame-by-frame tracking using YOLO + BoxMOT")
    parser.add_argument('--yolo-weights', type=str, default='yolov8n.pt',
                        help="YOLO model weights (yolov8n.pt, yolov11s.pt, etc.)")

    parser.add_argument("--tracking-method", type=str, default="ocsort",
                        choices=["ocsort","boosttrack","deepocsort","strongsort","hybridsort","bytetrack","botsort"],
                        help="Tracking algorithm to use")

    parser.add_argument('--reid-model', type=str, default="osnet_x0_25_msmt17.pt",
                        choices=[
                            "lmbn_n_cuhk03_d.pt","osnet_x0_25_market1501.pt","mobilenetv2_x1_4_msmt17.engine",
                            "resnet50_msmt17.onnx","osnet_x1_0_msmt17.pt",
                            "clip_market1501.pt","clip_vehicleid.pt"
                        ],
                        help="Re-ID model for trackers")

    parser.add_argument("--source", type=str, required=True,
                        help="Folder containing .jpg frames")

    parser.add_argument("--output", type=str, default="tracking_results.txt",
                        help="Output .txt file for MOT results")

    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO confidence threshold")

    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="Filter by class IDs (optional)")

    return parser.parse_args()

# ----------------------------
# Extract frame number from filename
# ----------------------------
def extract_frame_number(img_path):
    match = re.search(r'frame(\d+)', img_path.stem)
    return int(match.group(1)) if match else 0


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Load YOLO detector
    # ----------------------------
    print(f"\nLoading YOLO model: {args.yolo_weights}")
    detector = YOLO(args.yolo_weights)

    # ----------------------------
    # Initialize tracker
    # ----------------------------
    reid_weights = Path(args.reid_model)

    if args.tracking_method == "boosttrack":
        tracker = BoostTrack(reid_weights=reid_weights, device=device, half=False)
    elif args.tracking_method == "ocsort":
        tracker = OcSort(reid_weights=reid_weights, device=device, half=False)
    elif args.tracking_method == "deepocsort":
        tracker = DeepOcSort(reid_weights=reid_weights, device=device, half=False)
    elif args.tracking_method == "strongsort":
        tracker = StrongSort(reid_weights=reid_weights, device=device, half=False)
    elif args.tracking_method == "hybridsort":
        tracker = HybridSort(reid_weights=reid_weights, device=device, half=False)
    elif args.tracking_method == "bytetrack":
        tracker = ByteTrack(reid_weights=reid_weights, device=device, half=False)
    elif args.tracking_method == "botsort":
        tracker = BotSort(reid_weights=reid_weights, device=device, half=False)
    else:
        raise ValueError("Unknown tracker type")

    # ----------------------------
    # Load images
    # ----------------------------
    source_dir = Path(args.source)
    image_files = sorted(source_dir.glob("*.jpg"), key=extract_frame_number)

    if not image_files:
        print(f"ERROR: No .jpg images found in {args.source}")
        return

    output_file = open(args.output, "w")

    print("\n======== STARTING YOLO + TRACKING ========\n")

    for img_path in image_files:

        frame_num = extract_frame_number(img_path)
        print(f"Processing file: {img_path.name}  →  frame {frame_num}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"ERROR: Cannot read {img_path.name}")
            continue

        # ----------------------------
        # YOLO detection
        # ----------------------------
        results = detector(frame, conf=args.conf, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()   # (N,4)
        confs = results.boxes.conf.cpu().numpy()   # (N,)
        labels = results.boxes.cls.cpu().numpy()   # (N,)

        # Optional class filter
        if args.classes is not None:
            mask = np.isin(labels, args.classes)
            boxes = boxes[mask]
            confs = confs[mask]
            labels = labels[mask]

        if len(boxes) == 0:
            print(f"No detections at frame {frame_num}")
            continue

        # (x1, y1, x2, y2, conf, cls)
        detections = np.concatenate([boxes, confs[:, None], labels[:, None]], axis=1)

        # ----------------------------
        # Tracker update
        # ----------------------------
        tracks = tracker.update(detections, frame)

        if len(tracks) == 0:
            print(f"No tracks at frame {frame_num}")
            continue

        # ----------------------------
        # Write MOT output
        # ----------------------------
        for det in tracks:
            x1, y1, x2, y2, track_id, conf, cls, *_ = det
            w = x2 - x1
            h = y2 - y1

            line = f"{frame_num},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
            output_file.write(line)

        print(f"✔ {len(tracks)} tracks written for frame {frame_num}")

        # ----------------------------
        # Visualization
        # ----------------------------
        tracker.plot_results(frame, show_trajectories=True)
        cv2.imshow("YOLO Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    output_file.close()
    cv2.destroyAllWindows()

    print("\n======== DONE! OUTPUT SAVED TO ========")
    print(args.output)


if __name__ == "__main__":
    main()

