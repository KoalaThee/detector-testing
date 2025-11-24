#!/usr/bin/env python3
import cv2
import re
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights as Weights
)

# Correct imports from boxmot
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
    parser = argparse.ArgumentParser(description="Frame-by-frame tracking")
    parser.add_argument("--tracking-method", type=str, default="ocsort",
                        choices=["ocsort", "boosttrack", "deepocsort", "strongsort", "hybridsort", "deepocsort", "bytetrack", "botsort"],
                        help="Tracking algorithm to use")
    parser.add_argument('--reid-model', type=str, default="osnet_x0_25_msmt17.pt", choices=["lmbn_n_cuhk03_d.pt","osnet_x0_25_market1501.pt","mobilenetv2_x1_4_msmt17.engine", "resnet50_msmt17.onnx", "osnet_x1_0_msmt17.pt", "clip_market1501.pt", "clip_vehicleid.pt"], help="Re-id Model")
    parser.add_argument("--source", type=str, required=True,
                        help="Folder containing .jpg frames")
    parser.add_argument("--output", type=str, default="tracking_results.txt",
                        help="Output .txt file for tracking results")    
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="Filter by class IDs (optional)")
    return parser.parse_args()

# ----------------------------
# Helper: extract frame number
# ----------------------------
def extract_frame_number(img_path):
    match = re.search(r'frame(\d+)', img_path.stem)
    if match:
        return int(match.group(1))
    return 0

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Load detector
    # ----------------------------
    weights = Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    detector.to(device).eval()
    transform = weights.transforms()

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
    # Load image frames
    # ----------------------------
    source_dir = Path(args.source)
    image_files = sorted(source_dir.glob("*.jpg"), key=extract_frame_number)

    if not image_files:
        print(f"ERROR: No .jpg images found in folder: {args.source}")
        return
        
    # ----------------------------
    # Open output file
    # ----------------------------
    output_file = open(args.output, "w")

    print("\n======== STARTING PROCESSING ========\n")

    with torch.inference_mode():
        for img_path in image_files:

            # Extract frame number
            frame_num = extract_frame_number(img_path)
            print(f"Processing file: {img_path.name}  →  extracted frame = {frame_num}")

            # Load image
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"ERROR: Cannot read image: {img_path.name}")
                continue

            # Convert to RGB and prepare tensor
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            input_tensor = transform(pil_img).to(device)

            # Run detection
            output = detector([input_tensor])[0]
            scores = output["scores"].cpu().numpy()
            keep = scores >= args.conf

            boxes = output["boxes"][keep].cpu().numpy()
            labels = output["labels"][keep].cpu().numpy()
            confs = scores[keep]

            # Optional: filter by class
            if args.classes is not None:
                mask = np.isin(labels, args.classes)
                boxes = boxes[mask]
                labels = labels[mask]
                confs = confs[mask]

            if len(boxes) == 0:
                print(f"No detections found at frame {frame_num}")
                continue

            # Prepare detections → (x1, y1, x2, y2, conf, cls)
            detections = np.concatenate([boxes, confs[:, None], labels[:, None]], axis=1)

            # Update tracker
            results = tracker.update(detections, frame)

            if len(results) == 0:
                print(f"No tracks for frame {frame_num}")
                continue

            # Write results
            for det in results:
                x1, y1, x2, y2, track_id, conf, cls, *_ = det
                w = x2 - x1
                h = y2 - y1
                output_line = f"{frame_num},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
                output_file.write(output_line)

            print(f"✔ {len(results)} detections written for frame {frame_num}")

            # Visualization
            tracker.plot_results(frame, show_trajectories=True)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    output_file.close()
    cv2.destroyAllWindows()

    print("\n======== DONE. OUTPUT SAVED TO ========")
    print(args.output)


if __name__ == "__main__":
    main()

