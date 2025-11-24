import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path
import re

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 Tracking')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--source', type=str, default='dataset1')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml')
    parser.add_argument('--output', type=str, default='tracking_results.txt')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--classes', nargs='+', type=int, default=[0])
    return parser.parse_args()

# Extract frame number e.g. dataset1_frame12.jpg → 12
def extract_frame_number(filename):
    match = re.search(r'frame(\d+)', filename.name)
    if match:
        return int(match.group(1))
    return -1


def main():
    args = parse_args()
    model = YOLO(args.model)

    # Load files and ensure correct numeric sorting
    source_dir = Path(args.source)
    image_files = sorted(source_dir.glob('*.jpg'),
                         key=lambda f: extract_frame_number(f))

    if not image_files:
        print(f"ERROR: No .jpg images found in folder: {args.source}")
        return

    output_file = open(args.output, 'w')

    print("\n======== STARTING PROCESSING ========\n")

    for img_path in image_files:

        # Extract frame number
        frame_num = extract_frame_number(img_path)

        print(f"Processing file: {img_path.name}  →  extracted frame = {frame_num}")

        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"ERROR: Cannot read image: {img_path.name}")
            continue

        # Run YOLO tracking
        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            classes=args.classes,
            verbose=False
        )

        # Get boxes
        if results[0].boxes.id is None:
            print(f"No detections found at frame {frame_num}")
            continue

        boxes = results[0].boxes.xywh.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        # Write detections
        for box, track_id, conf in zip(boxes, ids, confs):
            cx, cy, w, h = box

            # Convert center xywh to top-left xywh
            x = cx - w / 2
            y = cy - h / 2

            output_line = f"{frame_num},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
            output_file.write(output_line)

        print(f"✔ {len(ids)} detections written for frame {frame_num}")

    print("\n======== DONE. OUTPUT SAVED TO ========")
    print(args.output)

    output_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

