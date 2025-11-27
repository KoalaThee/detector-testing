from ultralytics import YOLO
import cv2
import time
import argparse
import statistics
import torch
import numpy as np
import csv
from PIL import Image
from pathlib import Path
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights as Weights
)

# BoxMOT imports (optional - only if using BoxMOT trackers)
try:
    from boxmot import BoostTrack, OcSort, StrongSort, HybridSort, DeepOcSort, ByteTrack, BotSort
    BOXMOT_AVAILABLE = True
except Exception as e:
    print("Warning: boxmot not available. BoxMOT trackers will be disabled.")
    print("BoxMOT import error: ", repr(e))
    BOXMOT_AVAILABLE = False


class BenchmarkResult:
    """Store benchmark results for a single algorithm combination"""
    def __init__(self, name, detector_type, tracker_type):
        self.name = name
        self.detector_type = detector_type
        self.tracker_type = tracker_type
        self.inference_times = []
        self.frame_count = 0
        self.total_elapsed = 0
        self.avg_tracks_per_frame = []
        
    def add_result(self, infer_ms, num_tracks=0):
        self.inference_times.append(infer_ms)
        self.avg_tracks_per_frame.append(num_tracks)
        self.frame_count += 1
        
    def get_stats(self):
        if not self.inference_times:
            return None
        return {
            'name': self.name,
            'min_ms': min(self.inference_times),
            'max_ms': max(self.inference_times),
            'mean_ms': statistics.mean(self.inference_times),
            'median_ms': statistics.median(self.inference_times),
            'stddev_ms': statistics.stdev(self.inference_times) if len(self.inference_times) > 1 else 0,
            'fps': 1000.0 / statistics.mean(self.inference_times) if statistics.mean(self.inference_times) > 0 else 0,
            'total_time': self.total_elapsed,
            'frame_count': self.frame_count,
            'avg_tracks': statistics.mean(self.avg_tracks_per_frame) if self.avg_tracks_per_frame else 0
        }


class YOLODetector:
    """YOLO detector wrapper"""
    def __init__(self, model_path, conf=0.5, classes=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes
        
    def detect(self, frame):
        results = self.model(frame, conf=self.conf, classes=self.classes, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy()
        return boxes, confs, labels


class RCNNDetector:
    """Faster R-CNN detector wrapper"""
    def __init__(self, conf=0.5, classes=None, device='cpu'):
        self.device = torch.device(device)
        weights = Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=conf)
        self.detector.to(self.device).eval()
        self.transform = weights.transforms()
        self.conf = conf
        self.classes = classes
        
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        input_tensor = self.transform(pil_img).to(self.device)
        
        with torch.inference_mode():
            output = self.detector([input_tensor])[0]
        
        scores = output["scores"].cpu().numpy()
        keep = scores >= self.conf
        
        boxes = output["boxes"][keep].cpu().numpy()
        labels = output["labels"][keep].cpu().numpy()
        confs = scores[keep]
        
        if self.classes is not None:
            mask = np.isin(labels, self.classes)
            boxes = boxes[mask]
            labels = labels[mask]
            confs = confs[mask]
            
        return boxes, confs, labels


def create_tracker(tracker_type, reid_model=None, device='cpu'):
    """Create a BoxMOT tracker"""
    if not BOXMOT_AVAILABLE:
        raise ValueError("BoxMOT not available. Install with: pip install boxmot")
    
    # Pass reid_weights to all trackers - BoxMOT handles it internally
    # Some trackers (ocsort, bytetrack) don't use ReID but accept the parameter
    reid_weights = Path(reid_model) if reid_model else None
    device = torch.device(device)
    
    tracker_map = {
        "boosttrack": BoostTrack,
        "ocsort": OcSort,
        "deepocsort": DeepOcSort,
        "strongsort": StrongSort,
        "hybridsort": HybridSort,
        "bytetrack": ByteTrack,
        "botsort": BotSort
    }
    
    if tracker_type not in tracker_map:
        raise ValueError(f"Unknown tracker: {tracker_type}")
    
    return tracker_map[tracker_type](reid_weights=reid_weights, device=device, half=False)


def benchmark_yolo_ultralytics(video_path, model_path, tracker_yaml, conf=0.5, classes=None, 
                                max_frames=None, show_viz=False):
    """Benchmark YOLO with Ultralytics built-in tracker"""
    print(f"  Loading model: {model_path}...")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    result = BenchmarkResult(
        f"YOLO+{tracker_yaml.replace('.yaml', '').upper()}",
        "YOLO",
        tracker_yaml.replace('.yaml', '')
    )
    
    # Warmup
    warmup_frames = 5
    print(f"  Warmup ({warmup_frames} frames)...", end=" ", flush=True)
    for i in range(warmup_frames):
        ret, frame = cap.read()
        if ret:
            model.track(frame, persist=True, tracker=tracker_yaml, conf=conf, 
                       classes=classes, verbose=False)
            print(f"{i+1}", end=" ", flush=True)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("done")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    total_frames = max_frames if max_frames else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_start = time.time()
    
    print(f"  Processing {total_frames} frames:", end=" ", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
            
        t0 = time.time()
        results = model.track(frame, persist=True, tracker=tracker_yaml, conf=conf,
                             classes=classes, verbose=False)
        t1 = time.time()
        
        infer_ms = (t1 - t0) * 1000.0
        num_tracks = len(results[0].boxes.id) if results[0].boxes.id is not None else 0
        result.add_result(infer_ms, num_tracks)
        frame_count += 1
        
        # Progress indicator every 10 frames
        if frame_count % 10 == 0 or frame_count == total_frames:
            print(f"{frame_count}", end=" ", flush=True)
        
        if show_viz:
            annotated = results[0].plot()
            cv2.putText(annotated, f"{result.name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {1000.0/infer_ms:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f"Benchmark: {result.name}", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    print("done")
    result.total_elapsed = time.time() - t_start
    cap.release()
    return result


def benchmark_yolo_boxmot(video_path, model_path, tracker_type, reid_model, conf=0.5, 
                          classes=None, max_frames=None, show_viz=False, device='cpu'):
    """Benchmark YOLO detector with BoxMOT tracker"""
    print(f"  Loading YOLO model: {model_path}...")
    detector = YOLODetector(model_path, conf=conf, classes=classes)
    print("  Creating tracker...")
    tracker = create_tracker(tracker_type, reid_model, device=device)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    result = BenchmarkResult(
        f"YOLO+{tracker_type.upper()}",
        "YOLO",
        tracker_type
    )
    
    # Warmup
    warmup_frames = 5
    print(f"  Warmup ({warmup_frames} frames)...", end=" ", flush=True)
    for i in range(warmup_frames):
        ret, frame = cap.read()
        if ret:
            boxes, confs, labels = detector.detect(frame)
            if len(boxes) > 0:
                detections = np.concatenate([boxes, confs[:, None], labels[:, None]], axis=1)
                tracker.update(detections, frame)
            print(f"{i+1}", end=" ", flush=True)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("done")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    total_frames = max_frames if max_frames else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_start = time.time()
    
    print(f"  Processing {total_frames} frames:", end=" ", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
            
        t0 = time.time()
        boxes, confs, labels = detector.detect(frame)
        
        if len(boxes) > 0:
            detections = np.concatenate([boxes, confs[:, None], labels[:, None]], axis=1)
            tracks = tracker.update(detections, frame)
            num_tracks = len(tracks)
        else:
            tracks = []
            num_tracks = 0
        t1 = time.time()
        
        infer_ms = (t1 - t0) * 1000.0
        result.add_result(infer_ms, num_tracks)
        frame_count += 1
        
        # Progress indicator every 10 frames
        if frame_count % 10 == 0 or frame_count == total_frames:
            print(f"{frame_count}", end=" ", flush=True)
        
        if show_viz:
            if len(tracks) > 0:
                tracker.plot_results(frame, show_trajectories=True)
            cv2.putText(frame, f"{result.name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {1000.0/infer_ms:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f"Benchmark: {result.name}", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    print("done")
    result.total_elapsed = time.time() - t_start
    cap.release()
    return result


def benchmark_rcnn_boxmot(video_path, tracker_type, reid_model, conf=0.5, 
                         classes=None, max_frames=None, show_viz=False, device='cpu'):
    """Benchmark Faster R-CNN detector with BoxMOT tracker"""
    print("  Loading R-CNN detector...")
    detector = RCNNDetector(conf=conf, classes=classes, device=device)
    print("  Creating tracker...")
    tracker = create_tracker(tracker_type, reid_model, device=device)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    result = BenchmarkResult(
        f"R-CNN+{tracker_type.upper()}",
        "R-CNN",
        tracker_type
    )
    
    # Warmup (reduced to 3 frames for CPU)
    warmup_frames = 3
    print(f"  Warmup ({warmup_frames} frames)...", end=" ", flush=True)
    for i in range(warmup_frames):
        ret, frame = cap.read()
        if ret:
            boxes, confs, labels = detector.detect(frame)
            if len(boxes) > 0:
                detections = np.concatenate([boxes, confs[:, None], labels[:, None]], axis=1)
                tracker.update(detections, frame)
            print(f"{i+1}", end=" ", flush=True)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("done")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    total_frames = max_frames if max_frames else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_start = time.time()
    
    print(f"  Processing {total_frames} frames:", end=" ", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
            
        t0 = time.time()
        boxes, confs, labels = detector.detect(frame)
        
        if len(boxes) > 0:
            detections = np.concatenate([boxes, confs[:, None], labels[:, None]], axis=1)
            tracks = tracker.update(detections, frame)
            num_tracks = len(tracks)
        else:
            tracks = []
            num_tracks = 0
        t1 = time.time()
        
        infer_ms = (t1 - t0) * 1000.0
        result.add_result(infer_ms, num_tracks)
        frame_count += 1
        
        # Progress indicator every 10 frames
        if frame_count % 10 == 0 or frame_count == total_frames:
            print(f"{frame_count}", end=" ", flush=True)
        
        if show_viz:
            if len(tracks) > 0:
                tracker.plot_results(frame, show_trajectories=True)
            cv2.putText(frame, f"{result.name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {1000.0/infer_ms:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f"Benchmark: {result.name}", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    print("done")
    result.total_elapsed = time.time() - t_start
    cap.release()
    return result


def generate_all_ultralytics_combinations():
    """Generate all YOLO + Ultralytics tracker combinations"""
    # All YOLO models supported by Ultralytics
    yolo_models = [
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
        'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
        'yolov9t.pt', 'yolov9s.pt', 'yolov9m.pt', 'yolov9c.pt', 'yolov9e.pt',
        'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10b.pt', 'yolov10l.pt', 'yolov10x.pt',
        'yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt', 'yolov11l.pt', 'yolov11x.pt',
        'yolov6n.pt', 'yolov6s.pt', 'yolov6m.pt', 'yolov6l.pt',
        'rtdetr-l.pt', 'rtdetr-x.pt'
    ]
    
    # Ultralytics trackers
    ultralytics_trackers = ['bytetrack.yaml', 'botsort.yaml']
    
    combinations = []
    for model in yolo_models:
        for tracker in ultralytics_trackers:
            combinations.append({
                'type': 'yolo_ultralytics',
                'model': model,
                'tracker': tracker,
                'reid': None
            })
    
    return combinations


def generate_all_boxmot_combinations(reid_model='osnet_x0_25_market1501.pt'):
    """Generate all R-CNN + BoxMOT tracker combinations"""
    # All BoxMOT trackers
    boxmot_trackers = [
        'boosttrack', 'ocsort', 'strongsort', 'hybridsort', 
        'deepocsort', 'bytetrack', 'botsort'
    ]
    
    combinations = []
    for tracker in boxmot_trackers:
        combinations.append({
            'type': 'rcnn_boxmot',
            'model': None,
            'tracker': tracker,
            'reid': reid_model
        })
    
    return combinations


def parse_combinations(combos_str):
    """Parse combination strings like 'yolo:yolov8n.pt:bytetrack.yaml' or 'yolo_boxmot:yolov8n.pt:botsort:osnet_x0_25_msmt17.pt'"""
    combinations = []
    for combo in combos_str:
        parts = combo.split(':')
        if len(parts) < 3:
            raise ValueError(f"Invalid combination format: {combo}. Use format: 'type:detector:tracker[:reid]'")
        
        combo_type = parts[0].lower()
        
        if combo_type == 'yolo':
            # YOLO + Ultralytics tracker: yolo:yolov8n.pt:bytetrack.yaml
            combinations.append({
                'type': 'yolo_ultralytics',
                'model': parts[1],
                'tracker': parts[2],
                'reid': None
            })
        elif combo_type == 'yolo_boxmot':
            # YOLO + BoxMOT: yolo_boxmot:yolov8n.pt:botsort:osnet_x0_25_msmt17.pt
            if len(parts) < 4:
                raise ValueError(f"YOLO+BoxMOT requires ReID model: {combo}")
            combinations.append({
                'type': 'yolo_boxmot',
                'model': parts[1],
                'tracker': parts[2],
                'reid': parts[3]
            })
        elif combo_type == 'rcnn':
            # R-CNN + BoxMOT: rcnn:botsort:osnet_x0_25_msmt17.pt
            if len(parts) < 3:
                raise ValueError(f"R-CNN+BoxMOT requires ReID model: {combo}")
            combinations.append({
                'type': 'rcnn_boxmot',
                'model': None,
                'tracker': parts[1],
                'reid': parts[2]
            })
        else:
            raise ValueError(f"Unknown combination type: {combo_type}")
    
    return combinations


def print_comparison(results, csv_output=None):
    """Print benchmark comparison - mean ms and FPS only, and export to CSV"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Sort by FPS (descending)
    stats_list = []
    for r in results:
        stats = r.get_stats()
        if stats is not None:
            stats['detector'] = r.detector_type
            stats['tracker'] = r.tracker_type
            stats_list.append(stats)
    
    stats_list.sort(key=lambda x: x['fps'], reverse=True)
    
    if len(stats_list) == 0:
        print("No results to compare.")
        return
    
    # Print header
    print(f"\n{'Algorithm':<40} {'Mean (ms)':<15} {'FPS':<15}")
    print("-" * 70)
    
    # Print each result
    for stats in stats_list:
        print(f"{stats['name']:<40} {stats['mean_ms']:>14.2f} {stats['fps']:>14.2f}")
    
    print("="*80)
    
    # Export to CSV
    if csv_output:
        csv_path = Path(csv_output)
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['detector', 'tracker', 'mean_inference_time_ms', 'fps'])
                # Write data
                for stats in stats_list:
                    writer.writerow([
                        stats['detector'],
                        stats['tracker'],
                        f"{stats['mean_ms']:.2f}",
                        f"{stats['fps']:.2f}"
                    ])
            print(f"\n✓ Results exported to: {csv_path}")
        except Exception as e:
            print(f"\n⚠ Error exporting to CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-algorithm tracking benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all Ultralytics combinations (all YOLO models with bytetrack.yaml and botsort.yaml)
  python test_yolo.py --video testing_vid5.mp4 --all-ultralytics
  
  # Run all BoxMOT combinations (all 7 trackers with R-CNN)
  python test_yolo.py --video testing_vid5.mp4 --all-boxmot
  
  # Run both Ultralytics and BoxMOT combinations
  python test_yolo.py --video testing_vid5.mp4 --all-ultralytics --all-boxmot --csv results.csv
  
  # Custom combinations
  python test_yolo.py --video testing_vid5.mp4 --combos yolo:yolov8n.pt:bytetrack.yaml
  
  # Multiple custom algorithms
  python test_yolo.py --video testing_vid5.mp4 \\
    --combos yolo:yolov8n.pt:bytetrack.yaml \\
             yolo_boxmot:yolov8n.pt:botsort:osnet_x0_25_market1501.pt \\
             rcnn:strongsort:osnet_x0_25_market1501.pt
  
  # With visualization
  python test_yolo.py --video testing_vid5.mp4 \\
    --combos yolo:yolov8s.pt:bytetrack.yaml yolo:yolov8s.pt:botsort.yaml \\
    --show-viz
  
Combination formats:
  - yolo:MODEL:TRACKER.yaml          (YOLO + Ultralytics tracker)
  - yolo_boxmot:MODEL:TRACKER:REID   (YOLO + BoxMOT tracker)
  - rcnn:TRACKER:REID                (R-CNN + BoxMOT tracker)
        """
    )
    
    parser.add_argument('--video', type=str, default='testing_vid5.mp4',
                       help='Video file path')
    parser.add_argument('--combos', type=str, nargs='+', default=None,
                       help='Algorithm combinations (see examples). Not required if using --all-ultralytics or --all-boxmot')
    parser.add_argument('--all-ultralytics', action='store_true',
                       help='Run all YOLO models with Ultralytics trackers (bytetrack.yaml, botsort.yaml)')
    parser.add_argument('--all-boxmot', action='store_true',
                       help='Run all BoxMOT trackers with R-CNN detector')
    parser.add_argument('--reid-model', type=str, default='osnet_x0_25_market1501.pt',
                       help='ReID model path for BoxMOT trackers (default: osnet_x0_25_market1501.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Filter by class IDs (e.g., 0 for person only)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (for quick testing)')
    parser.add_argument('--show-viz', action='store_true',
                       help='Show visualization windows')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (auto detects CUDA)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Export results to CSV file (e.g., results.csv)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.combos and not args.all_ultralytics and not args.all_boxmot:
        parser.error("Either --combos, --all-ultralytics, or --all-boxmot must be specified")
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Generate combinations based on flags
    combinations = []
    
    if args.all_ultralytics:
        print("Generating all Ultralytics combinations...")
        ultralytics_combos = generate_all_ultralytics_combinations()
        combinations.extend(ultralytics_combos)
        print(f"  Added {len(ultralytics_combos)} Ultralytics combinations")
    
    if args.all_boxmot:
        print("Generating all BoxMOT combinations...")
        boxmot_combos = generate_all_boxmot_combinations(reid_model=args.reid_model)
        combinations.extend(boxmot_combos)
        print(f"  Added {len(boxmot_combos)} BoxMOT combinations")
    
    if args.combos:
        print("Parsing custom combinations...")
        try:
            custom_combos = parse_combinations(args.combos)
            combinations.extend(custom_combos)
            print(f"  Added {len(custom_combos)} custom combinations")
        except ValueError as e:
            print(f"Error parsing combinations: {e}")
            return
    
    if not combinations:
        print("Error: No combinations to benchmark")
        return
    
    print(f"\n{'='*80}")
    print(f"BENCHMARKING {len(combinations)} ALGORITHM COMBINATION(S)")
    print(f"{'='*80}")
    for i, combo in enumerate(combinations, 1):
        print(f"{i}. {combo['type']}: {combo.get('model', 'N/A')} + {combo['tracker']}")
    print(f"{'='*80}\n")
    
    # Get video info
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Video: {args.video}")
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Total frames: {total_frames}")
    if args.max_frames:
        print(f"Processing limit: {args.max_frames} frames")
    print()
    
    # Run benchmarks
    results = []
    for i, combo in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] Benchmarking: {combo['type']} + {combo['tracker']}")
        print("-" * 80)
        
        try:
            if combo['type'] == 'yolo_ultralytics':
                result = benchmark_yolo_ultralytics(
                    args.video, combo['model'], combo['tracker'],
                    conf=args.conf, classes=args.classes,
                    max_frames=args.max_frames, show_viz=args.show_viz
                )
            elif combo['type'] == 'yolo_boxmot':
                if not BOXMOT_AVAILABLE:
                    print(f"  ⚠ Skipping {combo['type']}: BoxMOT not available")
                    continue
                result = benchmark_yolo_boxmot(
                    args.video, combo['model'], combo['tracker'], combo['reid'],
                    conf=args.conf, classes=args.classes,
                    max_frames=args.max_frames, show_viz=args.show_viz, device=device
                )
            elif combo['type'] == 'rcnn_boxmot':
                if not BOXMOT_AVAILABLE:
                    print(f"  ⚠ Skipping {combo['type']}: BoxMOT not available")
                    continue
                result = benchmark_rcnn_boxmot(
                    args.video, combo['tracker'], combo['reid'],
                    conf=args.conf, classes=args.classes,
                    max_frames=args.max_frames, show_viz=args.show_viz, device=device
                )
            else:
                print(f"  ⚠ Unknown combination type: {combo['type']}")
                continue
            
            results.append(result)
            stats = result.get_stats()
            if stats:
                print(f"  ✓ Completed: {stats['frame_count']} frames")
                print(f"    Average FPS: {stats['fps']:.2f}")
                print(f"    Average inference: {stats['mean_ms']:.2f} ms")
                print(f"    Average tracks: {stats['avg_tracks']:.1f}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    if results:
        print_comparison(results, csv_output=args.csv)
    else:
        print("\nNo successful benchmarks to compare.")
    
    if args.show_viz:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
