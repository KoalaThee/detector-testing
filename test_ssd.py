import cv2
import time
import argparse
import statistics
import torch
import numpy as np
import csv
import ssl
from PIL import Image
from pathlib import Path

# Disable SSL verification for model downloads
ssl._create_default_https_context = ssl._create_unverified_context

# BoxMOT imports
try:
    from boxmot import BoostTrack, OcSort, StrongSort, HybridSort, DeepOcSort, ByteTrack, BotSort
    BOXMOT_AVAILABLE = True
except ImportError:
    BOXMOT_AVAILABLE = False
    print("Warning: boxmot not available. BoxMOT trackers will be disabled.")

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

class SSDDetector:
    """SSD MobileNet V2 detector wrapper"""
    def __init__(self, model_path, config_path, conf=0.3, classes=None):
        self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        self.conf = conf
        self.classes = classes
        
    def detect(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.conf:
                class_id = int(detections[0, 0, i, 1])
                
                # Filter by class if specified (Person is class 1 in COCO SSD)
                if self.classes is not None:
                    # Map SSD class IDs (1-indexed) to potential user requested IDs
                    # Assuming user passes 0 for person, but SSD uses 1
                    if class_id != 1: # Only support person for now to simplify
                        continue
                    
                if class_id == 1:  # Person
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    if x2 > x1 and y2 > y1:
                        # [x1, y1, x2, y2]
                        results.append([x1, y1, x2, y2, confidence, 0])
        
        if not results:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))
            
        results = np.array(results)
        boxes = results[:, :4]
        confs = results[:, 4]
        labels = results[:, 5]
        
        return boxes, confs, labels

def create_tracker(tracker_type, reid_model=None, device='cpu'):
    """Create a BoxMOT tracker"""
    if not BOXMOT_AVAILABLE:
        raise ValueError("BoxMOT not available. Install with: pip install boxmot")
    
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

def benchmark_ssd_boxmot(video_path, model_path, config_path, tracker_type, reid_model, conf=0.5, 
                          classes=None, max_frames=None, show_viz=False, device='cpu'):
    """Benchmark SSD detector with BoxMOT tracker"""
    detector = SSDDetector(model_path, config_path, conf=conf, classes=classes)
    tracker = create_tracker(tracker_type, reid_model, device=device)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    result = BenchmarkResult(
        f"SSD+{tracker_type.upper()}",
        "SSD",
        tracker_type
    )
    
    # Warmup
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            boxes, confs, labels = detector.detect(frame)
            if len(boxes) > 0:
                detections = np.concatenate([boxes, confs[:, None], labels[:, None]], axis=1)
                tracker.update(detections, frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    t_start = time.time()
    
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
    
    result.total_elapsed = time.time() - t_start
    cap.release()
    return result

def generate_all_boxmot_combinations(reid_model='osnet_x0_25_market1501.pt'):
    """Generate all SSD + BoxMOT tracker combinations"""
    boxmot_trackers = [
        'boosttrack', 'ocsort', 'strongsort', 'hybridsort', 
        'deepocsort', 'bytetrack', 'botsort'
    ]
    
    combinations = []
    for tracker in boxmot_trackers:
        combinations.append({
            'type': 'ssd_boxmot',
            'tracker': tracker,
            'reid': reid_model
        })
    
    return combinations

def parse_combinations(combos_str):
    """Parse combination strings like 'ssd:botsort:osnet_x0_25_msmt17.pt'"""
    combinations = []
    for combo in combos_str:
        parts = combo.split(':')
        if len(parts) < 3:
            raise ValueError(f"Invalid combination format: {combo}. Use format: 'ssd:tracker:reid'")
        
        combo_type = parts[0].lower()
        
        if combo_type == 'ssd':
            combinations.append({
                'type': 'ssd_boxmot',
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
    
    print(f"\n{'Algorithm':<40} {'Mean (ms)':<15} {'FPS':<15}")
    print("-" * 70)
    
    for stats in stats_list:
        print(f"{stats['name']:<40} {stats['mean_ms']:>14.2f} {stats['fps']:>14.2f}")
    
    print("="*80)
    
    if csv_output:
        csv_path = Path(csv_output)
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['detector', 'tracker', 'mean_inference_time_ms', 'fps'])
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
        description='SSD MobileNet V2 + BoxMOT Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all BoxMOT combinations (all 7 trackers with SSD)
  python test_ssd.py --video testing_vid5.mp4 --all-boxmot
  
  # Custom combinations
  python test_ssd.py --video testing_vid5.mp4 --combos ssd:botsort:osnet_x0_25_market1501.pt
  
Combination formats:
  - ssd:TRACKER:REID                (SSD + BoxMOT tracker)
        """
    )
    
    parser.add_argument('--video', type=str, required=True,
                       help='Video file path')
    parser.add_argument('--combos', type=str, nargs='+', default=None,
                       help='Algorithm combinations (see examples)')
    parser.add_argument('--all-boxmot', action='store_true',
                       help='Run all BoxMOT trackers with SSD detector')
    parser.add_argument('--reid-model', type=str, default='osnet_x0_25_market1501.pt',
                       help='ReID model path for BoxMOT trackers')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Detection confidence threshold')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Filter by class IDs')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--show-viz', action='store_true',
                       help='Show visualization windows')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--csv', type=str, default=None,
                       help='Export results to CSV file')
    
    parser.add_argument('--ssd-model', type=str, default="ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
                        help="Path to SSD frozen inference graph")
    parser.add_argument('--ssd-config', type=str, default="ssd_mobilenet_v2_coco.pbtxt",
                        help="Path to SSD config file")
    
    args = parser.parse_args()
    
    if not args.combos and not args.all_boxmot:
        parser.error("Either --combos or --all-boxmot must be specified")
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    combinations = []
    
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
    
    print(f"\n{'='*80}")
    print(f"BENCHMARKING {len(combinations)} ALGORITHM COMBINATION(S)")
    print(f"{'='*80}")
    for i, combo in enumerate(combinations, 1):
        print(f"{i}. {combo['type']}: SSD + {combo['tracker']}")
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
    
    results = []
    for i, combo in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] Benchmarking: SSD + {combo['tracker']}")
        print("-" * 80)
        
        try:
            if combo['type'] == 'ssd_boxmot':
                if not BOXMOT_AVAILABLE:
                    print(f"  ⚠ Skipping {combo['type']}: BoxMOT not available")
                    continue
                result = benchmark_ssd_boxmot(
                    args.video, args.ssd_model, args.ssd_config, 
                    combo['tracker'], combo['reid'],
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
    
    if results:
        print_comparison(results, csv_output=args.csv)
    else:
        print("\nNo successful benchmarks to compare.")
    
    if args.show_viz:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
