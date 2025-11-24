# Multi-Algorithm Tracking Benchmark Tool

`test_yolo.py` is a comprehensive benchmarking tool that compares multiple detector+tracker combinations simultaneously. It supports YOLO models with Ultralytics trackers, YOLO models with BoxMOT trackers, and Faster R-CNN with BoxMOT trackers.

## 🎯 Current Testing Plan

### Ultralytics (YOLO + Ultralytics Trackers)
- **Detectors**: ALL YOLO models (~80+ variants)
- **Trackers**: ByteTrack (`bytetrack.yaml`) and BotSort (`botsort.yaml`)
- **Total**: ~160+ combinations

### BoxMOT (Faster R-CNN + BoxMOT Trackers)
- **Detector**: Faster R-CNN (`fasterrcnn_resnet50_fpn_v2`)
- **Trackers**: All 7 BoxMOT trackers
  - ByteTrack, OcSort (motion-only, no ReID)
  - BotSort, StrongSort, DeepOcSort, HybridSort, BoostTrack (with ReID)
- **ReID Model**: `osnet_x0_25_market1501.pt` (fastest)
- **Total**: 7 combinations

**Grand Total**: ~167+ combinations

---

## 📦 Installation

```bash
# Core dependencies
pip install ultralytics opencv-python torch torchvision numpy pillow

# BoxMOT (required for BoxMOT trackers)
pip install boxmot
```

---

## 🚀 Usage

### Basic Syntax

```bash
python test_yolo.py --video <video_path> --combos <combination1> [combination2] ...
```

### Quick Example

```bash
# Compare multiple algorithms
python test_yolo.py --video testing_vid5.mp4 \
  --combos yolo:yolov8n.pt:bytetrack.yaml \
           yolo:yolov8n.pt:botsort.yaml \
           rcnn:botsort:osnet_x0_25_market1501.pt \
  --csv results.csv
```

### Parallel Execution (All Combinations)

**Run all ~167+ combinations simultaneously**:

```bash
# Python script (recommended - controlled parallel execution)
python run_all_parallel.py

# Windows batch script (simple - opens many windows)
run_all_parallel.bat
```

**See `PARALLEL_EXECUTION_GUIDE.md` for details and resource management.**

---

## 📝 Combination Formats

### 1. YOLO + Ultralytics Tracker
```
yolo:MODEL:TRACKER.yaml
```
**Example**: `yolo:yolov8n.pt:bytetrack.yaml`

### 2. YOLO + BoxMOT Tracker
```
yolo_boxmot:MODEL:TRACKER:REID_MODEL
```
**Example**: `yolo_boxmot:yolov8n.pt:botsort:osnet_x0_25_market1501.pt`

### 3. Faster R-CNN + BoxMOT Tracker
```
rcnn:TRACKER:REID_MODEL
```
**Example**: `rcnn:botsort:osnet_x0_25_market1501.pt`

---

## ⚙️ Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--video` | Video file path | `testing_vid5.mp4` |
| `--combos` | Algorithm combinations | **Required** |
| `--conf` | Detection confidence threshold | `0.5` |
| `--classes` | Filter by class IDs | `None` |
| `--max-frames` | Limit frames for testing | `None` |
| `--show-viz` | Show visualization | `False` |
| `--device` | Device: `auto`, `cpu`, or `cuda` | `auto` |
| `--csv` | Export results to CSV | `None` |

---

## 📊 Output

The script outputs:
1. Video information (resolution, FPS, frames)
2. Real-time progress for each algorithm
3. Comparison table with Mean (ms) and FPS
4. CSV export (if `--csv` specified)

### CSV Columns
- `detector` - Detector type (YOLO or R-CNN)
- `tracker` - Tracker name
- `mean_inference_time_ms` - Average inference time
- `fps` - Frames per second

---

## 🔧 Troubleshooting

**BoxMOT not available**: `pip install boxmot`

**CUDA out of memory**: Use `--device cpu` or `--max-frames 100`

**ReID model not found**: Download from [BoxMOT releases](https://github.com/mikel-brostrom/yolov8_tracking/releases)

**No detections**: Lower `--conf` threshold (e.g., `--conf 0.3`)

---

## 📚 Additional Resources

- **Available Models & Trackers**: See `AVAILABLE_TRACKERS_DETECTORS.md`
- **Usage Guide**: See `BENCHMARK_USAGE.md`
- **Final Testing Plan**: See `FINAL_TESTING_PLAN.md`

---

**Last Updated**: 2024  
**Version**: 1.0
