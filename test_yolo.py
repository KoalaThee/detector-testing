from ultralytics import YOLO
import cv2
import time

# 1. Load model
model = YOLO("yolov8s.pt")  # swap with yolov8n.pt, yolov8m.pt, yolo11s.pt, etc.

# 2. Open video file
video_path = "testing_vid5.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Error: Cannot open video file '{video_path}'")

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {video_path}")
print(f"Resolution: {actual_width}x{actual_height}")
print(f"Video FPS: {video_fps:.2f}")
print(f"Total frames: {total_frames}")

# 3. Warmup
print("Warming up...")
for i in range(10):
    ret, frame = cap.read()
    if ret:
        results = model(frame, verbose=False)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset if video ends

# Reset video to start for actual benchmark
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 4. Benchmark loop
inference_times = []  # Store inference times for statistics
frame_count = 0
t_start = time.time()

print("\nStarting benchmark...")
print("Press ESC to stop early\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Measure ONLY inference time (excludes I/O)
    t0 = time.time()
    results = model(frame, verbose=False)
    t1 = time.time()

    infer_ms = (t1 - t0) * 1000.0
    inference_times.append(infer_ms)
    frame_count += 1

    # Calculate FPS from inference time only (pure inference FPS)
    avg_inference_ms = sum(inference_times) / len(inference_times)
    inference_fps = 1000.0 / avg_inference_ms if avg_inference_ms > 0 else 0

    # Visualization
    annotated = results[0].plot()
    cv2.putText(annotated, f"Infer: {infer_ms:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Avg Infer: {avg_inference_ms:.1f} ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"FPS: {inference_fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLO Orin Nano", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# 5. Print summary statistics
if inference_times:
    import statistics
    t_end = time.time()
    total_elapsed = t_end - t_start
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"\n--- Inference Time Statistics (ms) ---")
    print(f"Min:    {min(inference_times):.2f} ms")
    print(f"Max:    {max(inference_times):.2f} ms")
    print(f"Mean:   {statistics.mean(inference_times):.2f} ms")
    print(f"Median: {statistics.median(inference_times):.2f} ms")
    if len(inference_times) > 1:
        print(f"StdDev: {statistics.stdev(inference_times):.2f} ms")
    print(f"\n--- FPS Statistics ---")
    print(f"FPS: {inference_fps:.2f}")
    print(f"Average inference time: {avg_inference_ms:.2f} ms")
    print("="*60)
