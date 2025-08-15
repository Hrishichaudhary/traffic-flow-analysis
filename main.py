import cv2
import numpy as np
import pandas as pd
import os
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from types import SimpleNamespace

# =============================
# Configurable Constants
# =============================
COUNT_TOLERANCE = 20         # ± pixels for lanes 1 & 2
LANE3_TOLERANCE = 25         # ± pixels for lane 3
MIN_CONFIDENCE = 0.45

# ============================================
# Detector
# ============================================
class Detector:
    def __init__(self, model_name='yolov5s'):
        self.model = YOLO(model_name + '.pt')
        self.target_classes = [2, 3, 5, 7]  # Vehicle classes

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in self.target_classes and conf > MIN_CONFIDENCE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append([x1, y1, x2, y2, conf, cls])
        return detections

# ============================================
# Lane Assigner
# ============================================
class LaneAssigner:
    def __init__(self, lanes):
        self.lanes = lanes

    def get_lane(self, point):
        for idx, lane in enumerate(self.lanes):
            if cv2.pointPolygonTest(lane, point, False) >= 0:
                return idx + 1
        return None

# ============================================
# Draw lanes
# ============================================
def draw_lanes(frame, lanes):
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    for i, lane in enumerate(lanes):
        cv2.polylines(frame, [lane], isClosed=True, color=colors[i], thickness=3)
        M = cv2.moments(lane)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, f'Lane {i+1}', (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)

# ============================================
# Overlay counts
# ============================================
def overlay_info(frame, count_dict):
    y0, dy = 30, 30
    for i, (lane, count) in enumerate(sorted(count_dict.items())):
        cv2.putText(frame, f'Lane {lane}: {count}',
                    (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# ============================================
# Main
# ============================================
def main():
    frame_height = 1080  # Adjust to your video resolution

    lane1 = np.array([
        (142, 729), (314, 718), (511, 711), (711, 698),
        (711, frame_height), (142, frame_height)
    ], np.int32)

    lane2 = np.array([
        (790, 695), (880, 701), (978, 701), (1080, 700),
        (1080, frame_height), (790, frame_height)
    ], np.int32)

    vertical_shift = 100  # pixels to move down

    lane3 = np.array([
        (1350, 571 + vertical_shift),
        (1520, 612 + vertical_shift),
        (1700, 642 + vertical_shift),
        (1880, 669 + vertical_shift),
        (1880, frame_height),
        (1300, frame_height)
    ], np.int32)

    lanes = [lane1, lane2, lane3]

    # === MANUAL YELLOW LINE POSITIONS ===
    count_lines_y = {
        1: 720,  # Lane 1
        2: 700,  # Lane 2
        3: 860   # Lane 3
    }

    detector = Detector()
    assigner = LaneAssigner(lanes)

    args = SimpleNamespace(track_thresh=0.5, match_thresh=0.8, track_buffer=50, mot20=False)
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()

    video_path = 'traffic.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video frame size: {width}x{height}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, 'traffic_counting_output.mp4')
    out = cv2.VideoWriter(out_video_path, fourcc, 25.0, (width, height))

    vehicles = {}
    lane_counts = {1: 0, 2: 0, 3: 0}
    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_lanes(frame, lanes)

        # Draw yellow lines from the start
        for lane_id, y_line in count_lines_y.items():
            x_coords = [p[0] for p in lanes[lane_id - 1]]
            cv2.line(frame, (min(x_coords), y_line), (max(x_coords), y_line),
                     (0, 255, 255), 2)

        detections = detector.detect(frame)
        det_array = torch.tensor(detections, dtype=torch.float32).cpu() if detections else \
                    torch.empty((0, 6), dtype=torch.float32)

        timer.tic()
        online_targets = tracker.update(det_array, [height, width], (height, width))
        timer.toc()

        for t in online_targets:
            x1, y1, w, h = map(int, t.tlwh)
            x2, y2 = x1 + w, y1 + h

            lane_id = assigner.get_lane(((x1 + x2) // 2, y2))

            # ==== Extend bounding boxes upward (per lane) ====
            if lane_id == 3:        # Lane 3 gets more extension
                y1 = max(0, y1 - 40)
            elif lane_id in (1, 2): # Lane 1 & 2 get standard extension
                y1 = max(0, y1 - 25)
            # =================================================

            bottom_center = ((x1 + x2) // 2, y2)

            if t.track_id not in vehicles:
                vehicles[t.track_id] = {'lane': lane_id, 'counted': False}
            elif vehicles[t.track_id]['lane'] is None and lane_id:
                vehicles[t.track_id]['lane'] = lane_id

            lane_for_this = vehicles[t.track_id]['lane']
            if lane_for_this and not vehicles[t.track_id]['counted']:
                line_y = count_lines_y[lane_for_this]
                tol = LANE3_TOLERANCE if lane_for_this == 3 else COUNT_TOLERANCE
                if line_y - tol <= bottom_center[1] <= line_y + tol:
                    lane_counts[lane_for_this] += 1
                    vehicles[t.track_id]['counted'] = True
                    results.append({
                        'id': t.track_id,
                        'lane': lane_for_this,
                        'frame': frame_idx,
                        'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    })

            # Draw box + id
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0) if lane_for_this else (255, 255, 255), 2)
            cv2.putText(frame, f'ID{t.track_id} L{lane_for_this if lane_for_this else "?"}',
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if lane_for_this else (255, 255, 255), 2)
            cv2.circle(frame, bottom_center, 5, (0, 0, 255), -1)

        overlay_info(frame, lane_counts)

        cv2.imshow("Debug View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'counts_by_lane.csv'), index=False)
    print(f"Done! Output video: {out_video_path}, CSV saved")

if __name__ == "__main__":
    main()