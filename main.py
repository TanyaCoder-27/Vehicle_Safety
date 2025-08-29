'''
Working Uindirectional Vehicle Speed Detection with YOLOv8 and SORT


import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

# Load model (YOLOv8n is lightweight, use yolov8s/m/l if needed)
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("sample.mp4")
if not cap.isOpened():
    print("❌ Failed to open video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(f"Video FPS: {fps}, Resolution: {frame_width}x{frame_height}")

# Calibration: real-world distance between two lines (in meters)
METERS_BETWEEN_LINES = 10.0  # adjust this based on your video
ROC_LINE_Y1 = 230
ROC_LINE_Y2 = 360

# Initialize tracker
tracker = Sort(max_age=20, min_hits=3)

# Track vehicles and speed
entry_frames = {}     # track_id: (frame_num, y)
exit_frames = {}      # track_id: (frame_num, y)
final_speeds = {}     # track_id: (speed_kmph, class_id)

# Overspeed limits
overspeed_limits = {
    2: 100,   # car
    3: 80,    # motorcycle
    5: 90,    # bus
    7: 80     # truck
}

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_disp = frame.copy()
    tracked = []
    # Detection
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
    detections = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            detections.append([x1, y1, x2, y2, track_id, cls])

    # Draw ROC Zone
    cv2.line(frame_disp, (0, ROC_LINE_Y1), (frame.shape[1], ROC_LINE_Y1), (0, 0, 255), 2)
    cv2.line(frame_disp, (0, ROC_LINE_Y2), (frame.shape[1], ROC_LINE_Y2), (0, 0, 255), 2)
    cv2.putText(frame_disp, "Speed Detection Zone", (20, ROC_LINE_Y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Update tracker
    if detections:
        dets_array = np.array([d[:4] for d in detections])
        tracked = tracker.update(dets_array)

        for trk in tracked:
            x1, y1, x2, y2, track_id = map(int, trk)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Match class ID
            cls_id = None
            for d in detections:
                if d[4] == track_id:
                    cls_id = d[5]
                    break
            if cls_id is None:
                continue

            # Draw box and ID
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_disp, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Record entry into ROC
            if ROC_LINE_Y1 <= cy <= ROC_LINE_Y2:
                if track_id not in entry_frames:
                    entry_frames[track_id] = (frame_count, cy)

            # Record exit
            elif cy > ROC_LINE_Y2 and track_id in entry_frames and track_id not in final_speeds:
                enter_frame, enter_y = entry_frames[track_id]
                time_secs = (frame_count - enter_frame) / fps
                if time_secs > 0:
                    speed_mps = METERS_BETWEEN_LINES / time_secs
                    speed_kmph = speed_mps * 3.6
                    final_speeds[track_id] = (speed_kmph, cls_id)

            # Display speed if available
            if track_id in final_speeds:
                speed_kmph, cls_id = final_speeds[track_id]
                overspeed = speed_kmph > overspeed_limits.get(cls_id, 100)
                color = (0, 0, 255) if overspeed else (0, 255, 0)
                cv2.putText(frame_disp, f"{speed_kmph:.1f} km/h", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if overspeed:
                    cv2.putText(frame_disp, "OVERSPEED", (x1, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                # Optional: you can show "Detecting..." text
                cv2.putText(frame_disp, f"Detecting...", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Show stats
    cv2.putText(frame_disp, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_disp, f"Objects: {len(tracked)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Vehicle Tracking", frame_disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

#Working Bidirectional Vehicle detection and Vehicle Speed Detection with YOLOv8 and SORT for all kinds of vehicles from start to end of the video :)
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

# Load model
model = YOLO("yolov8s.pt")  # Change to 'yolov8s.pt' if running on lower specs

# Load video
cap = cv2.VideoCapture("sample.mp4")
if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video FPS: {fps}, Resolution: {frame_width}x{frame_height}")

# ROC lines
METERS_BETWEEN_LINES = 8  # adjust based on your actual setup
ROC_LINE_1 = int(0.4 * frame_height)
ROC_LINE_2 = int(0.6 * frame_height)

# Initialize tracker
tracker = Sort()

# Speed detection vars
entry_times = {}
speeds = {}

# Speed limit
SPEED_LIMIT = 90  # km/h

frame_count = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_disp = frame.copy()

    # Inference
    results = model(frame, verbose=False)[0]
    detections = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if int(class_id) in [2, 3, 5, 7]:  # vehicle classes
            detections.append([x1, y1, x2, y2, score])

    # Tracking
    tracked = tracker.update(np.array(detections))
    cv2.line(frame_disp, (0, ROC_LINE_1), (frame_width, ROC_LINE_1), (0, 0, 255), 2)
    cv2.line(frame_disp, (0, ROC_LINE_2), (frame_width, ROC_LINE_2), (0, 0, 255), 2)

    # Annotate frame count
    cv2.putText(frame_disp, f"Frame: {frame_count}", (10, 30), font, 0.9, (255, 255, 255), 2)
    cv2.putText(frame_disp, f"Objects: {len(tracked)}", (10, 70), font, 0.9, (255, 255, 255), 2)
    cv2.putText(frame_disp, "Speed Detection Zone", (50, ROC_LINE_2 + 40), font, 1.2, (0, 0, 255), 4)

    for track in tracked:
        x1, y1, x2, y2, obj_id = track.astype(int)
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        # Skip near edges
        if y_center < 40 or y_center > frame_height - 40:
            continue

        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_disp, f"ID:{int(obj_id)}", (x1, y1 - 10), font, 0.8, (0, 255, 255), 2)

        # Late entry fallback: if already in ROC zone
        if ROC_LINE_1 < y_center < ROC_LINE_2 and obj_id not in entry_times:
            entry_times[obj_id] = frame_count

        # Downward movement
        elif y_center >= ROC_LINE_2 and obj_id in entry_times and obj_id not in speeds:
            elapsed = frame_count - entry_times[obj_id]
            time_sec = elapsed / fps
            speed = (METERS_BETWEEN_LINES / time_sec) * 3.6  # m/s to km/h
            speeds[obj_id] = speed

        # Upward movement
        elif y_center <= ROC_LINE_1 and obj_id in entry_times and obj_id not in speeds:
            elapsed = frame_count - entry_times[obj_id]
            time_sec = elapsed / fps
            speed = (METERS_BETWEEN_LINES / time_sec) * 3.6
            speeds[obj_id] = speed

        # Show speed if calculated
        if obj_id in speeds:
            spd = round(speeds[obj_id], 1)
            color = (0, 255, 0) if spd <= SPEED_LIMIT else (0, 0, 255)
            cv2.putText(frame_disp, f"{spd} km/h", (x1, y2 + 25), font, 0.8, color, 2)
            if spd > SPEED_LIMIT:
                cv2.putText(frame_disp, "OVERSPEED", (x1, y2 + 50), font, 0.8, color, 2)
        else:
            cv2.putText(frame_disp, "Detecting...", (x1, y2 + 25), font, 0.7, (200, 200, 200), 2)

    cv2.imshow("Vehicle Tracking", frame_disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
