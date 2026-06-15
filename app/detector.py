from ultralytics import YOLO
import cv2

CLASS_NAMES = {0: 'smoke', 1: 'fire'}
COLORS = {
    'fire':  (30,  80, 255),
    'smoke': (160, 160, 160),
}
BOX_COLORS = {
    'fire':  (30,  80, 255),
    'smoke': (140, 140, 140),
}


class FireDetector:
    def __init__(self, weights=r'C:\prsnl data\vscode\projects\wildfire\weights\best_v4.pt', conf=0.45):
        self.model = YOLO(weights)
        self.conf = conf

    def predict(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            name = CLASS_NAMES.get(cls, 'unknown')
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = BOX_COLORS.get(name, (255, 255, 255))

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{name.upper()}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({'class': name, 'conf': round(conf, 3)})

        return frame, detections
