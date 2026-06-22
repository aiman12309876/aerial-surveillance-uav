import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.colors = {
            'person': (0, 255, 0),
            'car': (255, 0, 0),
            'truck': (0, 0, 255),
            'default': (0, 255, 255)
        }

    def draw_bbox(self, frame, bbox, label, conf, color=None):
        x1, y1, x2, y2 = bbox
        if color is None:
            color = self.colors.get(label, self.colors['default'])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_metrics(self, frame, frame_count, fps):
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def add_heatmap(self, frame, detections):
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(heatmap, (cx, cy), 50, 1, -1)
        
        heatmap = cv2.GaussianBlur(heatmap, (101, 101), 30)
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    def draw_grid(self, frame, grid_size=50):
        h, w = frame.shape[:2]
        for x in range(0, w, grid_size):
            cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
        for y in range(0, h, grid_size):
            cv2.line(frame, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_crosshair(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        cv2.line(frame, (cx - 30, cy), (cx - 10, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx + 10, cy), (cx + 30, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 30), (cx, cy - 10), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy + 10), (cx, cy + 30), (0, 255, 0), 1)
        
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), 1)
        
        return frame

    def draw_corner_ticks(self, frame, length=30):
        h, w = frame.shape[:2]
        color = (0, 255, 0)
        
        cv2.line(frame, (0, 0), (length, 0), color, 2)
        cv2.line(frame, (0, 0), (0, length), color, 2)
        cv2.line(frame, (w, 0), (w - length, 0), color, 2)
        cv2.line(frame, (w, 0), (w, length), color, 2)
        cv2.line(frame, (0, h), (length, h), color, 2)
        cv2.line(frame, (0, h), (0, h - length), color, 2)
        cv2.line(frame, (w, h), (w - length, h), color, 2)
        cv2.line(frame, (w, h), (w, h - length), color, 2)
        
        return frame