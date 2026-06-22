import cv2
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.classes = self.model.names
        self.detections = []

    def preprocess(self, frame, use_clahe=True):
        if use_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame

    def detect_frame(self, frame, conf_threshold=0.3):
        results = self.model(frame, conf=conf_threshold)
        annotated = results[0].plot()
        
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.classes[cls]
            detections.append({
                'label': label,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
        
        return annotated, detections

    def process_video(self, video_path, output_path, conf=0.3, use_clahe=True):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.detections = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = self.preprocess(frame, use_clahe)
            annotated, dets = self.detect_frame(processed, conf)
            out.write(annotated)
            
            for d in dets:
                d['frame'] = frame_count
                self.detections.append(d)
            
            frame_count += 1

        cap.release()
        out.release()
        return output_path, self.detections