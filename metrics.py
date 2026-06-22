import time
from collections import deque

class MetricsTracker:
    def __init__(self):
        self.fps_buffer = deque(maxlen=30)
        self.frame_times = []
        self.start_time = None
        self.total_frames = 0

    def start_session(self):
        self.start_time = time.time()
        self.total_frames = 0
        self.fps_buffer.clear()
        self.frame_times.clear()

    def update(self, frame_time):
        self.frame_times.append(frame_time)
        self.fps_buffer.append(frame_time)
        self.total_frames += 1

    def get_fps(self):
        if self.fps_buffer:
            avg_time = sum(self.fps_buffer) / len(self.fps_buffer)
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0

    def get_avg_fps(self):
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0

    def get_total_time(self):
        if self.start_time:
            return time.time() - self.start_time
        return 0

    def get_processed_frames(self):
        return self.total_frames