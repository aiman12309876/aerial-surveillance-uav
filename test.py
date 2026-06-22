import unittest
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.detector import ObjectDetector

class TestDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = ObjectDetector()
    
    def test_preprocess(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.detector.preprocess(frame)
        self.assertEqual(result.shape, (100, 100, 3))
    
    def test_detect_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        annotated, detections = self.detector.detect_frame(frame)
        self.assertEqual(annotated.shape, (100, 100, 3))
        self.assertIsInstance(detections, list)

if __name__ == "__main__":
    unittest.main()