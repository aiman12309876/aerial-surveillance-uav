markdown
# 🛰️ Aerial Surveillance and Object Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Overview

A real-time aerial surveillance system using **YOLOv8** for drone-based object detection with a professional **Streamlit** dashboard. Designed for national security, surveillance, and defense applications.

---

## 🎯 How It Works

1. **Upload** a drone video (MP4, AVI, MOV, MKV)
2. **Adjust** detection settings (confidence threshold, CLAHE preprocessing)
3. **Click** "Run Detection"
4. **View** real-time object detection with bounding boxes
5. **Download** the processed video

---

## 🚀 Features

- ✅ Real-time object detection with YOLOv8
- ✅ Interactive Streamlit dashboard
- ✅ Drone video processing
- ✅ Heatmap visualization
- ✅ Detection analytics
- ✅ Military HUD-style UI
- ✅ Download processed videos
- ✅ CLAHE preprocessing for aerial imagery

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|------------|
| AI Model | YOLOv8 (Ultralytics) |
| Frontend | Streamlit |
| Data Processing | OpenCV, NumPy |
| Visualization | Plotly, Pandas |

---

## 📁 Project Structure
aerial-surveillance-uav/
├── app.py # Main Streamlit dashboard
├── detector.py # YOLOv8 detection logic
├── visualizer.py # Visualization tools
├── metrics.py # FPS and performance tracking
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── dashboard.png # Dashboard screenshot
├── detection_results.png # Detection output screenshot
├── detection_log.png # Detection log screenshot
└── test.py # Test script

text

---

## ⚙️ Installation

```bash
git clone https://github.com/aiman12309876/aerial-surveillance-uav.git
cd aerial-surveillance-uav
pip install -r requirements.txt
🚀 Running the Application
bash
streamlit run app.py
📊 Sample Output
After running detection on a drone video, the system displays:

Bounding boxes around detected objects

Confidence scores for each detection

Detection log with timestamps

Analytics dashboard with charts

Objects Detected Include:
Person

Car

Truck

Bicycle

And 80+ YOLO classes

📸 Screenshots
Dashboard
https://dashboard.png

Detection Results
https://detection_results.png

Detection Log
https://detection_log.png

🔧 Dependencies
bash
streamlit
opencv-python
ultralytics
pillow
numpy
pandas
plotly
🚀 Future Improvements
Add real-time drone video streaming

Implement object tracking (Deep SORT)

Add thermal image support

Deploy on cloud (AWS/Azure)

Mobile app integration

📝 License
MIT License

👤 Author
Aiman Zahoor

GitHub: aiman12309876

Email: aimanzahoor87@gmail.com

LinkedIn: linkedin.com/in/aiman-zahoor

⭐ Show Your Support
If you found this project useful, please give it a star on GitHub! ⭐

text

---

## ✅ How to Update:

1. Go to your repository:  
   `https://github.com/aiman12309876/aerial-surveillance-uav`

2. Click on **README.md**

3. Click the **pencil icon** (Edit)

4. Delete everything inside

5. Copy and paste the full README above

6. Click **"Commit changes"**
