import streamlit as st
import cv2
import os
import time
import pandas as pd
import plotly.express as px
import numpy as np
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.detector import ObjectDetector
from utils.visualizer import Visualizer
from utils.metrics import MetricsTracker

st.set_page_config(
    page_title="Aerial Surveillance System",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0a0a0a; }
    .stButton>button { background-color: #00ff88; color: #000; font-weight: bold; }
    .css-1aumxhk { background-color: #1a1a2e; }
    .metric-card { background-color: #1a1a2e; padding: 20px; border-radius: 10px; text-align: center; }
    .st-emotion-cache-1r6slb0 { background-color: #1a1a2e; }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Aerial Surveillance & Object Detection")
st.markdown("### National Security Drone Monitoring System")

if 'detector' not in st.session_state:
    st.session_state.detector = ObjectDetector()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'metrics' not in st.session_state:
    st.session_state.metrics = MetricsTracker()
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'processed' not in st.session_state:
    st.session_state.processed = False

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🛸 Active Drones", "2")
with col2:
    st.metric("🎯 Objects Detected", len(st.session_state.detections))
with col3:
    st.metric("⚠️ High Threats", "0")
with col4:
    st.metric("🟢 System Status", "Online")

tab1, tab2, tab3 = st.tabs(["📡 Live Detection", "📊 Analytics", "⚙️ Settings"])

with tab1:
    uploaded_file = st.file_uploader("Upload Drone Footage", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file:
        video_path = f"videos/{uploaded_file.name}"
        os.makedirs("videos", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        col1, col2 = st.columns(2)
        with col1:
            st.video(video_path)
        
        with col2:
            st.subheader("Detection Controls")
            conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3)
            use_clahe = st.checkbox("Enable CLAHE Preprocessing", value=True)
            use_heatmap = st.checkbox("Enable Heatmap Overlay", value=False)
            
            if st.button("🔍 Run Detection", use_container_width=True):
                with st.spinner("Processing drone footage..."):
                    detector = st.session_state.detector
                    visualizer = st.session_state.visualizer
                    metrics = st.session_state.metrics
                    
                    output_path = f"outputs/detected_{uploaded_file.name}"
                    os.makedirs("outputs", exist_ok=True)
                    
                    cap = cv2.VideoCapture(video_path)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    st.session_state.detections = []
                    frame_count = 0
                    progress_bar = st.progress(0)
                    fps_display = st.empty()
                    
                    metrics.start_session()
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        start_time = time.time()
                        
                        processed_frame = detector.preprocess(frame, use_clahe)
                        annotated, dets = detector.detect_frame(processed_frame, conf_threshold)
                        
                        if use_heatmap:
                            annotated = visualizer.add_heatmap(annotated, dets)
                        
                        annotated = visualizer.draw_metrics(annotated, frame_count, fps)
                        annotated = visualizer.draw_crosshair(annotated)
                        annotated = visualizer.draw_corner_ticks(annotated)
                        
                        out.write(annotated)
                        
                        for d in dets:
                            d['frame'] = frame_count
                            st.session_state.detections.append(d)
                        
                        frame_count += 1
                        metrics.update(time.time() - start_time)
                        
                        progress_bar.progress((frame_count + 1) / total_frames)
                        fps_display.text(f"FPS: {metrics.get_fps():.2f}")
                    
                    cap.release()
                    out.release()
                    st.session_state.processed = True
                    
                    avg_fps = metrics.get_avg_fps()
                    total_detections = len(st.session_state.detections)
                
                st.success(f"✅ Detection Complete! Processed {frame_count} frames, {total_detections} objects detected")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.video(output_path)
                with col2:
                    st.metric("📊 Average FPS", f"{avg_fps:.2f}")
                    st.metric("📋 Total Objects", total_detections)
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        "📥 Download Processed Video",
                        f,
                        "detected_footage.mp4",
                        "video/mp4"
                    )
    
    if st.session_state.detections:
        st.subheader("📋 Detection Log")
        df = pd.DataFrame(st.session_state.detections)
        if 'label' in df.columns:
            st.dataframe(df[['label', 'confidence', 'frame']], use_container_width=True)

with tab2:
    st.subheader("📊 Detection Analytics")
    if st.session_state.detections:
        df = pd.DataFrame(st.session_state.detections)
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(df, names='label', title='Object Distribution', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.histogram(df, x='confidence', title='Confidence Distribution', nbins=20, color_discrete_sequence=['#00ff88'])
            st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = px.bar(df, x='label', title='Detection Count by Class', color_discrete_sequence=['#00b4ff'])
        st.plotly_chart(fig3, use_container_width=True)
        
        if 'frame' in df.columns:
            detections_over_time = df.groupby('frame').size().reset_index(name='count')
            fig4 = px.line(detections_over_time, x='frame', y='count', title='Detections Over Time')
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Run detection first to see analytics")

with tab3:
    st.subheader("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        model_option = st.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        if st.button("🔄 Load Model"):
            st.session_state.detector = ObjectDetector(model_option)
            st.success(f"Model {model_option} loaded!")
    
    with col2:
        iou_threshold = st.slider("IoU Threshold", 0.1, 0.9, 0.5)
        max_detections = st.slider("Max Detections", 50, 500, 300)
    
    if st.button("🔄 Reset All"):
        st.session_state.detector = ObjectDetector()
        st.session_state.detections = []
        st.session_state.processed = False
        st.session_state.metrics = MetricsTracker()
        st.success("System reset successfully!")

st.markdown("---")
st.caption("🛡️ NASTP AI Internship Project | Aerial Surveillance System | Datasets: VisDrone, DOTA, UAVDT")