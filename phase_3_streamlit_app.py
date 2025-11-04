# ============================================================
# SmartVision AI - Final Professional Edition (use_container_width fix)
# ============================================================
# Author: Jalloul Joudi
# Description: Full production-ready version with all features integrated.
# Fixes:
# - Replaced deprecated use_column_width with use_container_width
# - Removed model.pt access
# ============================================================

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import tempfile
import time
import os
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# UI STYLE (White + Soft Blue)
# ============================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {
    background-color: #ffffff;
    padding-top: 18px;
    padding-bottom: 24px;
}
[data-testid="stSidebar"] {
    background-color: #f7fbff;
}
.custom-header {
    background: linear-gradient(90deg, rgba(255,255,255,0.9), rgba(247,250,255,0.95));
    padding: 22px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(20,50,100,0.06);
    margin-bottom: 20px;
}
.custom-header h1 {
    margin: 0;
    font-size: 30px;
    color: #0b57a4;
    font-weight: 800;
}
.custom-header p {
    margin: 6px 0 0 0;
    color: #486e9e;
}
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 6px 18px rgba(20,50,100,0.04);
    margin-bottom: 12px;
}
.metric-card {
    border-radius: 12px;
    padding: 14px;
    background: linear-gradient(180deg, #f3f9ff 0%, #ffffff 100%);
    box-shadow: 0 6px 18px rgba(10,60,110,0.04);
    text-align: center;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #0b57a4;
}
.metric-label {
    font-size: 12px;
    color: #6b8ab6;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.alert-danger {
    background: #ffdcdc;
    border-left: 4px solid #ff6b6b;
    color: #6b2323;
    padding: 10px;
    border-radius: 8px;
}
.alert-warning {
    background: #fff7df;
    border-left: 4px solid #f59f00;
    color: #6b4b00;
    padding: 10px;
    border-radius: 8px;
}
.alert-info {
    background: #e8fbff;
    border-left: 4px solid #2aa8d6;
    color: #074a5c;
    padding: 10px;
    border-radius: 8px;
}
.alert-success {
    background: #e8fff0;
    border-left: 4px solid #2fa84f;
    color: #1b5b30;
    padding: 10px;
    border-radius: 8px;
}
.debug-box {
    background: #0f1724;
    color: #9fffb6;
    font-family: monospace;
    padding: 10px;
    border-radius: 8px;
    font-size: 13px;
}
.stButton>button {
    background: linear-gradient(90deg, #0b57a4, #1b82d6);
    color: white;
    border-radius: 8px;
    padding: 8px 18px;
    border: none;
    box-shadow: 0 6px 18px rgba(27,130,214,0.12);
}
.stButton>button:hover {
    transform: translateY(-1px);
}
small {
    color: #6b8ab6;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL CONFIG
# ============================================================
BASE_PROJECT_PATH = Path("./SmartVision")
CUSTOM_MODEL_PATH = BASE_PROJECT_PATH / "phase_2_yolo_training" / "smartvision_detector9" / "weights" / "best.pt"
USE_PRETRAINED = True  # set False to use custom model

@st.cache_resource
def load_model():
    try:
        if not USE_PRETRAINED and CUSTOM_MODEL_PATH.exists():
            return YOLO(str(CUSTOM_MODEL_PATH))
        else:
            return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# ============================================================
# DETECTION HELPERS
# ============================================================
def run_detection(frame_or_pil, model, conf_threshold=0.25):
    if isinstance(frame_or_pil, np.ndarray):
        img = cv2.cvtColor(frame_or_pil, cv2.COLOR_BGR2RGB)
    elif isinstance(frame_or_pil, Image.Image):
        img = np.array(frame_or_pil.convert("RGB"))
    else:
        raise ValueError("Unsupported input type")
    return model.predict(source=img, conf=conf_threshold, verbose=False)[0]

def draw_detections_on_frame(frame_bgr, results):
    annotated = frame_bgr.copy()
    color_map = {
        'traffic light': (0, 191, 255),
        'car': (255, 255, 0),
        'truck': (255, 0, 255),
        'bus': (0, 128, 255),
        'bicycle': (0, 255, 0),
        'motorcycle': (0, 255, 128),
        'person': (0, 0, 255),
        'stop sign': (255, 0, 0),
    }
    for box in results.boxes:
        try:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cname = results.names[cls_id].lower()
            color = color_map.get(cname, (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{cname.upper()} {conf:.1%}"
            cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)
        except Exception:
            continue
    return annotated

def get_detection_stats(results):
    counts = {}
    dets = []
    for box in results.boxes:
        cid = int(box.cls[0])
        cname = results.names[cid]
        conf = float(box.conf[0])
        counts[cname] = counts.get(cname, 0) + 1
        dets.append({'class': cname, 'confidence': conf, 'class_id': cid})
    return counts, dets

def get_alerts_from_results(results):
    conf_thresholds = {
        'traffic light': 0.3, 'person': 0.4, 'car': 0.3,
        'truck': 0.3, 'bus': 0.3, 'bicycle': 0.3,
        'motorcycle': 0.3, 'stop sign': 0.4
    }
    alerts = []
    for box in results.boxes:
        conf = float(box.conf[0])
        cid = int(box.cls[0])
        cname = results.names[cid].lower()
        if cname in conf_thresholds and conf > conf_thresholds[cname]:
            if cname == 'person':
                alerts.append(('🚶 Pedestrian', f'{conf:.1%}', 'danger'))
            elif cname == 'stop sign':
                alerts.append(('🛑 STOP Sign', f'{conf:.1%}', 'danger'))
            elif cname == 'traffic light':
                alerts.append(('🚦 Traffic Light', f'{conf:.1%}', 'warning'))
            else:
                alerts.append((f'🚗 {cname.title()}', f'{conf:.1%}', 'info'))
    return alerts

# ============================================================
# UI HEADER + SIDEBAR
# ============================================================
model = load_model()
if model is None:
    st.error("Failed to load YOLO model.")
    st.stop()

model_info = "YOLOv8n (pretrained)" if USE_PRETRAINED else f"Custom: {CUSTOM_MODEL_PATH.name}"

st.markdown("""
<div class="custom-header">
  <h1>🚗 SmartVision AI</h1>
  <p>Real-Time Driver Assistance System — Clean White + Soft Blue UI</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    mode = st.radio("Mode", ["📤 Upload Image", "📹 Upload Video", "🎥 Live Webcam"])
    conf_thr = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
    show_stats = st.checkbox("Show Statistics", True)
    debug = st.checkbox("Debug Mode", False)
    timestamps = st.checkbox("Show Timestamps", False)
    st.markdown("---")
    st.markdown(f"**Model:** {model_info}")
    st.markdown("---")
    st.markdown("SmartVision © 2025 — Designed by Jalloul Joudi")

# ============================================================
# MODE: IMAGE
# ============================================================
if mode == "📤 Upload Image":
    st.markdown("## 📤 Upload Image for Analysis")
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp"])
    if file:
        # save upload to avoid transient streamlit media id problems
        os.makedirs("uploads", exist_ok=True)
        fp = os.path.join("uploads", file.name)
        with open(fp, "wb") as f:
            f.write(file.read())

        pil_img = Image.open(fp).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-card'><div class='metric-value'>{img_bgr.shape[1]}</div><div class='metric-label'>Width</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><div class='metric-value'>{img_bgr.shape[0]}</div><div class='metric-label'>Height</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:20px;'>{file.name}</div><div class='metric-label'>File</div></div>", unsafe_allow_html=True)

        # show original image
        col_show_orig, _ = st.columns([1,2])
        col_show_orig.image(pil_img, caption="Original image", use_container_width=True)

        if st.button("🔍 Run Detection"):
            with st.spinner("Analyzing..."):
                res = run_detection(pil_img, model, conf_threshold=conf_thr)
                annotated = draw_detections_on_frame(img_bgr, res)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            left, right = st.columns([2,1])
            left.image(annotated_rgb, use_container_width=True, caption="Detection Result")
            cls_counts, dets = get_detection_stats(res)
            alerts = get_alerts_from_results(res)
            with right:
                if show_stats and cls_counts:
                    st.markdown(f"<div class='card'><strong>Total:</strong> {sum(cls_counts.values())}</div>", unsafe_allow_html=True)
                    for cname, cnt in sorted(cls_counts.items(), key=lambda x: x[1], reverse=True):
                        st.markdown(f"<div class='card'><strong>{cname}</strong> <span style='float:right;color:#0b57a4;'>{cnt}</span></div>", unsafe_allow_html=True)
                elif not cls_counts:
                    st.markdown("<div class='alert-warning'>No detections.</div>", unsafe_allow_html=True)
                st.markdown("#### 🚨 Alerts")
                if alerts:
                    for name, conf, sev in alerts:
                        style = {'danger': 'alert-danger', 'warning': 'alert-warning', 'info': 'alert-info'}[sev]
                        st.markdown(f"<div class='{style}'>{name} — {conf}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='alert-success'>✅ All Clear</div>", unsafe_allow_html=True)
                if debug:
                    lines = []
                    for i,d in enumerate(dets,1):
                        line = f"{i}. [{d['class_id']}] {d['class']} -> {d['confidence']:.2%}"
                        if timestamps: line = f"{datetime.now().isoformat()} | {line}"
                        lines.append(line)
                    st.markdown("<div class='debug-box'>{}</div>".format("<br>".join(lines)), unsafe_allow_html=True)

# ============================================================
# MODE: VIDEO
# ============================================================
elif mode == "📹 Upload Video":
    st.markdown("## 📹 Upload Video for Processing")
    vid = st.file_uploader("Upload MP4/MOV/AVI", type=["mp4","mov","avi","mkv"])
    if vid:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(vid.read())
        cap = cv2.VideoCapture(temp.name)
        frame_disp = st.empty()
        start = st.button("▶️ Start")
        stop = st.button("⏹️ Stop")
        if "run_video" not in st.session_state: st.session_state.run_video = False
        if start: st.session_state.run_video = True
        if stop: st.session_state.run_video = False
        while cap.isOpened() and st.session_state.run_video:
            ret, frame = cap.read()
            if not ret: break
            res = run_detection(frame, model, conf_threshold=conf_thr)
            annotated = draw_detections_on_frame(frame, res)
            frame_disp.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            # small throttle
            time.sleep(0.03)
        cap.release()
        st.session_state.run_video = False
        st.success("✅ Video processing completed")

# ============================================================
# MODE: WEBCAM
# ============================================================
elif mode == "🎥 Live Webcam":
    st.markdown("## 🎥 Live Webcam Detection")
    start = st.button("▶️ Start Webcam")
    stop = st.button("⏹️ Stop Webcam")
    if "run_webcam" not in st.session_state: st.session_state.run_webcam = False
    if start: st.session_state.run_webcam = True
    if stop: st.session_state.run_webcam = False
    frame_disp = st.empty()
    cap = None
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Webcam not accessible")
            st.session_state.run_webcam = False
        else:
            try:
                while st.session_state.run_webcam:
                    ret, frame = cap.read()
                    if not ret: break
                    res = run_detection(frame, model, conf_threshold=conf_thr)
                    annotated = draw_detections_on_frame(frame, res)
                    frame_disp.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                    time.sleep(0.03)
            except Exception as e:
                st.error(f"Webcam error: {e}")
            finally:
                cap.release()
                st.session_state.run_webcam = False
                st.success("✅ Webcam stopped")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("<div style='text-align:center;color:#6b8ab6;font-size:13px;'>SmartVision AI • YOLOv8 + Streamlit • Clean White & Soft Blue UI — © 2025</div>", unsafe_allow_html=True)
