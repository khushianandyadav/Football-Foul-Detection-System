from turtle import mode

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import zipfile
import io
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch
from VARS_model.model import MVNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_mvfoul_model():
    model = MVNetwork(net_name="mvit_v2_s")
    ckpt = torch.load("14_model.pth.tar", map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model

mvfoul_model = load_mvfoul_model()

def preprocess_clip(frames):

    processed = []

    for frame in frames:
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
        processed.append(frame)

    clip = torch.stack(processed, dim=1)   # C,T,H,W
    clip = clip.unsqueeze(0)              # B,C,T,H,W
    clip = clip.unsqueeze(1).repeat(1,2,1,1,1,1)  # B,V,C,T,H,W

    #print("Clip tensor shape:", clip.shape)

    return clip.to(device)
# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(page_title="Soccer Foul Detection", page_icon="⚽", layout="wide")

# -----------------------------------------
# CUSTOM CSS (UI ONLY)
# -----------------------------------------
st.markdown("""
<style>

/* ===== MAIN BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    color: #1b5e20;
    font-family: "Segoe UI", sans-serif;
}

/* Remove top padding */
header {visibility: hidden;}
section.main > div {padding-top: 20px;}

/* ===== HERO ===== */
.hero {
    background: linear-gradient(135deg, #a5d6a7, #c8e6c9);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
}

.hero-title {
    font-size: 38px;
    font-weight: 900;
    color: #1b5e20;
}

.hero-sub {
    font-size: 16px;
    color: #2e7d32;
}

/* ===== SECTION CARD ===== */
.section-card {
    background: #ffffff;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    border: 1px solid #e0f2f1;
}

.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #2e7d32;
    margin-bottom: 15px;
}

/* ===== BUTTONS ===== */
.stButton button {
    background: linear-gradient(90deg, #43a047, #66bb6a);
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    border: none !important;
    font-size: 15px !important;
    transition: 0.2s ease;
}

.stButton button:hover {
    background: linear-gradient(90deg, #2e7d32, #43a047);
    transform: scale(1.05);
}

/* STOP BUTTON */
.stop-btn button {
    background: linear-gradient(90deg, #e53935, #c62828) !important;
    color: white !important;
}

/* ===== RADIO ===== */
div[role="radiogroup"] * {
    color: #1b5e20 !important;
    font-weight: 600 !important;
}

/* ===== DOWNLOAD BUTTON ===== */
.stDownloadButton button {
    background: linear-gradient(90deg, #29b6f6, #0288d1) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: #f9fbe7;
    border: 2px dashed #aed581;
    padding: 18px;
    border-radius: 12px;
}

/* uploader text */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderFileName"] {
    color: #1b5e20 !important;
    font-weight: 600 !important;
}

/* ===== TABLE ===== */
table {
    background: white;
    border-radius: 10px;
    overflow: hidden;
}

/* ===== IMAGES ===== */
img {
    border-radius: 12px !important;
}

/* ===== EXTRA TOUCH (subtle pitch feel) ===== */
.stApp::before {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background: repeating-linear-gradient(
        0deg,
        rgba(0, 100, 0, 0.03),
        rgba(0, 100, 0, 0.03) 2px,
        transparent 2px,
        transparent 60px
    );
    pointer-events: none;
}
            
hr { display: none !important; }

.block-container,
[data-testid="stVerticalBlock"] {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# HEADER
# -----------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title"> Multi-View Deep Learning System for Foul Detection in Football</div>
    <div class="hero-sub">
        Real-time tracking • Download Annotated Video • Snapshot extraction
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# LOAD MODEL

# -----------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
# -----------------------------------------
# LOAD SPORT CLASSIFIER (NEW)
# -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_sport_classifier():
    clf = models.mobilenet_v2(pretrained=False)
    clf.classifier[1] = nn.Linear(clf.last_channel, 2)
    clf.load_state_dict(torch.load("sport_classifier.pth", map_location=device))
    clf = clf.to(device)
    clf.eval()
    return clf

classifier = load_sport_classifier()

sport_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# -----------------------------------------
# SESSION STATE
# -----------------------------------------
if "run" not in st.session_state:
    st.session_state.run = False

if "live_file_ready" not in st.session_state:
    st.session_state.live_file_ready = False


# Snapshots state
if "foul_snapshots" not in st.session_state:
    st.session_state.foul_snapshots = []

if "foul_count" not in st.session_state:
    st.session_state.foul_count = 0

if "snapshot_dir" not in st.session_state:
    st.session_state.snapshot_dir = None

if "validated" not in st.session_state:
    st.session_state.validated = False

if "is_soccer" not in st.session_state:
    st.session_state.is_soccer = True

# -----------------------------------------
# UTILS
# -----------------------------------------

def reset_snapshots(mode_type):
    # Create snapshot directory
    snapshot_dir = f"{mode_type}_snapshots"

    if os.path.exists(snapshot_dir):
        for f in os.listdir(snapshot_dir):
            os.remove(os.path.join(snapshot_dir, f))
    else:
        os.makedirs(snapshot_dir)

    st.session_state.foul_snapshots = []
    st.session_state.foul_count = 0
    st.session_state.snapshot_dir = snapshot_dir

def center(box):
    x1, y1, x2, y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# -----------------------------------------
# SOCCER VIDEO VALIDATION (NEW)
# -----------------------------------------

def is_soccer_video(cap, sample_frames=6, threshold=0.6):

    soccer_votes = 0
    total = 0

    for _ in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = sport_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = classifier(img)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        if pred == 1:
            soccer_votes += 1

        total += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if total == 0:
        return False

    confidence = soccer_votes / total
    return confidence > threshold

def process_video(cap, save_path):

    width = int(cap.get(3))
    height = int(cap.get(4))

    # 🔥 FIXED OUTPUT FPS (important!)
    output_fps = 12   # you can change to 10 if you want slower

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, output_fps, (width, height))

    st.session_state.mv_buffer = []

    frame_placeholder = st.empty()

    window_size = 16
    threshold = 0.45   # much safer
    min_prob_gate = 0.40
    cooldown_frames = int(output_fps * 2)   # 3 seconds gap
    skip_rate = 1

    frame_idx = 0
    rolling_score = 0.0
    last_foul_state = False
    last_snapshot_frame_idx = -100000
    min_gap_frames = int(output_fps * 2)

    while cap.isOpened() and st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            break
        foul = False
        card = "No Card"
        frame_idx += 1

        # ---------------- YOLO TRACKING ----------------
        results = model.track(frame, persist=True, conf=0.5, verbose=False)

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, id_, cls in zip(boxes, ids, classes):

                if cls == 0:
                    cv2.rectangle(frame, (box[0], box[1]),
                                  (box[2], box[3]), (0, 255, 0), 2)

                    cv2.putText(frame, f"ID {id_}",
                                (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2)

        # ---------------- MVFOUL ----------------
        # ---------------- MVFOUL ----------------
        st.session_state.mv_buffer.append(frame.copy())

        foul = False
        card = "No Card"

        if len(st.session_state.mv_buffer) >= window_size and frame_idx % 5 == 0:

            clip_frames = st.session_state.mv_buffer[-window_size:]
            clip_tensor = preprocess_clip(clip_frames)
            clip_tensor = clip_tensor.to(device)

            with torch.no_grad():
                output = mvfoul_model(clip_tensor)

            logits = output[0]
            probs = torch.softmax(logits, dim=0)
            max_prob = torch.max(probs[1:]).item()

            if max_prob > threshold:
                foul = True
                card = "Yellow Card"
        

        # ---------------- FOUL DISPLAY ----------------
        if foul:
            cv2.putText(frame, f"FOUL - {card}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

        # ---------------- SNAPSHOT ----------------
        if foul and not last_foul_state and \
        (frame_idx - last_snapshot_frame_idx >= cooldown_frames):

            if st.session_state.snapshot_dir:

                st.session_state.foul_count += 1
                snap_name = f"foul_{st.session_state.foul_count:03d}.jpg"
                snap_path = os.path.join(st.session_state.snapshot_dir, snap_name)

                cv2.imwrite(snap_path, frame)
                st.session_state.foul_snapshots.append(snap_path)
                last_snapshot_frame_idx = frame_idx

        last_foul_state = foul

        # ---------------- SHOW FRAME ----------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb)

        # 🔥 THIS WRITES EVERY FRAME (full video)
        out.write(frame)

        # 🔥 slight delay for smoother preview
        time.sleep(1 / output_fps)

        

    cap.release()
    out.release()

st.markdown("""
<div class="section-card">
<div class="section-title"> System Comparison</div>

<table style="width:100%; font-size:15px;">
<tr>
<th align="left">Feature</th>
<th align="center">This System</th>
<th align="center">Traditional Systems</th>
</tr>

<tr><td>Live Foul Detection</td><td align="center"> Yes</td><td align="center"> No</td></tr>
<tr><td>Real-time Player Tracking</td><td align="center"> YOLO Tracking</td><td align="center"> Offline</td></tr>
<tr><td>Automatic Snapshots</td><td align="center"> Yes</td><td align="center"> No</td></tr>
<tr><td>Instant Annotated Video</td><td align="center"> Yes</td><td align="center"> Post-Processed</td></tr>
<tr><td>Deployment Ready</td><td align="center"> Yes</td><td align="center"> Dataset Only</td></tr>
</table>

</div>
""", unsafe_allow_html=True)
# -----------------------------------------
# MAIN UI
# -----------------------------------------

st.markdown('<div class="section-title"> Detection Mode</div>', unsafe_allow_html=True)

mode = st.radio("Select Mode", ["Upload Video", "Live Camera"],
                horizontal=True,
                label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

# Start/Stop Buttons
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    st.markdown('<div class="start-btn">', unsafe_allow_html=True)
    if st.button("▶ Start Detection"):
        st.session_state.run = True
        st.session_state.live_file_ready = False
        # reset snapshots for a clean run
        reset_snapshots("upload" if mode == "Upload Video" else "live")
        st.session_state.validated = False
        st.session_state.is_soccer = True
    st.markdown('</div>', unsafe_allow_html=True)

with btn_col2:
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    if st.button("⏹ Stop Detection"):
        st.session_state.run = False
        st.session_state.live_file_ready = True
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------
# UPLOAD MODE
# -----------------------------------------
if mode == "Upload Video":
    st.markdown('<div class="card"><div class="section-title">📤 Upload Soccer Video</div></div>',
                unsafe_allow_html=True)

    file = st.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])

    if file and st.session_state.run:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        output_path = "annotated_upload.mp4"

        cap = cv2.VideoCapture(tfile.name)

        # Validate if this is a soccer video (NEW)
        if not st.session_state.validated:
            with st.spinner("🔎 Validating video... (soccer check)"):
                ok = is_soccer_video(cap)

            st.session_state.validated = True
            st.session_state.is_soccer = ok

            if not ok:
                st.error("❌ Invalid Video Uploaded: This system only supports soccer match videos.")
                st.session_state.run = False

        if st.session_state.is_soccer and st.session_state.run:
            # Validate live feed (NEW)
            if not st.session_state.validated:
                with st.spinner("🔎 Validating live feed... (soccer check)"):
                    ok = is_soccer_video(cap)
                st.session_state.validated = True
                st.session_state.is_soccer = ok

                if not ok:
                    st.error("❌ Invalid video: This system only supports soccer match videos.")
                    st.session_state.run = False
                    cap.release()

            if st.session_state.is_soccer and st.session_state.run:
                process_video(cap, output_path)

        if st.session_state.is_soccer:
            st.success("✅ Upload video processing complete!")

        if st.session_state.is_soccer and os.path.exists(output_path):
            with open(output_path, "rb") as f:
                st.download_button("⬇ Download Annotated Video", f, file_name="upload_result.mp4")

        # Show foul snapshots (added feature)
        if st.session_state.foul_snapshots:
            st.markdown('<div class="card"><div class="section-title">📸 Detected Foul Snapshots</div></div>',
                        unsafe_allow_html=True)

            cols = st.columns(4)
            for idx, snap in enumerate(st.session_state.foul_snapshots):
                with cols[idx % 4]:
                    st.image(snap, caption=f"Foul #{idx+1}", use_container_width=True)

            # Zip snapshots in-memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for img_path in st.session_state.foul_snapshots:
                    if os.path.exists(img_path):
                        zipf.write(img_path, arcname=os.path.basename(img_path))
            zip_buffer.seek(0)

            st.download_button(
                "⬇ Download All Foul Snapshots (ZIP)",
                data=zip_buffer,
                file_name="upload_foul_snapshots.zip",
                mime="application/zip"
            )

# -----------------------------------------
# LIVE MODE
# -----------------------------------------
if mode == "Live Camera":
    st.markdown('<div class="card"><div class="section-title">📷 Live Foul Detection</div></div>',
                unsafe_allow_html=True)

    output_path = "annotated_live.mp4"

    if st.session_state.run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            st.error("❌ Camera not accessible")
        else:
            # Validate live feed (NEW)
            if not st.session_state.validated:
                with st.spinner("🔎 Validating live feed... (soccer check)"):
                    ok= is_soccer_video(cap, sample_frames=6, threshold=0.6)

                st.session_state.validated = True
                st.session_state.is_soccer = ok

                if not ok:
                    st.error("❌ Invalid video: This system only supports soccer match videos.")
                    st.session_state.run = False
                    cap.release()

            if st.session_state.is_soccer and st.session_state.run:
                process_video(cap, output_path)

    # SHOW DOWNLOAD AFTER STOP
    if st.session_state.live_file_ready and os.path.exists(output_path):
        st.success("✅ Live recording saved successfully!")
        with open(output_path, "rb") as f:
            st.download_button("⬇ Download Live Recording", f, file_name="live_result.mp4")

        # Show foul snapshots (added feature)
        if st.session_state.foul_snapshots:
            st.markdown('<div class="card"><div class="section-title">📸 Detected Foul Snapshots</div></div>',
                        unsafe_allow_html=True)

            cols = st.columns(4)
            for idx, snap in enumerate(st.session_state.foul_snapshots):
                with cols[idx % 4]:
                    st.image(snap, caption=f"Foul #{idx+1}", use_container_width=True)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for img_path in st.session_state.foul_snapshots:
                    if os.path.exists(img_path):
                        zipf.write(img_path, arcname=os.path.basename(img_path))
            zip_buffer.seek(0)

            st.download_button(
                "⬇ Download All Foul Snapshots (ZIP)",
                data=zip_buffer,
                file_name="live_foul_snapshots.zip",
                mime="application/zip"
            )


