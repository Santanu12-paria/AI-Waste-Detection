import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import os

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="AI Based Waste Detection",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .main-title {text-align: center; font-size: 42px; font-weight: 700; margin-bottom: 5px;}
    .subtitle {text-align: center; font-size: 18px; color: #666666; margin-bottom: 25px;}
    .footer {text-align: center; color: #777777; font-size: 14px; margin-top: 35px; padding: 15px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🗑️ AI Based Waste Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Waste Management using YOLOv8 Object Detection</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- MODEL ----------------
MODEL_PATH = os.path.join("models", "best.pt")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_model(path):
    return YOLO(path)


model = load_model(MODEL_PATH)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Project Information")
    st.write("### 🤖 Model")
    st.write("YOLOv8m (11 waste classes)")

    st.write("### 🎯 Detection Confidence")
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.90, 0.40, 0.05)

    st.write("### ♻️ Waste Classes")
    for name in model.names.values():
        st.write(f"• {name}")

    st.markdown("---")
    st.info("Upload a garbage image to detect and classify different types of waste.")

# ---------------- UPLOAD ----------------
st.subheader("📤 Upload Garbage Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG or PNG image."
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("---")

    with st.spinner("🔍 AI is detecting garbage..."):
        results = model.predict(
            source=image,
            conf=conf_threshold,
            imgsz=640,
            verbose=False
        )

    result = results[0]

    st.subheader("📷 Image Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### 🤖 Detection Result")
        annotated = result.plot()[..., ::-1]   # BGR -> RGB
        st.image(annotated, use_container_width=True)

    st.markdown("---")

    if result.boxes is not None and len(result.boxes) > 0:

        class_ids = result.boxes.cls.tolist()
        detected_names = [model.names[int(c)] for c in class_ids]
        confidences = result.boxes.conf.tolist()
        counts = Counter(detected_names)

        total_objects = len(detected_names)
        avg_conf = sum(confidences) / len(confidences)

        st.subheader("📊 Detection Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("🗑️ Total Objects", total_objects)
        m2.metric("🏷️ Waste Types", len(counts))
        m3.metric("🎯 Avg. Confidence", f"{avg_conf * 100:.1f}%")
        st.markdown("---")

        st.subheader("♻️ Waste Classification")
        st.table([{"Waste Type": n.upper(), "Count": c} for n, c in counts.items()])
        st.markdown("---")

        st.subheader("🚮 Waste Level")
        if total_objects <= 3:
            st.success("🟢 LOW WASTE — Small amount of garbage detected.")
        elif total_objects <= 7:
            st.warning("🟡 MEDIUM WASTE — Moderate amount of garbage detected.")
        else:
            st.error("🔴 HIGH WASTE — Large amount of garbage detected.")
        st.markdown("---")

        st.subheader("🔎 Detection Details")
        for i, (name, conf) in enumerate(zip(detected_names, confidences), start=1):
            st.write(f"**{i}. {name.upper()}** — {conf * 100:.1f}% confidence")
        st.markdown("---")

        st.subheader("📋 Detection Report")
        st.table([
            {"Waste Type": n.upper(), "Confidence": f"{c * 100:.1f}%"}
            for n, c in zip(detected_names, confidences)
        ])

    else:
        st.warning("⚠️ No garbage objects were detected. Try lowering the confidence threshold in the sidebar.")

# ---------------- ABOUT ----------------
st.markdown("---")
st.subheader("📚 About the Project")
st.write(
    """
    **AI Based Waste Detection for Smart Waste Management**

    This project uses a YOLOv8 deep learning object detection model to identify and
    classify different types of garbage from uploaded images. It shows class names,
    confidence scores and the total count of detected objects.

    Built with **Python, YOLOv8, Ultralytics and Streamlit**.
    """
)

st.subheader("✨ Key Features")
f1, f2, f3, f4 = st.columns(4)
with f1:
    st.write("📷 **Image Upload**")
    st.caption("Upload JPG, JPEG or PNG images.")
with f2:
    st.write("🤖 **AI Detection**")
    st.caption("YOLOv8 detects waste objects.")
with f3:
    st.write("📊 **Analysis**")
    st.caption("View counts and confidence scores.")
with f4:
    st.write("♻️ **Waste Level**")
    st.caption("LOW, MEDIUM or HIGH waste.")

st.markdown(
    '<div class="footer">AI Based Waste Detection | YOLOv8 | Deep Learning | Smart Waste Management</div>',
    unsafe_allow_html=True
)