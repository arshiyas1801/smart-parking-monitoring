"""
=============================================================================
Smart Parking Monitoring and Occupancy Detection System
Streamlit Interactive Web Application
Subject: Foundations of Computer Vision (FCV) Lab Project
Target: 2nd Year B.Tech (Artificial Intelligence & Data Science)
=============================================================================
"""

import os
import time
import json
import hashlib
import numpy as np
import cv2
import streamlit as st
from PIL import Image

import parking_detection as pd

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Parking Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .upload-section {
        background-color: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Base Directories & Verified Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_IMG_DIR = os.path.join(BASE_DIR, "dataset", "images")
DATASET_VID_DIR = os.path.join(BASE_DIR, "dataset", "videos")
PARKING_DATA_DIR = os.path.join(BASE_DIR, "parking_data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
JSON_PATH = os.path.join(PARKING_DATA_DIR, "parking_spaces.json")

IMG_REAL_1 = os.path.join(DATASET_IMG_DIR, "real_parking_lot_1.jpg")
IMG_REAL_2 = os.path.join(DATASET_IMG_DIR, "real_parking_lot_2.jpg")
IMG_SAMPLE_1 = os.path.join(DATASET_IMG_DIR, "parking_lot_1.jpg")
IMG_SAMPLE_2 = os.path.join(DATASET_IMG_DIR, "parking_lot_2.jpg")
VID_REAL_1 = os.path.join(DATASET_VID_DIR, "carPark.mp4")


@st.cache_data
def load_reference_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Reference parking image could not be read: {image_path}")
    return frame


def match_reference_lot(frame):
    """Reuse calibrated spaces when an upload is the same reference photo."""
    references = (
        ("real_parking_lot_1", IMG_REAL_1, "Real-World Parking Lot 1"),
        ("real_parking_lot_2", IMG_REAL_2, "Real-World Parking Lot 2"),
    )
    for lot_key, image_path, title in references:
        reference = load_reference_image(image_path)
        if frame.shape == reference.shape:
            similarity = cv2.matchTemplate(frame, reference, cv2.TM_CCOEFF_NORMED)[0, 0]
            if similarity >= 0.97:
                return lot_key, title
    return None, None


def load_drawing_canvas():
    """Bridge the drawable-canvas package to Streamlit's current image helper."""
    import streamlit.elements.image as image_element

    if not hasattr(image_element, "image_to_url"):
        from streamlit.elements.lib.image_utils import image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig

        def image_to_url_compat(image, width, clamp, channels, output_format, image_id):
            return image_to_url(
                image,
                LayoutConfig(width=width),
                clamp,
                channels,
                output_format,
                image_id,
            )

        image_element.image_to_url = image_to_url_compat

    from streamlit_drawable_canvas import st_canvas
    return st_canvas


# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
st.sidebar.title("🚗 Smart Parking")
st.sidebar.caption("FCV Lab Project | 2nd Year AI & DS")

app_mode = st.sidebar.radio(
    "Navigation:",
    ["📷 Image Occupancy Analysis", "🎥 Video Stream Analysis", "📖 Lab Manual & Syllabus Info"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Detection Parameters")

threshold_val = st.sidebar.slider(
    "Occupancy Threshold (Edge Pixels):",
    min_value=50,
    max_value=2500,
    value=900,
    step=50,
    help="If non-zero edge/texture pixels inside the parking space exceed this value, slot is classified as OCCUPIED (Red)."
)

warning_pct = st.sidebar.slider(
    "Occupancy Warning Threshold (%):",
    min_value=50,
    max_value=95,
    value=80,
    step=5,
    help="Displays high-occupancy warning alert when lot occupancy exceeds this percentage."
)

with st.sidebar.expander("Advanced Vision Settings"):
    blur_k = st.selectbox("Gaussian Blur Kernel Size:", [3, 5, 7], index=0)
    adapt_block = st.slider("Adaptive Threshold Block Size:", min_value=11, max_value=51, value=25, step=2)
    adapt_c = st.slider("Adaptive Threshold Constant C:", min_value=5, max_value=30, value=16, step=1)

pklot_available, pklot_msg = pd.check_pklot_available()
st.sidebar.markdown("---")
st.sidebar.caption(f"**PKLot Status:** {'🟢 Detected' if pklot_available else '🟡 Local Images Active (See README)'}")

if st.sidebar.button("🔄 Reset to Default Parameters"):
    st.rerun()

# -----------------------------------------------------------------------------
# Mode 1: Image Occupancy Analysis
# -----------------------------------------------------------------------------
if app_mode == "📷 Image Occupancy Analysis":
    st.markdown('<div class="main-title">Smart Parking Lot Occupancy Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a parking image to count occupied and vacant spaces.</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PROMINENT FRONT-AND-CENTER UPLOAD OPTION
    # -------------------------------------------------------------------------
    st.markdown("### 📤 Upload Your Parking Lot Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"],
        help="The two project reference images use their calibrated bay locations. For another image, draw a rectangle around each bay."
    )

    current_frame = None
    spaces_to_use = []
    lot_title = ""
    reference_lot = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        upload_hash = hashlib.sha256(file_bytes).hexdigest()
        current_frame = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), 1)
        if current_frame is None:
            st.error("The selected file could not be decoded as an image. Please choose a JPG or PNG image.")
        else:
            h_img, w_img = current_frame.shape[:2]
            st.success(f"✅ Image loaded successfully: **{uploaded_file.name}** ({w_img}x{h_img} px)")
            reference_lot, reference_title = match_reference_lot(current_frame)
            if reference_lot is not None:
                spaces_to_use = pd.load_parking_spaces(JSON_PATH, reference_lot)
                lot_title = reference_title
            else:
                st_canvas = load_drawing_canvas()

                st.write("Draw one rectangle around each parking bay. The app will classify the marked regions.")
                display_scale = min(960 / w_img, 650 / h_img, 1.0)
                canvas_width = max(1, int(round(w_img * display_scale)))
                canvas_height = max(1, int(round(h_img * display_scale)))
                canvas_background = Image.fromarray(
                    cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                ).resize((canvas_width, canvas_height))
                canvas_result = st_canvas(
                    fill_color="rgba(0, 180, 0, 0.12)",
                    stroke_width=2,
                    stroke_color="#00C853",
                    background_image=canvas_background,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",
                    display_toolbar=True,
                    key=f"parking_canvas_{upload_hash}",
                )
                canvas_objects = (
                    canvas_result.json_data.get("objects", [])
                    if canvas_result.json_data else []
                )
                for obj in canvas_objects:
                    if obj.get("type") != "rect":
                        continue
                    x = int(round(float(obj.get("left", 0)) * w_img / canvas_width))
                    y = int(round(float(obj.get("top", 0)) * h_img / canvas_height))
                    width = int(round(
                        float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
                        * w_img / canvas_width
                    ))
                    height = int(round(
                        float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
                        * h_img / canvas_height
                    ))
                    if width > 0 and height > 0:
                        spaces_to_use.append({
                            "id": len(spaces_to_use) + 1,
                            "x": x,
                            "y": y,
                            "w": width,
                            "h": height,
                        })
                lot_title = f"Uploaded Parking Lot ({w_img}x{h_img} px)"
    else:
        st.info(
            "Upload one of the supplied parking images to use its calibrated bay locations, "
            "or upload another image and mark its parking bays."
        )

    # -------------------------------------------------------------------------
    # RUN DETECTION & DISPLAY RESULTS
    # -------------------------------------------------------------------------
    if current_frame is not None and len(spaces_to_use) > 0:
        annotated, processed, summary, details = pd.analyze_parking_lot(
            frame=current_frame,
            spaces=spaces_to_use,
            threshold_pixels=threshold_val,
            blur_ksize=(blur_k, blur_k),
            threshold_block_size=adapt_block,
            threshold_c=adapt_c
        )

        st.markdown("---")
        st.write(f"### 📊 Detection Results: {lot_title}")

        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🅿️ Total Spaces", f"{summary['total_spaces']}")
        col2.metric("🚗 Occupied Spaces", f"{summary['occupied_spaces']}", delta=f"{summary['occupancy_percentage']}%", delta_color="inverse")
        col3.metric("🟢 Vacant Spaces Available", f"{summary['vacant_spaces']}")
        col4.metric("📊 Lot Occupancy", f"{summary['occupancy_percentage']}%")

        # Dynamic Status Banner
        if summary["vacant_spaces"] == 0:
            st.error("🛑 **PARKING STATUS: FULL** — Zero vacant spaces available.")
        elif summary["occupancy_percentage"] >= warning_pct:
            st.warning(f"⚠️ **PARKING STATUS: NEARLY FULL ({summary['occupancy_percentage']}%)** — Only **{summary['vacant_spaces']} vacant spaces** remaining!")
        else:
            st.success(f"✅ **PARKING STATUS: SPACES AVAILABLE** — **{summary['vacant_spaces']} vacant parking spaces** are ready for parking.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for detailed viewing
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Annotated Parking Detection View",
            "🔬 Computer Vision Pipeline (Binary Feature Map)",
            "📋 Space-by-Space Detailed Table",
            "📊 Ground Truth & Accuracy Metrics"
        ])

        with tab1:
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"Annotated Result: Green Rectangles = Vacant ({summary['vacant_spaces']}), Red Rectangles = Occupied ({summary['occupied_spaces']})",
                use_container_width=True
            )
            # Download button
            is_success, buffer = cv2.imencode(".jpg", annotated)
            if is_success:
                st.download_button(
                    label="💾 Download Annotated Output Image",
                    data=buffer.tobytes(),
                    file_name="parking_detection_result.jpg",
                    mime="image/jpeg"
                )

        with tab2:
            st.write("**What the Computer Vision Algorithm Analyzes:**")
            st.caption(
                "Pipeline: 1. Grayscale Conversion → 2. Gaussian Blur (noise removal) → "
                "3. Adaptive Gaussian Thresholding (inverse) → 4. Morphological Dilation. "
                "Notice: Empty pavement has almost zero edge pixels (dark), while cars have high edge density (white)."
            )
            c_orig, c_proc = st.columns(2)
            with c_orig:
                st.image(
                    cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB),
                    caption="Original Input Frame",
                    use_container_width=True
                )
            with c_proc:
                st.image(
                    processed,
                    caption=f"Binary Feature Map (Threshold cutoff: {threshold_val} pixels)",
                    use_container_width=True,
                    clamp=True
                )

        with tab3:
            st.write("### Parking Spaces Status Table")
            table_data = []
            for d in details:
                table_data.append({
                    "Space ID": f"#{d['id']}",
                    "Bounding Box (X, Y, W, H)": f"({d['x']}, {d['y']}, {d['w']}, {d['h']})",
                    "Edge Pixel Count": d["non_zero_pixels"],
                    "Status": "🔴 Occupied" if d["is_occupied"] else "🟢 Vacant",
                    "Ground Truth": ("Occupied" if d["ground_truth"] is True else "Vacant") if d["ground_truth"] is not None else "—",
                    "Accuracy": "✅ Match" if (d["ground_truth"] is not None and d["is_occupied"] == d["ground_truth"]) else ("❌ Discrepancy" if d["ground_truth"] is not None else "—")
                })
            st.dataframe(table_data, use_container_width=True)

        with tab4:
            eval_metrics = pd.evaluate_detection(details)
            if eval_metrics["evaluated_spaces"] > 0:
                st.write("### Evaluation Against Ground Truth Labels")
                if lot_title in ("Real-World Parking Lot 1", "Real-World Parking Lot 2"):
                    st.info(
                        "These labels and bay locations belong to this same bundled reference image. "
                        "Agreement here is a reference-image check, not an independent test of general accuracy."
                    )
                e1, e2, e3, e4 = st.columns(4)
                accuracy_label = (
                    "Reference-Image Agreement"
                    if reference_lot is not None else "Empirical Accuracy"
                )
                e1.metric(accuracy_label, f"{eval_metrics['accuracy']}%")
                e2.metric("Precision", f"{eval_metrics['precision']}%")
                e3.metric("Recall", f"{eval_metrics['recall']}%")
                e4.metric("F1-Score", f"{eval_metrics['f1_score']}%")

                st.write("#### Confusion Matrix:")
                cm = eval_metrics["confusion_matrix"]
                c_cm1, c_cm2 = st.columns(2)
                with c_cm1:
                    st.success(f"**True Occupied (TP):** {cm['TP (True Occupied)']} slots")
                    st.info(f"**True Vacant (TN):** {cm['TN (True Vacant)']} slots")
                with c_cm2:
                    st.warning(f"**False Occupied (FP):** {cm['FP (False Occupied)']} slots")
                    st.error(f"**False Vacant (FN):** {cm['FN (False Vacant)']} slots")
            else:
                st.info("ℹ️ Uploaded custom image does not have pre-recorded ground truth labels. Detection was performed dynamically.")
    elif uploaded_file is not None and current_frame is not None and not spaces_to_use:
        st.info("Draw a rectangle around each parking bay above to display occupancy results.")

# -----------------------------------------------------------------------------
# Mode 2: Video Stream Analysis
# -----------------------------------------------------------------------------
elif app_mode == "🎥 Video Stream Analysis":
    st.markdown('<div class="main-title">Real-Time Parking Video Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Stream surveillance video and detect vacant parking spaces dynamically frame-by-frame.</div>', unsafe_allow_html=True)

    video_choice = st.radio(
        "Select Video Source:",
        ["Real-World Surveillance Video (dataset/videos/carPark.mp4)", "Upload Custom MP4 Video"],
        horizontal=True
    )

    vid_path = None
    spaces = pd.load_parking_spaces(JSON_PATH, "real_parking_lot_1")

    if "Real-World Surveillance Video" in video_choice:
        if os.path.exists(VID_REAL_1):
            vid_path = VID_REAL_1
            st.success(f"✅ Real-world surveillance video ready: `{os.path.basename(VID_REAL_1)}` (679 frames, 1100x720, 24 FPS).")
        else:
            st.warning("carPark.mp4 not found in dataset/videos.")
    else:
        uploaded_vid = st.file_uploader("Upload MP4 Video File", type=["mp4", "avi"])
        if uploaded_vid is not None:
            vid_path = os.path.join(DATASET_VID_DIR, "temp_uploaded.mp4")
            with open(vid_path, "wb") as f:
                f.write(uploaded_vid.read())
            st.success("Custom video uploaded successfully.")

    if vid_path and os.path.exists(vid_path):
        max_frames_to_process = st.slider("Max Frames to Process:", min_value=30, max_value=300, value=120, step=30)
        start_btn = st.button("▶️ Start Video Analysis Stream")

        if start_btn:
            cap = cv2.VideoCapture(vid_path)
            frame_placeholder = st.empty()
            metric_placeholder = st.empty()
            progress_bar = st.progress(0)

            frame_idx = 0
            while cap.isOpened() and frame_idx < max_frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % 2 != 0:
                    continue

                annotated, _, summary, _ = pd.analyze_parking_lot(
                    frame, spaces, threshold_pixels=threshold_val,
                    blur_ksize=(blur_k, blur_k),
                    threshold_block_size=adapt_block,
                    threshold_c=adapt_c
                )

                with metric_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Frame", f"{frame_idx} / {max_frames_to_process}")
                    m2.metric("🚗 Occupied", f"{summary['occupied_spaces']}")
                    m3.metric("🟢 Vacant Available", f"{summary['vacant_spaces']}")
                    m4.metric("📊 Occupancy", f"{summary['occupancy_percentage']}%")

                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Real CCTV Stream — Frame #{frame_idx} | Vacant Spaces: {summary['vacant_spaces']}",
                    use_container_width=True
                )
                progress_bar.progress(frame_idx / max_frames_to_process)
                time.sleep(0.04)

            cap.release()
            st.success("✅ Video stream analysis complete.")

# -----------------------------------------------------------------------------
# Mode 3: Lab Manual & Syllabus Info
# -----------------------------------------------------------------------------
else:
    st.markdown('<div class="main-title">Foundations of Computer Vision (FCV) Lab Manual</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Theoretical Concepts, Practical Mapping, and Viva Voce Preparation.</div>', unsafe_allow_html=True)

    tab_s1, tab_s2, tab_s3 = st.tabs(["📚 Syllabus Experiments Mapping", "📐 Mathematical Formulations", "🎓 Viva Voce Q&A"])

    with tab_s1:
        st.write("""
        ### FCV Curriculum Mapping Matrix

| Experiment # | Syllabus Topic | OpenCV Functions Used | Role in Smart Parking System |
|:---|:---|:---|:---|
| **Exp 1** | Basic Image Operations | `cv2.imread`, `cv2.resize`, `cv2.cvtColor` | CCTV frame acquisition, standardizing image dimensions, pixel matrix access. |
| **Exp 2** | Spatial Filtering | `cv2.blur`, `cv2.GaussianBlur`, `cv2.Laplacian` | Gaussian blur suppresses camera sensor noise and road grain before thresholding. |
| **Exp 3** | Edge Detection | `cv2.Sobel`, `cv2.Canny`, `cv2.filter2D` | Distinguishes vehicles from smooth asphalt using gradient magnitude. |
| **Exp 4** | Color Space Conversion | BGR to Grayscale, HSV, CIE-LAB | Grayscale reduces dimensionality; HSV Value channel helps handle shadows. |
| **Exp 5** | Image Segmentation | `cv2.adaptiveThreshold`, Otsu's | Adaptive thresholding handles non-uniform sunlight/shade across the parking lot. |
| **Exp 6** | Texture Feature Extraction | Statistical Contrast, Energy, Entropy | Asphalt exhibits low entropy/contrast; vehicle surfaces exhibit high texture complexity. |
| **Exp 7** | Image Pyramids | `cv2.pyrDown`, `cv2.pyrUp` | Multi-resolution spatial analysis of small or distant parking spaces. |
| **Exp 8** | Capstone Detection | Complete Pipeline Integration | Slot ROI extraction, non-zero pixel thresholding, HUD display, ground truth metrics. |
        """)

    with tab_s2:
        st.write("### Key Mathematical Foundations")
        st.latex(r"G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}} \quad \text{(2D Gaussian Filter)}")
        st.latex(r"T(x, y) = \mu(x, y) - C \quad \text{(Adaptive Gaussian Thresholding)}")
        st.latex(r"\text{Decision Rule: } \text{Slot Status} = \begin{cases} \text{OCCUPIED}, & \text{if } \sum_{(x,y) \in \text{ROI}} I_{\text{binary}}(x, y) > \tau \\ \text{VACANT}, & \text{otherwise} \end{cases}")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \quad \text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}")

    with tab_s3:
        st.write("### Frequently Asked Viva Voce Questions")
        with st.expander("Q1: Why do we convert the BGR image to Grayscale?"):
            st.write("Grayscale reduces color dimensionality from 3 channels (Red, Green, Blue) to 1 intensity channel (0–255). Parking space occupancy depends on geometric contours, edges, and texture rather than car color, saving computation time.")
        with st.expander("Q2: Why is Adaptive Thresholding preferred over Global Thresholding?"):
            st.write("Global thresholding applies a single cutoff across the entire frame. In outdoor parking lots, lighting is non-uniform due to shadows and sunlight. Adaptive thresholding calculates a localized threshold for each pixel neighborhood, making it robust against shadow gradients.")
        with st.expander("Q3: Why is Gaussian Blur applied before Edge Detection / Thresholding?"):
            st.write("Derivative operations (like gradients and Laplacians) amplify high-frequency noise. A Gaussian filter smooths out pixel noise and asphalt gravel grain, preventing spurious edge artifacts.")
        with st.expander("Q4: How does counting non-zero pixels determine parking occupancy?"):
            st.write("Empty asphalt has smooth, uniform texture, resulting in almost zero white pixels after thresholding. In contrast, a vehicle has high-contrast features (windshield, roof seams, headlights, metallic reflections), producing a high non-zero pixel count.")
        with st.expander("Q5: What are the main real-world limitations of this classical vision method?"):
            st.write("1. Heavy shadows cast by adjacent tall trees/buildings can occasionally generate false edges. 2. Severe vehicle occlusion at oblique camera angles. 3. Extreme weather (snow cover or torrential rain) alters pavement reflection.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Smart Parking Monitoring System | AI & Data Science Engineering Lab Project | Built with Python & OpenCV")
