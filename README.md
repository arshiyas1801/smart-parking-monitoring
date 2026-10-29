# Smart Parking Monitoring and Occupancy Detection System

> **Academic Level:** Second-Year B.Tech (Artificial Intelligence & Data Science)  
> **Course:** Foundations of Computer Vision (FCV) Lab Mini-Project  
> **Technology Stack:** Python, OpenCV, NumPy, Matplotlib, Streamlit  

---

## 1. Project Overview

The **Smart Parking Monitoring and Occupancy Detection System** is a vision-based laboratory project for counting and visualizing occupied and vacant parking spaces in images and surveillance video.

Traditional parking management systems often rely on sensors embedded into each individual parking stall. This project demonstrates image-processing techniques on parking-bay crops. Two included reference images have individually calibrated bay locations; other uploaded images require the user to draw each bay.

### Key Output Features
- **Reference Image Support:** Upload either included 69-space parking image to reuse its calibrated bay locations.
- **Occupied Spaces:** Displayed with **Red bounding rectangles** and non-zero edge pixel counts.
- **Vacant Spaces:** Displayed with **Green bounding rectangles**.
- **Live HUD & Metrics:** Dynamic cards for Total Spaces, Occupied Spaces, Vacant Spaces, Occupancy Percentage, and alert banners ("Spaces Available" vs. "Nearly Full" vs. "Parking Full").
- **Streamlit Web Dashboard:** Upload-first image interface, real-world video player, configurable threshold slider, and side-by-side binary vision feature map inspection.

---

## 2. Project Directory Structure

```text
Smart-Parking-Monitoring/
│
├── app.py                             # Interactive Streamlit dashboard for real image & video testing
├── parking_detection.py              # Core computer vision detection engine & slot manager
├── requirements.txt                   # Project dependencies (OpenCV, NumPy, Matplotlib, Streamlit)
├── README.md                          # Comprehensive lab documentation & viva voce preparation
│
├── dataset/
│   ├── images/
│   │   ├── real_parking_lot_1.jpg     # Real-world aerial CCTV frame (69 spaces: 57 occupied, 12 vacant)
│   │   ├── real_parking_lot_2.jpg     # Real-world frame during vehicle departure (55 occupied, 14 vacant)
│   │   ├── real_parking_lot_3.jpg     # Real-world frame with dynamic occupancy change
│   │   ├── parking_lot_1.jpg          # Calibrated 20-space parking lot (13 occupied, 7 vacant)
│   │   └── parking_lot_2.jpg          # Calibrated 8-space parking lot (5 occupied, 3 vacant)
│   ├── videos/
│   │   └── carPark.mp4                # Real-world surveillance video (679 frames, 1100x720, 24 FPS)
│   └── PKLot/                         # Directory for optional full PKLot dataset (see section 5)
│
├── parking_data/
│   ├── parking_spaces.json            # Calibrated bounding boxes [x, y, w, h] + ground truth
│   └── CarParkPos                     # Binary coordinate positions for 69 real slots
│
├── models/
│   └── pklot_occupancy.joblib         # Retained classifier artifact; current app uses image processing
│
└── outputs/
    └── Smart_Parking_Monitoring_Report.pdf  # Project report
```

---

## 3. Foundations of Computer Vision (FCV) Syllabus Mapping

This project maps directly to core theoretical and practical modules of the Foundations of Computer Vision curriculum:

| Exp # | Syllabus Topic | OpenCV Functions Used | Implementation in Project | Practical Purpose |
|:---:|:---|:---|:---|:---|
| **1** | **Basic Image Operations** | `cv2.imread`, `cv2.resize`, `cv2.cvtColor` | Read parking images, inspect dimensions, display pixel matrices, compute min/max/mean/std. | Standardizes CCTV inputs and inspects camera signal quality. |
| **2** | **Spatial Filtering** | `cv2.blur`, `cv2.GaussianBlur`, `cv2.Laplacian`, `cv2.addWeighted` | Compares Mean Box Blur, Gaussian Smoothing ($\sigma=1.0$), and Laplacian 2nd derivative sharpening. | Suppresses camera sensor noise and asphalt grain before edge extraction. |
| **3** | **Edge Detection** | `cv2.Sobel`, `cv2.filter2D` (Prewitt), `cv2.Canny` | Implements Sobel $G_x/G_y$, custom $3\times3$ Prewitt convolution kernels, and Canny hysteresis. | Vehicles have sharp mechanical contours; empty pavement has near-zero edges. |
| **4** | **Color Space Conversion** | BGR $\rightarrow$ Grayscale, HSV, CIE-LAB | Splits Hue, Saturation, Value ($V$), and Lightness ($L^*$). | Grayscale removes color redundancy; HSV isolates shadow luminance. |
| **5** | **Image Segmentation** | `cv2.threshold`, Otsu's, `cv2.adaptiveThreshold` | Compares global fixed threshold, Otsu's optimal inter-class variance, and Adaptive Gaussian. | **Core detection foundation:** Adaptive Gaussian thresholding handles non-uniform outdoor sunlight and shadows. |
| **6** | **Texture Feature Extraction** | Statistical Contrast, Energy, and Entropy | Extracts contrast (intensity $\sigma$), energy ($\sum p_i^2$), and entropy ($-\sum p_i \log_2 p_i$) on slot crops. | Quantifies surface roughness: parked vehicles exhibit high entropy; asphalt exhibits low entropy. |
| **7** | **Image Pyramids** | `cv2.pyrDown`, `cv2.pyrUp`, `cv2.subtract` | Constructs 4-level Gaussian pyramid and 3-level Laplacian bandpass detail pyramid. | Multi-resolution representation for cameras placed at high or variable elevations. |
| **8** | **Capstone Application** | Complete Pipeline Integration | End-to-end calibrated slot crop, Gaussian blur, adaptive threshold, `cv2.countNonZero`, HUD rendering. | Demonstrates occupancy counting on known layouts. |

---

## 4. Detection Algorithm & Mathematical Formulation

The detection pipeline is designed using classical computer vision principles rather than an opaque black-box neural network:

```
[ Real-World Camera Frame ]
            │
            ▼
[ 1. Grayscale Conversion ]  ──> cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            │
            ▼
[ 2. Gaussian Smoothing ]    ──> cv2.GaussianBlur(gray, (3, 3), 1)
            │
            ▼
[ 3. Adaptive Threshold ]   ──> cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 16)
            │
            ▼
[ 4. Morphological Filter ]  ──> cv2.medianBlur(..., 5) + cv2.dilate(..., 3x3 kernel)
            │
            ▼
[ 5. Slot ROI Extraction ]   ──> crop = binary_map[y:y+h, x:x+w]
            │
            ▼
[ 6. Non-Zero Pixel Count ]  ──> N = cv2.countNonZero(crop)
            │
            ├── If N > Threshold (e.g. 900) ──> OCCUPIED (Red Box)
            └── If N <= Threshold           ──> VACANT (Green Box)
            │
            ▼
[ 7. Summary HUD & Display ] ──> Overlay statistics & status banner
```

### Mathematical Intuition
1. **Gaussian Filter:**
   $$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
   A $3\times3$ Gaussian kernel acts as a low-pass filter, attenuating high-frequency noise from road gravel and sensor gain.

2. **Adaptive Gaussian Thresholding:**
   $$T(x, y) = \sum_{(u,v) \in \mathcal{N}(x,y)} w(u,v) \cdot I(u,v) - C$$
   Unlike global thresholding where a single cutoff fails when part of the lot is shaded, adaptive thresholding dynamically computes a weighted localized threshold for every $25\times25$ neighborhood.

3. **Occupancy Decision Rule:**
   $$\text{Slot Status} = \begin{cases} \text{OCCUPIED (Red)}, & \text{if } \sum_{(x,y) \in \text{ROI}} I_{\text{binary}}(x, y) > \tau \\ \text{VACANT (Green)}, & \text{otherwise} \end{cases}$$
   Where $\tau$ is the configurable non-zero pixel threshold (default: $900$ pixels for real slots, $100$ pixels for large bays).

---

## 5. Dataset Information

### A. Included Real-World Footage & Images
The project comes pre-configured with real-world surveillance assets ready for testing:
- **`dataset/images/real_parking_lot_1.jpg`**: 1100x720 reference frame with 69 calibrated bays. Upload this file to apply its saved layout.
- **`dataset/images/real_parking_lot_2.jpg`**: 1100x720 reference frame with the same 69-bay layout. Upload this file to apply its saved layout.
- **`dataset/images/real_parking_lot_3.jpg`**: Frame 600 showing continued vehicle movement.
- **`dataset/videos/carPark.mp4`**: Full 679-frame (24 FPS) real-world parking lot surveillance video.
- **`parking_data/parking_spaces.json`**: Calibrated coordinates and reference labels for the two 69-bay images.

The Image Occupancy Analysis screen has no preloaded image choices. Upload either bundled reference image with **Browse files**; the app recognizes its image pixels and reuses the corresponding calibrated 69-bay layout. For another image, draw a rectangle around every parking bay in the canvas. No grid size, width, or height settings are required. The occupancy status uses the configured image-processing threshold.

### B. Official PKLot Benchmark Dataset Integration
- **Official Webpage:** [UFPR PKLot Dataset Repository](https://web.inf.ufpr.br/vri/databases/pklot/)
- **Description:** The PKLot benchmark contains 12,417 images captured under Sunny, Cloudy, and Rainy conditions from cameras at UFPR and PUCPR.
- The project includes an extracted PKLot COCO-format dataset under `dataset/PKLot/`.
- The current Streamlit app does not load this dataset or use a PKLot-trained classifier for predictions. No training script is included; do not describe the app as PKLot-trained.

---

## 6. Installation & Execution Guide

### Step 1: Open the Workspace
```bash
cd Smart-Parking-Monitoring
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 7. Reference Image Checks

The following checks compare the threshold-based predictions with labels stored for these same calibrated reference images. They are not independent test-set results and do not establish accuracy on new images:

| Reference image | Spaces | Predicted occupied | Predicted vacant | Agreement with stored labels |
|:---|:---:|:---:|:---:|:---:|
| **`real_parking_lot_1.jpg`** | 69 | 57 | 12 | **69/69** |
| **`real_parking_lot_2.jpg`** | 69 | 55 | 14 | **69/69** |

---

## 8. Viva Voce Preparation Guide (Questions & Model Answers)

#### Q1: Why do we convert the BGR image to Grayscale in Experiment 1?
> **Answer:** Color images contain 3 channels (Red, Green, Blue). In parking occupancy detection, distinguishing a car from pavement depends primarily on structural shapes, edges, and texture rather than color. Grayscale reduces the data from 3 bytes per pixel to 1 byte, cutting computational overhead by ~66% without losing structural geometry.

#### Q2: What is the mathematical difference between Mean Filtering and Gaussian Blur?
> **Answer:** In a Mean filter, all pixels within the kernel window have equal weights ($1/N$). This creates a boxy blur and can smudge sharp edges. In a Gaussian filter, weights are assigned according to a 2D Gaussian bell curve where the center pixel has the highest weight, decaying with distance from the center. Gaussian blur smooths noise while preserving significant edge structures better.

#### Q3: Why is Laplacian sharpening calculated by adding the Laplacian back to the original image?
> **Answer:** The Laplacian operator $\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$ is a second-order derivative that produces zero on uniform areas and large positive/negative values at rapid brightness transitions (edges). By subtracting or adding scaled Laplacian values back to the original image ($I_{\text{sharp}} = I - c \nabla^2 I$), high-frequency edge gradients are boosted, creating a crisper image.

#### Q4: Why is Prewitt edge detection implemented with custom convolution kernels in Experiment 3?
> **Answer:** Sobel kernels have center weighting ($[-1, 0, 1; -2, 0, 2; -1, 0, 1]$) which incorporates smoothing. Prewitt kernels have uniform weighting ($[-1, 0, 1; -1, 0, 1; -1, 0, 1]$). Implementing Prewitt manually via `cv2.filter2D` demonstrates the fundamental discrete 2D spatial convolution operation: $O(x, y) = \sum_i \sum_j I(x-i, y-j) K(i, j)$.

#### Q5: How does Canny edge detection achieve thin, single-pixel edges?
> **Answer:** Canny uses **Non-Maximum Suppression (NMS)**. After computing gradient magnitudes and directions, NMS compares each pixel's gradient against its two neighbors along the gradient normal. Only if the center pixel is a local maximum is it preserved; otherwise, it is suppressed to zero. This thins thick gradient ridges down to 1-pixel contours.

#### Q6: When is the HSV color space preferred over RGB in computer vision?
> **Answer:** In RGB, color and brightness are intertwined across all three channels. In HSV, chromatic content is isolated in **Hue** ($H$) and **Saturation** ($S$), while illumination intensity is separated into **Value** ($V$). When shadows fall over a parking lot, the $V$ channel drops, but the $H$ channel remains relatively constant, allowing shadow-invariant tracking.

#### Q7: Why does Otsu’s thresholding fail when lighting is non-uniform?
> **Answer:** Otsu’s algorithm assumes a bimodal intensity histogram (one peak for background, one for foreground) and searches for a single global threshold that maximizes inter-class variance. When an outdoor scene has strong sunlight on one side and deep shadows on the other, the histogram is multimodal, causing a single global threshold to incorrectly segment the shaded half.

#### Q8: How does Adaptive Gaussian Thresholding solve this illumination issue?
> **Answer:** Instead of a single cutoff, Adaptive Gaussian calculates a distinct threshold for every pixel based on the Gaussian-weighted mean of its surrounding $B \times B$ window (e.g., $25\times25$) minus a constant $C$. Pixels in dark shadow regions are compared only against their dark local neighborhood, ensuring reliable edge extraction everywhere.

#### Q9: What is the physical meaning of Texture Entropy in Experiment 6?
> **Answer:** Entropy measures the randomness or information content of the gray-level probability distribution: $H = -\sum p_i \log_2(p_i)$. A smooth, uniform asphalt patch has most pixels concentrated in a few gray levels (low entropy $\approx 2.5$). A vehicle has metallic bodywork, dark glass, bright chrome, and tires (high entropy $\approx 7.4$).

#### Q10: What is the difference between Gaussian and Laplacian pyramids?
> **Answer:** A Gaussian pyramid is a low-pass representation where each level is smoothed and downsampled by half (`cv2.pyrDown`). A Laplacian pyramid is a bandpass representation: each level contains the difference between a Gaussian level and the expanded version of the next coarser level ($L_i = G_i - \text{pyrUp}(G_{i+1})$), capturing details lost during downsampling.

#### Q11: Why is Morphological Dilation applied after thresholding in our parking system?
> **Answer:** Binary thresholding can break vehicle contours into fragmented, disconnected dots. Morphological dilation (`cv2.dilate`) uses a structuring element (e.g., $3\times3$ ones) to expand white pixel boundaries, bridging small gaps and forming coherent vehicle clusters.

#### Q12: Why do we use `cv2.countNonZero` instead of calculating mean intensity?
> **Answer:** After binary inverse thresholding, edges and sharp contours appear as white pixels (value 255) while flat surfaces remain black (value 0). `cv2.countNonZero` directly tallies the area of structural edges inside the slot ROI. Counting non-zero pixels is computationally faster than computing floating-point mean and variance.

#### Q13: What is the effect of changing the Gaussian blur kernel from $3\times3$ to $15\times15$?
> **Answer:** A larger kernel ($15\times15$) excessively blurs the image, eliminating fine vehicle contours such as window frames and side mirrors. This lowers the non-zero pixel count for parked cars and increases False Negatives. A compact $3\times3$ or $5\times5$ kernel removes sensor noise while preserving crucial car edges.

#### Q14: How does camera perspective affect the bounding box dimensions?
> **Answer:** Due to perspective projection, parking spaces closer to the camera appear larger in pixel dimensions, while distant slots appear smaller. When defining fixed $(x, y, w, h)$ bounding boxes, slot dimensions must be scaled or calibrated row-by-row according to the camera's tilt angle.

#### Q15: How could this classical vision pipeline be improved for commercial deployment?
> **Answer:**
> 1. Incorporate temporal smoothing (a rolling average over 15–30 frames) to prevent spot flickering when pedestrians walk past.
> 2. Use a homography matrix / bird's-eye perspective transformation (`cv2.warpPerspective`) to rectify oblique angles into a top-down orthogonal grid.
> 3. Add an optional lightweight CNN (e.g., MobileNetV3) fine-tuned on PKLot to classify ambiguous or heavily shaded slots.

---

## 9. Conclusion

This project demonstrates fundamental image-processing techniques—filtering, edge detection, color-space conversion, and adaptive thresholding—through the Streamlit dashboard and its detection module. The two included reference images use their calibrated bay layouts; other images require manual bay marking, and the reference checks should not be interpreted as generalization accuracy.
