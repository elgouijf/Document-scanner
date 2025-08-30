# Document-scanner

An educational and practical project designed to automatically scan and transform photos of documents into clear, flattened scanned images using Python and OpenCV. This tool not only applies powerful image processing techniques but also emphasizes the underlying mathematics that make these techniques work.

Developed by: [elgouijf](https://github.com/elgouijf)

---

## 📌 Overview

This project is both **functional and educational**: it demonstrates how to build a document scanner pipeline, while deeply exploring the **mathematical principles** behind each stage.

The scanning workflow involves:
- Preprocessing and noise reduction  
- Edge, contour and corner detection  (although corner detection proved to be less intersting in this context )
- Perspective transformation via homographies  
- Contrast enhancement with adaptive thresholding  
- building a "bird view" of the document

Although built with OpenCV, this project prioritizes **mathematical understanding**. Many functions (e.g., Gaussian filtering, gradient detection, thresholding) are candidates for manual reimplementation to **ground theory in practice**.

---

## 🔍 Mathematical Foundations

This section outlines the theoretical tools that underlie each step of the pipeline.

---

### 📐 1. Linear Algebra in Image Transformations

Images are treated as matrices where each pixel corresponds to a scalar (grayscale) or vector (color). Transformation of images — such as rotations, scalings, and projections — are done through **matrix multiplication**.

- **Affine transforms** use a 2×3 matrix.
- **Homographies** use a 3×3 matrix in projective space (`P^2`) to model perspective warping.
- **Projection matrices** map from 3D world coordinates to 2D image planes. By moving to **homogeneous coordinates**, we can express complex transformations as **matrix chains**, enabling elegant composition of operations (e.g., rotation + scaling + projection).

---

### 🌫️ 2. Gaussian Filtering and Noise Removal

Noise in scanned images arises from sensor artifacts or lighting inconsistencies. We remove it using the **Gaussian filter**, which performs a low-pass operation:

- **Spatial domain:** weights neighboring pixels using a Gaussian kernel  
- **Frequency domain:** suppresses high-frequency noise (via convolution theorem)

**G(x, y) = (1 / (2πσ²)) * exp(-(x² + y²) / (2σ²))**


Applying it is equivalent to **smoothing** the image, preparing it for stable gradient and contour detection.

---

### ⚡ 3. Gradients and Edge Detection

Edges are detected by computing **image gradients** (first derivatives):

- **∇I = [∂I/∂x, ∂I/∂y]**
- Typically computed via Sobel or Scharr filters (discrete convolution masks)
- Strong gradient magnitudes indicate likely edges

---

### 🧠 4. Harris Corner Detection & Second-Moment Matrix

To extract **the outer-layer corners**, I initially considered using the Harris corner detector, which identifies points where image intensity varies significantly in two orthogonal directions—typically corresponding to corners and junctions in the image.

It uses the **second moment matrix** :
       ┌                            ┐
   M = │ ∑(Iₓ²)      ∑(Iₓ·Iᵧ)        |
       │ ∑(Iₓ·Iᵧ)    ∑(Iᵧ²)          │
       └                            ┘

The **eigenvalues** of \( M \) measure variation along different directions:
- Two large eigenvalues → corner
- One large, one small → edge
- Both small → flat region

The **Harris response**:

**R = det(M) - k * (trace(M))²**

Corner candidates are selected based on \( R \).

However, filtering out **the inner-layer corners** proved to be quite challenging, so I eventually abandoned that approach. Instead, I discovered a more effective method using contour detection. Nonetheless, exploring this path gave me valuable insights and deepened my understanding of image analysis.


### 🎭 5. Adaptive Thresholding

To produce a clean black-and-white scan, we apply **adaptive thresholding**:

- Instead of a global threshold, compute a local one based on neighborhood statistics.
- Often use a **Gaussian-weighted mean** of the local area, subtracting a constant offset.

This handles lighting variation across the document and improves visual clarity.

---

## 🚀 Getting Started

### 1. Clone the repository


    git clone https://github.com/elgouijf/Document-scanner.git
    cd Document-scanner

### 2. Run the code and observe

## How to run example.py on test_doc.png: (The manual version of our document scanner)

python example.py --image testing_files/test_doc.png --coordinates 120,80,380,60,420,380,80,400

where 120,80,380,60,420,380,80,400 are the coordinates of the corners that are to enter manually

## How to run document_scanner

python document_scanner.py --image path to image (in my case a file from testing_files/)

Once you launch the script, it will open one or more OpenCV windows to show each step of the scanning pipeline:

- To exit the program, click any key while your mouse is hovering over one of the OpenCV windows.

- A message box will appear, asking you whether to display the output in:

    - Grayscale

    - Color (RGB)


⚠️ Compatibility Note (NumPy, scikit-image, Python)

Running this document scanner requires careful alignment of library versions. In particular:

- scikit-image (especially skimage.filters.threshold_local) may conflict with newer versions of NumPy (e.g., ≥ 1.24).

- Some functions may also be unstable or broken with Python 3.11+.

✅ Recommended Setup (Using Conda)

To avoid versioning issues, it is highly recommended to create a Conda virtual environment with pinned versions of the dependencies:

conda create -n docscanner-env python=3.9
conda activate docscanner-env

conda install numpy=1.23.5 scikit-image=0.20.0 opencv imutils

If imutils is not found via conda, you can install it with pip inside the activated environment:

pip install imutils

📌 Why this matters

These version pins are essential to ensure compatibility with key functions like:

- threshold_local() from skimage.filters

- cv2.findContours() behavior across OpenCV versions

- Data type assumptions in NumPy array operations
