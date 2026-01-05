import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Analisis Tumor", layout="wide")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Parameter")

metode = st.sidebar.selectbox(
    "Metode Thresholding",
    ["Manual", "Otsu"]
)

if metode == "Manual":
    thresh_val = st.sidebar.number_input(
        "Nilai Threshold (0–255)",
        min_value=0,
        max_value=255,
        value=150,
        step=1
    )
else:
    thresh_val = None

operasi_morf = st.sidebar.selectbox(
    "Operasi Morfologi",
    ["None", "Opening", "Closing", "Erosi", "Dilasi"]
)

kernel_size = st.sidebar.number_input(
    "Ukuran Kernel (ganjil)",
    min_value=1,
    max_value=15,
    value=3,
    step=2
)

show_hist = st.sidebar.checkbox("Tampilkan Histogram", value=True)

# =========================
# MAIN
# =========================
st.title("🧠 Dashboard Analisis Tumor Citra Medis")

uploaded_file = st.file_uploader(
    "Upload Citra Medis (MRI / CT / Mammografi)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file).convert("L"))

    # =========================
    # MASKING AWAL (FIXED)
    # =========================
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    mask_init = cv2.adaptiveThreshold(
        img_norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    img_target = cv2.bitwise_and(img, img, mask=mask_init)

    # =========================
    # THRESHOLDING
    # =========================
    if metode == "Manual":
        _, binary = cv2.threshold(
            img_target, thresh_val, 255, cv2.THRESH_BINARY
        )
    else:
        pixels = img_target[img_target > 0]
        if len(pixels) > 0:
            t_otsu, _ = cv2.threshold(
                pixels, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            _, binary = cv2.threshold(
                img_target, t_otsu, 255, cv2.THRESH_BINARY
            )
        else:
            binary = np.zeros_like(img_target)

    # =========================
    # MORFOLOGI
    # =========================
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operasi_morf == "Opening":
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif operasi_morf == "Closing":
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    elif operasi_morf == "Erosi":
        morph = cv2.erode(binary, kernel)
    elif operasi_morf == "Dilasi":
        morph = cv2.dilate(binary, kernel)
    else:
        morph = binary.copy()

    # =========================
    # MASKING LANJUTAN (TUMOR)
    # =========================
    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    tumor_mask = np.zeros_like(morph)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(tumor_mask, [largest], -1, 255, -1)

    tumor_only = cv2.bitwise_and(img, img, mask=tumor_mask)

    # =========================
    # VISUALISASI
    # =========================
    fig, ax = plt.subplots(1, 3, figsize=(22, 7))

    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("1. Citra Asli")
    ax[0].axis("off")

    ax[1].imshow(img_target, cmap="inferno")
    ax[1].set_title("2. Hasil Masking Awal (ROI)")
    ax[1].axis("off")

    ax[2].imshow(tumor_only, cmap="gray")
    ax[2].set_title("3. Tumor Candidate (Final)")
    ax[2].axis("off")

    st.pyplot(fig)

    # =========================
    # HISTOGRAM
    # =========================
    if show_hist:
        fig_h, ax_h = plt.subplots(1, 2, figsize=(15, 5))

        ax_h[0].hist(img.ravel(), 256)
        ax_h[0].set_title("Histogram Citra Asli")

        pix = tumor_only[tumor_only > 0].ravel()
        if len(pix) > 0:
            ax_h[1].hist(pix, 256, color="orange")
            ax_h[1].set_title("Histogram Area Tumor")

        st.pyplot(fig_h)

    st.success(f"Estimasi luas area tumor: {np.sum(tumor_mask == 255)} piksel")

else:
    st.info("Silakan upload citra medis.")
