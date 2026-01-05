import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Analisis Tumor",
    layout="wide"
)

# =========================
# STYLE SEDERHANA (AMAN)
# =========================
st.markdown("""
<style>
.main .block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR PARAMETER
# =========================
st.sidebar.header("⚙️ Kontrol Parameter")

metode = st.sidebar.selectbox(
    "Metode Thresholding",
    ["Manual", "Otsu"]
)

if metode == "Manual":
    thresh_val = st.sidebar.slider(
        "Nilai Threshold (T)",
        0, 255, 160
    )
else:
    thresh_val = None

operasi_morf = st.sidebar.selectbox(
    "Operasi Morfologi",
    ["None", "Opening", "Closing", "Erosi", "Dilasi"]
)

kernel_size = st.sidebar.slider(
    "Ukuran Kernel",
    1, 15, 3, step=2
)

show_hist = st.sidebar.checkbox("Tampilkan Histogram", value=True)

# =========================
# HALAMAN UTAMA
# =========================
st.title("🧠 Dashboard Analisis Tumor Berbasis Segmentasi")
st.caption("Citra Medis: MRI / CT Scan / Mammografi (Simulasi Akademik)")

uploaded_file = st.file_uploader(
    "Upload Citra Medis (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # =========================
    # LOAD & PREPROCESS
    # =========================
    img = np.array(
        Image.open(uploaded_file).convert("L")
    )

    # --- MASKING AWAL (ROI KASAR) ---
    # Menghilangkan background hitam & tulang terlalu terang
    mask_init = cv2.inRange(img, 100, 255)
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
            otsu_val, _ = cv2.threshold(
                pixels, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            _, binary = cv2.threshold(
                img_target, otsu_val, 255, cv2.THRESH_BINARY
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
    # MASKING LANJUTAN (KUNCI ILMIAH)
    # =========================
    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    tumor_mask = np.zeros_like(morph)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(
            tumor_mask, [largest], -1, 255, thickness=-1
        )
    else:
        tumor_mask = morph.copy()

    tumor_only = cv2.bitwise_and(img, img, mask=tumor_mask)

    # =========================
    # VISUALISASI UTAMA
    # =========================
    st.subheader("🖼️ Hasil Pemrosesan Citra")

    fig, ax = plt.subplots(1, 3, figsize=(22, 7))

    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("1. Citra Asli")
    ax[0].axis("off")

    ax[1].imshow(img_target, cmap="inferno")
    ax[1].set_title("2. ROI Awal (Mask Intensitas)")
    ax[1].axis("off")

    ax[2].imshow(tumor_only, cmap="gray")
    ax[2].set_title("3. Tumor Candidate (Final Mask)")
    ax[2].axis("off")

    st.pyplot(fig)

    # =========================
    # HISTOGRAM
    # =========================
    if show_hist:
        st.markdown("---")
        st.subheader("📊 Analisis Histogram")

        fig_h, ax_h = plt.subplots(1, 2, figsize=(15, 5))

        ax_h[0].hist(
            img.ravel(), bins=256, range=(0, 256),
            color="gray"
        )
        ax_h[0].set_title("Histogram Citra Asli")

        pixels_tumor = tumor_only[tumor_only > 0].ravel()
        if len(pixels_tumor) > 0:
            ax_h[1].hist(
                pixels_tumor, bins=256, range=(1, 256),
                color="orange"
            )
            ax_h[1].set_title("Histogram Area Tumor")

        st.pyplot(fig_h)

        st.markdown("""
        **Interpretasi Histogram:**
        - Histogram kiri menunjukkan distribusi intensitas seluruh citra.
        - Histogram kanan hanya merepresentasikan area hasil segmentasi.
        - Pergeseran distribusi ke intensitas tinggi menunjukkan karakteristik
          **hyperintens** yang umum pada massa tumor di citra MRI.
        """)

    # =========================
    # INFO KUANTITATIF
    # =========================
    area = np.sum(tumor_mask == 255)
    st.success(f"Estimasi Luas Area Tumor: **{area} piksel**")

else:
    st.info("Silakan upload citra medis untuk memulai analisis.")
