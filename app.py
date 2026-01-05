import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==============================
# 1. PAGE CONFIG (HARUS PALING ATAS)
# ==============================
st.set_page_config(
    page_title="Tumor Analyzer Pro",
    layout="wide"
)

# ==============================
# 2. CSS AMAN (TIDAK MENYEMBUNYIKAN ERROR)
# ==============================
st.markdown("""
<style>
.main .block-container {
    padding-top: 1.5rem;
}
header {
    visibility: hidden;
}
section[data-testid="stSidebar"] label {
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 3. SIDEBAR
# ==============================
st.sidebar.header("⚙️ Kontrol Parameter")

metode = st.sidebar.selectbox(
    "Metode Segmentasi:",
    ["Manual", "Otsu"]
)

t_val = 160
if metode == "Manual":
    t_val = st.sidebar.number_input(
        "Threshold (T Segment):",
        min_value=0,
        max_value=255,
        value=160,
        step=5
    )

operasi_morf = st.sidebar.selectbox(
    "Operasi Morfologi:",
    ["None", "Opening", "Closing", "Erosi", "Dilasi"]
)

kernel_size = st.sidebar.number_input(
    "Ukuran Kernel:",
    min_value=1,
    max_value=15,
    value=3,
    step=2
)

# ==============================
# 4. JUDUL
# ==============================
st.title("🧠 Dashboard Analisis Tumor")

tab1, tab2 = st.tabs([
    "🚀 Analisis Citra",
    "ℹ️ Informasi Metode"
])

# ==============================
# TAB 1 : ANALISIS
# ==============================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload Citra MRI",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        # --- LOAD IMAGE ---
        img_raw = Image.open(uploaded_file).convert("L")
        img = np.array(img_raw)

        # --- ISOLASI AREA (MASK DASAR) ---
        mask_init = cv2.inRange(img, 100, 255)
        img_target = cv2.bitwise_and(img, img, mask=mask_init)

        # --- SEGMENTASI ---
        if metode == "Manual":
            _, binary = cv2.threshold(
                img_target,
                t_val,
                255,
                cv2.THRESH_BINARY
            )
        else:
            # OTSU BENAR (2D IMAGE)
            _, binary = cv2.threshold(
                img_target,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # --- MORFOLOGI ---
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operasi_morf == "Opening":
            final = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif operasi_morf == "Closing":
            final = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif operasi_morf == "Erosi":
            final = cv2.erode(binary, kernel)
        elif operasi_morf == "Dilasi":
            final = cv2.dilate(binary, kernel)
        else:
            final = binary

        # --- VISUALISASI ---
        st.subheader("🖼️ Hasil Segmentasi")

        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        fig.patch.set_facecolor("white")

        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(img_target, cmap="inferno")
        ax[1].set_title("Isolasi Area")
        ax[1].axis("off")

        ax[2].imshow(final, cmap="gray")
        ax[2].set_title("Hasil Segmentasi")
        ax[2].axis("off")

        st.pyplot(fig)

        # --- HISTOGRAM ---
        st.subheader("📊 Histogram Intensitas")

        fig_h, ax_h = plt.subplots(1, 2, figsize=(14, 4))

        ax_h[0].hist(img.ravel(), 256, [0, 256])
        ax_h[0].set_title("Histogram Citra Asli")

        pixels = img_target[img_target > 0]
        if len(pixels) > 0:
            ax_h[1].hist(pixels, 256, [1, 256])
            ax_h[1].set_title("Histogram Area Target")

        st.pyplot(fig_h)

        # --- ESTIMASI LUAS ---
        luas = np.sum(final == 255)
        st.success(f"📐 Estimasi Luas Tumor: {luas} piksel")

    else:
        st.info("⬆️ Silakan upload citra MRI terlebih dahulu.")

# ==============================
# TAB 2 : INFO
# ==============================
with tab2:
    st.header("📘 Informasi Metode")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 Segmentasi")
        st.markdown("""
        **Manual Threshold**
        - Ambang ditentukan pengguna
        - Kontrol penuh intensitas

        **Otsu**
        - Ambang otomatis
        - Berdasarkan variansi piksel
        """)

    with col2:
        st.subheader("🔸 Morfologi")
        st.markdown("""
        **Opening**
        - Menghilangkan noise kecil

        **Closing**
        - Menutup lubang objek

        **Erosi & Dilasi**
        - Mengecilkan / membesarkan objek
        """)
