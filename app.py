import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Tumor Analyzer Dashboard",
    layout="wide"
)

# ================== CSS RAPI ==================
st.markdown("""
<style>
.main .block-container {
    padding-top: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.header("⚙️ Kontrol Parameter")

metode = st.sidebar.selectbox(
    "Metode Segmentasi",
    ["Manual", "Otsu"]
)

threshold_val = 160
if metode == "Manual":
    threshold_val = st.sidebar.number_input(
        "Nilai Threshold",
        min_value=0,
        max_value=255,
        value=160,
        step=1
    )

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

tampilkan_hist = st.sidebar.checkbox("Tampilkan Histogram", value=True)

# ================== JUDUL ==================
st.title("🧠 Dashboard Analisis Tumor Berbasis Citra")

tab1, tab2 = st.tabs([
    "🚀 Analisis Citra",
    "ℹ️ Informasi Metode"
])

# =================================================
# ================== TAB 1 ========================
# =================================================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload Citra Medis (MRI / CT)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.warning("Silakan upload citra medis untuk memulai analisis.")
    else:
        # ===== LOAD IMAGE =====
        img_gray = np.array(
            Image.open(uploaded_file).convert("L")
        )

        # ===== MASKING (PENTING) =====
        # Fokus area jaringan terang (calon tumor)
        mask_init = cv2.inRange(img_gray, 100, 255)
        img_target_tumor = cv2.bitwise_and(
            img_gray, img_gray, mask=mask_init
        )

        # ===== THRESHOLDING =====
        if metode == "Manual":
            _, binary = cv2.threshold(
                img_target_tumor,
                threshold_val,
                255,
                cv2.THRESH_BINARY
            )
        else:
            pixels = img_target_tumor[img_target_tumor > 0]
            if len(pixels) > 0:
                otsu_val, _ = cv2.threshold(
                    pixels,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                _, binary = cv2.threshold(
                    img_target_tumor,
                    otsu_val,
                    255,
                    cv2.THRESH_BINARY
                )
            else:
                binary = np.zeros_like(img_target_tumor)

        # ===== MORFOLOGI =====
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

        # ===== VISUALISASI =====
        st.subheader("🖼️ Hasil Segmentasi")

        fig, ax = plt.subplots(1, 3, figsize=(22, 7))

        ax[0].imshow(img_gray, cmap="gray")
        ax[0].set_title("1. Citra Original")
        ax[0].axis("off")

        ax[1].imshow(img_target_tumor, cmap="inferno")
        ax[1].set_title("2. Hasil Masking (Area Target)")
        ax[1].axis("off")

        ax[2].imshow(final, cmap="gray")
        ax[2].set_title(f"3. Hasil Biner ({metode})")
        ax[2].axis("off")

        st.pyplot(fig)

        # ===== HISTOGRAM (OPSIONAL) =====
        if tampilkan_hist:
            st.markdown("---")
            st.subheader("📊 Histogram Intensitas")

            fig_h, ax_h = plt.subplots(1, 2, figsize=(15, 5))

            ax_h[0].hist(
                img_gray.ravel(),
                256,
                [0, 256],
                color="gray"
            )
            ax_h[0].set_title("Histogram Seluruh Citra")

            pixels_iso = img_target_tumor[img_target_tumor > 0].ravel()
            if len(pixels_iso) > 0:
                ax_h[1].hist(
                    pixels_iso,
                    256,
                    [1, 256],
                    color="orange"
                )
                ax_h[1].set_title("Histogram Area Target (Tumor)")

            st.pyplot(fig_h)

            st.markdown("""
            **Interpretasi Histogram:**
            - Puncak intensitas tinggi menunjukkan area jaringan dengan kecerahan tinggi.
            - Distribusi pada area target mencerminkan karakteristik massa tumor
              yang bersifat *hyperintens* pada citra medis.
            """)

        # ===== LUAS =====
        luas = np.sum(final == 255)
        st.success(f"📐 Estimasi Luas Massa Tumor: **{luas} piksel**")

# =================================================
# ================== TAB 2 ========================
# =================================================
with tab2:
    st.header("📘 Informasi Metode")

    st.subheader("1️⃣ Masking Citra")
    st.write("""
    Masking digunakan untuk membatasi area analisis agar fokus
    pada jaringan dengan intensitas tinggi. Tahapan ini penting
    untuk menghindari gangguan dari background dan tulang.
    """)

    st.subheader("2️⃣ Segmentasi Threshold")
    st.write("""
    - **Manual Threshold**: Nilai ambang ditentukan langsung oleh pengguna.
    - **Otsu**: Algoritma otomatis yang meminimalkan varians intra-kelas
      untuk menentukan threshold optimal.
    """)

    st.subheader("3️⃣ Operasi Morfologi")
    st.write("""
    Operasi morfologi digunakan untuk menyempurnakan hasil segmentasi:
    - Opening / Erosi → menghilangkan noise kecil.
    - Closing / Dilasi → menutup lubang pada objek tumor.
    """)

    st.subheader("4️⃣ Analisis Luas Tumor")
    st.write("""
    Luas tumor dihitung berdasarkan jumlah piksel putih pada citra biner
    akhir, yang dapat digunakan sebagai parameter kuantitatif
    dalam analisis medis.
    """)
