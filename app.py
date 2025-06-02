import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Monitor Deteksi Distraksi Siswa",
    page_icon="🧑‍🎓",  # Ikon diubah
    layout="centered",
)

# CSS untuk styling (tetap sama, sudah cukup baik)
st.markdown(
    """
<style>
    .status-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .status-normal { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .status-engaged { background-color: #cce5ff; color: #004085; border: 2px solid #99d6ff; }
    .status-warning { background-color: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
    .status-danger { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Mapping status dan warna - Disesuaikan untuk siswa
STATUS_CONFIG = {
    "Normal": {"color": "status-normal", "emoji": "✅", "level": "FOKUS"},
    "Engaged": {"color": "status-engaged", "emoji": "👀", "level": "AKTIF BELAJAR"},
    "Distracted": {"color": "status-warning", "emoji": "⚠️", "level": "TERDISTRAKSI"},
    "Compromised": {
        "color": "status-danger",
        "emoji": "🚨",
        "level": "PERLU PERHATIAN",  # Diubah dari BAHAYA
    },
    "Face Covered": {
        "color": "status-warning",
        "emoji": "😷",
        "level": "WAJAH TERTUTUP",
    },
    "Face Concealed": {
        "color": "status-warning",
        "emoji": "🎭",
        "level": "WAJAH TERSEMBUNYI",
    },
    "Face Not Visible": {
        "color": "status-warning",
        "emoji": "👤",
        "level": "WAJAH TIDAK TERLIHAT",
    },
    "Object Detected": {
        "color": "status-warning",
        "emoji": "📱",  # Bisa berarti HP atau objek lain
        "level": "OBJEK TERDETEKSI",
    },
}


@st.cache_resource
def load_model():
    """Load YOLO model"""
    try:
        model = YOLO(
            "best.pt"
        )  # Pastikan file best.pt ada di direktori yang sama atau berikan path lengkap
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def get_detection_status(results, confidence_threshold=0.5):
    """Ambil status deteksi dengan confidence tertinggi"""
    best_detection = None
    max_confidence = 0

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence >= confidence_threshold and confidence > max_confidence:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    max_confidence = confidence
                    best_detection = {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": box.xyxy[0].tolist(),
                    }
    return best_detection


def draw_detection(image, detection):
    """Gambar bounding box pada gambar"""
    if detection is None:
        return image

    img_array = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2 = map(int, detection["bbox"])
    class_name = detection["class"]
    confidence = detection["confidence"]

    # Tentukan warna berdasarkan status dari STATUS_CONFIG
    status_info = STATUS_CONFIG.get(
        class_name, {"color": "status-normal"}
    )  # Default ke normal jika tidak ada

    # Ambil warna BGR dari hex (misalnya, #d4edda -> (218, 237, 212)) atau definisikan secara manual
    # Untuk simple, kita gunakan logika warna yang sudah ada
    if class_name in ["Normal", "Engaged"]:
        box_color = (0, 255, 0)  # Hijau
    elif (
        class_name in STATUS_CONFIG
        and STATUS_CONFIG[class_name]["color"] == "status-warning"
    ):
        box_color = (0, 165, 255)  # Orange
    elif (
        class_name in STATUS_CONFIG
        and STATUS_CONFIG[class_name]["color"] == "status-danger"
    ):
        box_color = (0, 0, 255)  # Merah
    else:
        box_color = (255, 0, 0)  # Biru untuk default jika tidak termapping

    cv2.rectangle(img_cv, (x1, y1), (x2, y2), box_color, 3)
    label = f"{class_name}: {confidence:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    cv2.rectangle(
        img_cv,
        (x1, y1 - text_height - 15),
        (x1 + text_width, y1 - 5),  # Disesuaikan sedikit agar pas
        box_color,
        -1,
    )
    cv2.putText(
        img_cv,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def main():
    # Header
    st.title("🧑‍🎓 Monitor Deteksi Distraksi Siswa")
    st.markdown("**Pantau fokus belajarmu secara real-time!**")
    st.markdown(
        "Aplikasi ini membantu mendeteksi tingkat fokus menggunakan kamera saat kamu belajar online (misalnya via Google Meet, Zoom, dll)."
    )
    st.markdown("---")

    model = load_model()
    if model is None:
        st.warning(
            "Model tidak dapat dimuat. Pastikan file 'best.pt' ada di direktori aplikasi."
        )
        st.stop()

    st.success("✅ Model pendeteksi berhasil dimuat!")

    with st.sidebar:
        st.header("⚙️ Pengaturan")
        confidence_threshold = st.slider(
            "Ambang Batas Kepercayaan (Confidence)",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Semakin tinggi nilainya, model akan semakin selektif dalam mendeteksi.",
        )
        st.markdown("---")
        st.markdown("### 📊 Legenda Status")
        for status_name, config in STATUS_CONFIG.items():
            st.markdown(f"{config['emoji']} **{status_name}** - {config['level']}")
        st.markdown("---")
        st.info(
            "Aplikasi ini menggunakan model YOLO untuk deteksi. Akurasi dapat bervariasi."
        )

    tab1, tab2 = st.tabs(
        ["📷 Kamera Real-time", "🖼️ Unggah Gambar"]
    )  # Mengubah ikon tab 1

    with tab1:
        st.header("📷 Monitoring Real-time via Kamera")
        st.write("Nyalakan kameramu dan lihat status fokusmu!")

        col1, col2, col3 = st.columns(3)
        with col1:
            start_btn = st.button(
                "🎥 Mulai Monitoring", type="primary", use_container_width=True
            )
        with col2:
            stop_btn = st.button("⏹️ Stop Monitoring", use_container_width=True)
        with col3:
            capture_btn_clicked = st.button(
                "📸 Info Frame",
                use_container_width=True,
                help="Menampilkan info frame saat ini (tidak menyimpan gambar).",
            )

        status_placeholder = st.empty()
        video_placeholder = st.empty()

        st.markdown("---")
        st.subheader("📈 Metrik Deteksi")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            fps_placeholder = st.empty()
        with metric_col2:
            confidence_display_placeholder = st.empty()
        with metric_col3:
            timestamp_placeholder = st.empty()

        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False

        if start_btn:
            st.session_state.webcam_running = True
            st.toast("Kamera sedang dimulai...", icon="📸")  # Mengganti ikon toast

        if stop_btn:
            st.session_state.webcam_running = False
            st.toast("Monitoring dihentikan.", icon="🛑")

        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error(
                    "❌ Webcam tidak terdeteksi. Pastikan terhubung dan tidak digunakan aplikasi lain."
                )
                st.session_state.webcam_running = (
                    False  # Hentikan jika tidak bisa buka webcam
                )
            else:
                frame_count = 0
                start_time = time.time()
                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning(
                            "⚠️ Gagal membaca frame. Mencoba lagi..."
                        )  # Mengganti ikon warning
                        cap.release()
                        time.sleep(0.5)
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened():
                            st.error("❌ Gagal menghubungkan kembali webcam.")
                            st.session_state.webcam_running = False
                            break
                        continue

                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    results = model(frame_rgb, verbose=False)
                    detection = get_detection_status(results, confidence_threshold)
                    detected_image = draw_detection(pil_image, detection)

                    video_placeholder.image(
                        detected_image, channels="RGB", use_container_width=True
                    )

                    current_status_name = "Tidak Terdeteksi"
                    current_confidence = 0.0

                    if detection:
                        status_name = detection["class"]
                        status_conf_val = detection["confidence"]
                        current_status_name = status_name
                        current_confidence = status_conf_val

                        if status_name in STATUS_CONFIG:
                            config = STATUS_CONFIG[status_name]
                            status_placeholder.markdown(
                                f"""
                                <div class="status-alert {config['color']}">
                                    {config['emoji']} Status: {config['level']} ({status_name})
                                    <br>Kepercayaan: {status_conf_val:.2f}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:  # Jika kelas tidak ada di STATUS_CONFIG
                            status_placeholder.markdown(
                                f"""
                                <div class="status-alert status-normal">
                                    ℹ️ Status: {status_name} (Kepercayaan: {status_conf_val:.2f})
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        confidence_display_placeholder.metric(
                            "Kepercayaan", f"{status_conf_val:.3f}"
                        )
                    else:
                        status_placeholder.markdown(
                            """
                            <div class="status-alert status-normal">
                                🔍 Mencari wajah atau aktivitas... Sistem standby.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        confidence_display_placeholder.metric("Kepercayaan", "N/A")

                    frame_count += 1
                    elapsed = time.time() - start_time
                    if elapsed > 0.1:  # Update FPS tidak terlalu sering
                        fps = frame_count / elapsed
                        fps_placeholder.metric("FPS", f"{fps:.1f}")
                        # Reset untuk perhitungan FPS berikutnya agar lebih akurat per interval
                        # frame_count = 0
                        # start_time = time.time()

                    timestamp_placeholder.metric("Waktu", time.strftime("%H:%M:%S"))

                    if capture_btn_clicked:
                        st.toast(
                            f"Info: Status '{current_status_name}' @{current_confidence*100:.1f}% | {time.strftime('%H:%M:%S')}",
                            icon="💡",
                        )

                    # Penting untuk Streamlit agar UI bisa update dan tidak freeze
                    # cv2.waitKey(1) # Tidak perlu di Streamlit, time.sleep lebih baik
                    time.sleep(0.01)  # Mengatur refresh rate, sesuaikan jika perlu

                cap.release()
                if (
                    not st.session_state.webcam_running
                ):  # Hanya kosongkan jika memang di-stop
                    video_placeholder.empty()
                    status_placeholder.empty()
                    fps_placeholder.empty()
                    confidence_display_placeholder.empty()
                    timestamp_placeholder.empty()
                    st.info("ℹ️ Monitoring via kamera telah dihentikan.")

    with tab2:
        st.header("🖼️ Analisis Gambar Siswa")
        st.write(
            "Unggah fotomu (atau screenshot saat belajar) untuk dianalisis tingkat fokusnya."
        )

        uploaded_file = st.file_uploader(
            "Pilih gambar siswa",
            type=["jpg", "jpeg", "png"],
            help="Format yang didukung: JPG, JPEG, PNG.",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            st.subheader("🖼️ Pratinjau Gambar")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.caption("Gambar Asli")
                st.image(image, use_container_width=True)

            with st.spinner(
                "🧠 Menganalisis gambar... Ini mungkin butuh beberapa detik."
            ):
                results = model(image, verbose=False)
                detection = get_detection_status(results, confidence_threshold)
                detected_image = draw_detection(image.copy(), detection)

            with col_img2:
                st.caption("Hasil Deteksi")
                st.image(detected_image, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Hasil Analisis Detail")
            if detection:
                status_name = detection["class"]
                status_conf_val = detection["confidence"]

                if status_name in STATUS_CONFIG:
                    config = STATUS_CONFIG[status_name]
                    st.markdown(
                        f"""
                        <div class="status-alert {config['color']}">
                            {config['emoji']} STATUS TERDETEKSI: {config['level']} ({status_name})
                            <br>Tingkat Kepercayaan Deteksi: {status_conf_val:.3f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:  # Jika kelas tidak ada di STATUS_CONFIG
                    st.markdown(
                        f"""
                        <div class="status-alert status-normal">
                           ℹ️ Status Terdeteksi: {status_name} (Kepercayaan: {status_conf_val:.3f})
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.subheader("💡 Rekomendasi Belajar")
                if status_name == "Normal":
                    st.success(
                        "✅ Bagus! Kamu terlihat fokus dan siap menyerap materi. Pertahankan!"
                    )
                elif status_name == "Engaged":
                    st.info(
                        "👀 Keren! Kamu tampak aktif dan terlibat dalam pelajaran. Semangat terus!"
                    )
                elif status_name == "Distracted":
                    st.warning(
                        "⚠️ Hati-hati, kamu mulai terdistraksi. Coba kembalikan fokusmu pada pelajaran, ya!"
                    )
                elif status_name in [
                    "Face Covered",
                    "Face Concealed",
                    "Face Not Visible",
                ]:
                    st.warning(
                        "😷 Wajahmu tidak terlihat jelas oleh kamera. Pastikan posisimu sudah pas agar bisa terpantau."
                    )
                elif status_name == "Object Detected":
                    st.warning(
                        "📱 Ada objek (seperti HP) yang terdeteksi. Singkirkan dulu yuk agar lebih fokus belajar!"
                    )
                elif status_name == "Compromised":
                    st.error(
                        "🚨 PERLU PERHATIAN! Kamu terlihat sangat tidak fokus atau mungkin lelah. Coba istirahat sebentar, minum air, atau regangkan badan."
                    )
            else:
                st.info(
                    "ℹ️ Tidak ada aktivitas atau wajah yang terdeteksi dengan jelas pada gambar ini sesuai ambang batas kepercayaan."
                )
    st.markdown("---")
    st.markdown(
        "© 2024 - Aplikasi Monitor Deteksi Distraksi Siswa (Versi Kustom oleh AI)"
    )  # Tahun diubah


if __name__ == "__main__":
    main()
