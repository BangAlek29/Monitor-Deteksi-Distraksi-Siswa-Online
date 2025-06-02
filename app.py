import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Student Engagement Monitoring System", page_icon="🎓", layout="centered"
)

# CSS untuk styling
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
    .status-engaged { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .status-compromised { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    .status-unknown { background-color: #e2e3e5; color: #383d41; border: 2px solid #d6d8db; }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    
    .header-container {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Mapping status dan warna untuk siswa
STATUS_CONFIG = {
    "engaged": {
        "color": "status-engaged",
        "emoji": "🎯",
        "level": "FOKUS",
        "description": "Siswa fokus mengikuti pembelajaran",
    },
    "compromised": {
        "color": "status-compromised",
        "emoji": "😴",
        "level": "DISTRAKSI",
        "description": "Siswa tidak fokus atau mengantuk",
    },
}


@st.cache_resource
def load_model():
    """Load YOLO model untuk deteksi engagement siswa"""
    try:
        # Pastikan model sudah dilatih untuk mendeteksi engaged dan compromised
        model = YOLO("best.pt")  # Ganti dengan path model yang sesuai
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info(
            "💡 Pastikan file model 'best.pt' tersedia dan sudah dilatih untuk deteksi engagement siswa"
        )
        return None


def get_student_status(results, confidence_threshold=0.5):
    """Ambil status engagement siswa dengan confidence tertinggi"""
    best_detection = None
    max_confidence = 0

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence >= confidence_threshold and confidence > max_confidence:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id].lower()

                    # Hanya terima deteksi engaged atau compromised
                    if class_name in ["engaged", "compromised"]:
                        max_confidence = confidence
                        best_detection = {
                            "status": class_name,
                            "confidence": confidence,
                            "bbox": box.xyxy[0].tolist(),
                        }

    return best_detection


def draw_detection(image, detection):
    """Gambar bounding box pada gambar siswa"""
    if detection is None:
        return image

    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Ambil koordinat
    x1, y1, x2, y2 = map(int, detection["bbox"])
    status = detection["status"]
    confidence = detection["confidence"]

    # Tentukan warna berdasarkan status
    if status == "engaged":
        color = (0, 255, 0)  # Hijau untuk fokus
    else:  # compromised
        color = (0, 0, 255)  # Merah untuk distraksi

    # Gambar bounding box
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)

    # Label
    label = f"{status.upper()}: {confidence:.2f}"

    # Background untuk text
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    cv2.rectangle(img_cv, (x1, y1 - text_height - 15), (x1 + text_width, y1), color, -1)

    # Text
    cv2.putText(
        img_cv, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    # Convert kembali ke RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def main():
    # Header
    st.markdown(
        """
    <div class="header-container">
        <h1>🎓 Student Engagement Monitoring System</h1>
        <p><strong>Sistem Monitoring Keterlibatan Siswa dalam Kelas Daring</strong></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    st.success("✅ Model engagement detector berhasil dimuat!")

    # Sidebar pengaturan
    with st.sidebar:
        st.header("⚙️ Pengaturan Monitoring")
        confidence_threshold = st.slider(
            "Threshold Confidence",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Tingkat kepercayaan minimum untuk deteksi",
        )

        st.markdown("### 📊 Status Pembelajaran")
        st.markdown("🎯 **FOKUS (Engaged)** - Siswa aktif mengikuti pembelajaran")
        st.markdown("😴 **DISTRAKSI (Compromised)** - Siswa tidak fokus/mengantuk")

        st.markdown("### 📈 Tips Monitoring")
        st.info(
            "• Pastikan pencahayaan yang cukup\n• Posisikan kamera di level mata\n• Hindari background yang ramai"
        )

    # Pilihan input
    tab1, tab2, tab3 = st.tabs(
        ["📷 Monitoring Real-time", "🖼️ Upload Foto", "📊 Statistik"]
    )

    with tab1:
        st.header("📷 Monitoring Engagement Real-time")

        col1, col2, col3 = st.columns(3)
        with col1:
            start_btn = st.button(
                "🎥 Mulai Monitoring", type="primary", use_container_width=True
            )
        with col2:
            stop_btn = st.button("⏹️ Stop Monitoring", use_container_width=True)
        with col3:
            capture_btn = st.button("📸 Capture Moment", use_container_width=True)

        # Status display
        status_placeholder = st.empty()

        # Video display
        video_placeholder = st.empty()

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fps_placeholder = st.empty()
        with col2:
            confidence_placeholder = st.empty()
        with col3:
            engagement_time = st.empty()
        with col4:
            timestamp_placeholder = st.empty()

        # Session state untuk webcam
        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False
        if "engaged_count" not in st.session_state:
            st.session_state.engaged_count = 0
        if "total_count" not in st.session_state:
            st.session_state.total_count = 0

        if start_btn:
            st.session_state.webcam_running = True
            st.session_state.engaged_count = 0
            st.session_state.total_count = 0

        if stop_btn:
            st.session_state.webcam_running = False

        # Webcam processing
        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Tidak dapat mengakses webcam. Pastikan kamera tersedia.")
            else:
                frame_count = 0
                start_time = time.time()

                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Gagal membaca frame dari webcam")
                        break

                    # Flip frame untuk mirror effect
                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    # Jalankan deteksi
                    results = model(frame, verbose=False)

                    # Ambil status engagement
                    detection = get_student_status(results, confidence_threshold)

                    # Gambar hasil deteksi
                    detected_image = draw_detection(pil_image, detection)

                    # Tampilkan video
                    video_placeholder.image(
                        detected_image, channels="RGB", use_container_width=True
                    )

                    # Update statistik
                    st.session_state.total_count += 1
                    if detection and detection["status"] == "engaged":
                        st.session_state.engaged_count += 1

                    # Update status
                    if detection:
                        status_name = detection["status"]
                        status_conf = detection["confidence"]

                        if status_name in STATUS_CONFIG:
                            config = STATUS_CONFIG[status_name]
                            status_placeholder.markdown(
                                f"""
                            <div class="status-alert {config['color']}">
                                {config['emoji']} {config['level']}: {config['description']}
                                <br>Confidence: {status_conf:.3f}
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                        confidence_placeholder.metric(
                            "Confidence", f"{status_conf:.3f}"
                        )
                    else:
                        status_placeholder.markdown(
                            """
                        <div class="status-alert status-unknown">
                            🔍 Siswa tidak terdeteksi - Pastikan posisi kamera
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                        confidence_placeholder.metric("Confidence", "0.000")

                    # Update metrics
                    frame_count += 1
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        fps = frame_count / elapsed
                        fps_placeholder.metric("FPS", f"{fps:.1f}")

                    # Engagement rate
                    if st.session_state.total_count > 0:
                        engagement_rate = (
                            st.session_state.engaged_count
                            / st.session_state.total_count
                        ) * 100
                        engagement_time.metric("Engagement", f"{engagement_rate:.1f}%")

                    timestamp_placeholder.metric("Waktu", time.strftime("%H:%M:%S"))

                    # Capture frame jika diminta
                    if capture_btn:
                        timestamp = int(time.time())
                        filename = f"student_capture_{timestamp}.jpg"
                        detected_image.save(filename)
                        st.success(f"📸 Screenshot tersimpan: {filename}")

                    time.sleep(0.03)  # ~30 FPS

                cap.release()

    with tab2:
        st.header("🖼️ Analisis Foto Siswa")

        uploaded_file = st.file_uploader(
            "Upload foto siswa untuk analisis engagement",
            type=["jpg", "jpeg", "png"],
            help="Upload foto siswa saat mengikuti kelas daring",
        )

        if uploaded_file is not None:
            # Load gambar
            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Foto Asli")
                st.image(image, use_container_width=True)

            # Jalankan deteksi
            with st.spinner("🔄 Menganalisis engagement siswa..."):
                results = model(image, verbose=False)
                detection = get_student_status(results, confidence_threshold)
                detected_image = draw_detection(image, detection)

            with col2:
                st.subheader("🎯 Hasil Analisis")
                st.image(detected_image, use_container_width=True)

            # Tampilkan hasil analisis
            if detection:
                status_name = detection["status"]
                status_conf = detection["confidence"]

                if status_name in STATUS_CONFIG:
                    config = STATUS_CONFIG[status_name]
                    st.markdown(
                        f"""
                    <div class="status-alert {config['color']}">
                        {config['emoji']} STATUS: {config['level']}
                        <br>{config['description']}
                        <br>Confidence: {status_conf:.3f}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Rekomendasi berdasarkan status
                st.subheader("💡 Rekomendasi untuk Pembelajaran")
                if status_name == "engaged":
                    st.success(
                        "🎯 Excellent! Siswa menunjukkan engagement yang baik dalam pembelajaran. Pertahankan metode pembelajaran ini."
                    )
                    st.info(
                        "📝 Tips: Berikan pujian dan variasi aktivitas untuk mempertahankan engagement."
                    )
                elif status_name == "compromised":
                    st.error("😴 Perhatian! Siswa tampak tidak fokus atau mengantuk.")
                    st.warning("📝 Saran:")
                    st.write("• Lakukan ice breaking atau energizer")
                    st.write("• Ajukan pertanyaan interaktif")
                    st.write("• Gunakan metode pembelajaran yang lebih engaging")
                    st.write("• Cek apakah siswa memerlukan istirahat")

            else:
                st.info(
                    "ℹ️ Tidak dapat mendeteksi status engagement dengan confidence yang cukup tinggi. Pastikan wajah siswa terlihat jelas."
                )

    with tab3:
        st.header("📊 Statistik Engagement")

        if "total_count" in st.session_state and st.session_state.total_count > 0:
            col1, col2, col3 = st.columns(3)

            engagement_rate = (
                st.session_state.engaged_count / st.session_state.total_count
            ) * 100
            distraction_rate = 100 - engagement_rate

            with col1:
                st.metric("Total Deteksi", st.session_state.total_count)
            with col2:
                st.metric(
                    "Engagement Rate",
                    f"{engagement_rate:.1f}%",
                    delta=f"+{st.session_state.engaged_count}",
                )
            with col3:
                st.metric("Distraction Rate", f"{distraction_rate:.1f}%")

            # Progress bar
            st.subheader("📈 Tingkat Engagement")
            st.progress(engagement_rate / 100)

            # Interpretasi
            if engagement_rate >= 80:
                st.success("🏆 Excellent! Tingkat engagement sangat baik.")
            elif engagement_rate >= 60:
                st.info("👍 Good! Tingkat engagement cukup baik.")
            elif engagement_rate >= 40:
                st.warning("⚠️ Moderate. Perlu peningkatan engagement.")
            else:
                st.error(
                    "🚨 Poor. Diperlukan intervensi untuk meningkatkan engagement."
                )
        else:
            st.info("📊 Mulai monitoring untuk melihat statistik engagement.")
            st.markdown("### 📋 Cara Menggunakan:")
            st.write("1. Klik tab 'Monitoring Real-time'")
            st.write("2. Klik 'Mulai Monitoring'")
            st.write("3. Biarkan sistem menganalisis engagement siswa")
            st.write("4. Kembali ke tab ini untuk melihat statistik")


if __name__ == "__main__":
    main()
