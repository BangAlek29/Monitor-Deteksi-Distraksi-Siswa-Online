import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Student Attention Monitor", page_icon="🎓", layout="wide"
)

# CSS untuk styling yang modern dan simpel
st.markdown(
    """
<style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .status-card {
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .status-engaged { 
        background: linear-gradient(135deg, #10B981, #34D399);
        color: white;
        border: 3px solid #059669;
    }
    
    .status-compromised { 
        background: linear-gradient(135deg, #EF4444, #F87171);
        color: white;
        border: 3px solid #DC2626;
    }
    
    .status-monitoring { 
        background: linear-gradient(135deg, #3B82F6, #60A5FA);
        color: white;
        border: 3px solid #2563EB;
    }
    
    .status-standby { 
        background: linear-gradient(135deg, #8B5CF6, #A78BFA);
        color: white;
        border: 3px solid #7C3AED;
    }
    
    .webcam-container {
        background: #f8f9fa;
        border-radius: 1rem;
        padding: 1rem;
        border: 2px solid #e9ecef;
        text-align: center;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-item {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f2937;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .header-title {
        text-align: center;
        color: #1f2937;
        margin-bottom: 2rem;
    }
    
    .instructions {
        background: #f0f9ff;
        border: 1px solid #0ea5e9;
        border-radius: 0.8rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .camera-info {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 0.8rem;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Status configuration - hanya 2 status utama
STATUS_CONFIG = {
    "Engaged": {
        "color": "status-engaged",
        "emoji": "✅",
        "level": "FOKUS",
        "message": "Siswa sedang memperhatikan pelajaran",
    },
    "Compromised": {
        "color": "status-compromised",
        "emoji": "⚠️",
        "level": "TIDAK FOKUS",
        "message": "Siswa tidak memperhatikan atau teralihkan",
    },
}


@st.cache_resource
def load_model():
    """Load YOLO model"""
    try:
        # Gunakan model YOLOv8 default atau model custom Anda
        model = YOLO("yolov8n.pt")  # Ganti dengan "best.pt" jika ada model custom
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 Pastikan file model 'best.pt' atau 'yolov8n.pt' tersedia")
        return None


def classify_attention(results, confidence_threshold=0.5):
    """Klasifikasi tingkat perhatian siswa berdasarkan deteksi"""
    person_detected = False
    max_confidence = 0
    face_detected = False

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # Class 0 biasanya adalah 'person' di YOLO
                if class_id == 0 and confidence >= confidence_threshold:
                    person_detected = True
                    max_confidence = max(max_confidence, confidence)

                    # Jika deteksi person dengan confidence tinggi, anggap sebagai engaged
                    if confidence > 0.7:
                        face_detected = True

    if person_detected and face_detected:
        return {"status": "Engaged", "confidence": max_confidence, "detected": True}
    elif person_detected:
        return {"status": "Engaged", "confidence": max_confidence, "detected": True}
    else:
        return {"status": "Compromised", "confidence": 0.0, "detected": False}


def draw_detection_boxes(image, results, confidence_threshold=0.5):
    """Gambar bounding box pada deteksi"""
    img_array = np.array(image)

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                if confidence >= confidence_threshold:
                    # Koordinat bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Warna berdasarkan class
                    if class_id == 0:  # person
                        color = (0, 255, 0) if confidence > 0.7 else (255, 255, 0)
                    else:
                        color = (255, 0, 0)

                    # Gambar rectangle
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)

                    # Label
                    label = f"Person: {confidence:.2f}"
                    cv2.putText(
                        img_array,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

    return Image.fromarray(img_array)


def main():
    # Initialize session state
    if "focus_sessions" not in st.session_state:
        st.session_state.focus_sessions = 0
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "last_status" not in st.session_state:
        st.session_state.last_status = "Standby"

    # Header
    st.markdown(
        """
    <div class="header-title">
        <h1>🎓 Student Attention Monitor</h1>
        <p style="font-size: 1.2rem; color: #6b7280;">Sistem Monitoring Perhatian Siswa untuk Pembelajaran Online</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # Layout utama
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📷 Live Camera Feed")

        # Webcam input menggunakan Streamlit
        camera_input = st.camera_input("📸 Ambil foto untuk monitoring")

        if camera_input is not None:
            # Convert uploaded image
            image = Image.open(camera_input)

            # Jalankan deteksi
            with st.spinner("🔄 Analyzing..."):
                results = model(image, verbose=False)
                detection = classify_attention(results, confidence_threshold=0.6)

                # Gambar bounding box
                processed_image = draw_detection_boxes(
                    image, results, confidence_threshold=0.6
                )

            # Tampilkan hasil
            col_img1, col_img2 = st.columns(2)

            with col_img1:
                st.markdown("**📷 Original**")
                st.image(image, use_container_width=True)

            with col_img2:
                st.markdown("**🎯 Detection Result**")
                st.image(processed_image, use_container_width=True)

            # Update status
            status = detection["status"]
            config = STATUS_CONFIG[status]

            # Update session tracking
            if st.session_state.last_status != status:
                if status == "Engaged":
                    st.session_state.focus_sessions += 1
                st.session_state.last_status = status

            if st.session_state.start_time is None:
                st.session_state.start_time = time.time()

            # Display status
            st.markdown(
                f"""
            <div class="status-card {config['color']}">
                {config['emoji']} {config['level']}
                <br><small>{config['message']}</small>
                <br><small>Confidence: {detection['confidence']:.3f}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

        else:
            # Instruksi penggunaan
            st.markdown(
                """
            <div class="instructions">
                <h4>📋 Cara Penggunaan:</h4>
                <ol>
                    <li>Klik tombol "Take Photo" di atas untuk mengaktifkan kamera</li>
                    <li>Posisikan wajah Anda di depan kamera</li>
                    <li>Tekan tombol untuk mengambil foto</li>
                    <li>Sistem akan menganalisis tingkat perhatian Anda</li>
                    <li>Ulangi proses untuk monitoring berkelanjutan</li>
                </ol>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
            <div class="camera-info">
                <h4>📷 Camera Ready</h4>
                <p>Klik tombol "Take Photo" untuk memulai monitoring</p>
                <p>Pastikan pencahayaan cukup dan wajah terlihat jelas</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### ⚙️ Control Panel")

        # Settings
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Tingkat kepercayaan minimum untuk deteksi",
        )

        # Status legend
        st.markdown("### 📊 Status Legend")
        st.markdown(
            """
        - ✅ **FOKUS** - Siswa memperhatikan pelajaran
        - ⚠️ **TIDAK FOKUS** - Siswa teralihkan atau tidak terlihat
        """
        )

        # Reset button
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.focus_sessions = 0
            st.session_state.start_time = None
            st.session_state.last_status = "Standby"
            st.success("Session direset!")
            st.rerun()

        # File upload untuk testing
        st.markdown("### 🖼️ Upload Test Image")
        uploaded_file = st.file_uploader(
            "Upload foto untuk testing",
            type=["jpg", "jpeg", "png"],
            help="Upload foto siswa untuk testing sistem",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Test Image", use_container_width=True)

            with st.spinner("🔄 Analyzing..."):
                results = model(image, verbose=False)
                detection = classify_attention(results, confidence_threshold)

            # Display result
            status = detection["status"]
            config = STATUS_CONFIG[status]

            st.markdown(
                f"""
            <div class="status-card {config['color']}">
                {config['emoji']} {config['level']}
                <br><small>{config['message']}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Status display area dan metrics
    st.markdown("---")
    status_col1, status_col2 = st.columns([1, 1])

    with status_col1:
        if camera_input is not None or uploaded_file is not None:
            # Menampilkan status terkini
            pass  # Status sudah ditampilkan di atas
        else:
            st.markdown(
                """
            <div class="status-card status-standby">
                🔍 SISTEM STANDBY
                <br><small>Menunggu input kamera...</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with status_col2:
        # Metrics
        if st.session_state.start_time:
            elapsed_time = int(time.time() - st.session_state.start_time)
            minutes, seconds = divmod(elapsed_time, 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
        else:
            time_str = "00:00"

        st.markdown(
            f"""
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-value">{st.session_state.focus_sessions}</div>
                <div class="metric-label">Sesi Fokus</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{time_str}</div>
                <div class="metric-label">Waktu Monitoring</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Footer information
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <h4>💡 Tips Penggunaan</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; max-width: 800px; margin: 0 auto;">
            <div style="background: #f9fafb; padding: 1rem; border-radius: 0.5rem;">
                <strong>📷 Kamera</strong><br>
                Pastikan pencahayaan cukup dan wajah terlihat jelas
            </div>
            <div style="background: #f9fafb; padding: 1rem; border-radius: 0.5rem;">
                <strong>🎯 Posisi</strong><br>
                Posisikan wajah di tengah frame kamera
            </div>
            <div style="background: #f9fafb; padding: 1rem; border-radius: 0.5rem;">
                <strong>⏱️ Monitoring</strong><br>
                Ambil foto secara berkala untuk tracking yang akurat
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
