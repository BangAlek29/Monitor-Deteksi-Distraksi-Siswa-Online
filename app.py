import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import base64
from io import BytesIO

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
    
    .webcam-container {
        background: #f8f9fa;
        border-radius: 1rem;
        padding: 1rem;
        border: 2px solid #e9ecef;
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
    
    .control-button {
        width: 100%;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border: none;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .btn-start {
        background: #10B981;
        color: white;
    }
    
    .btn-stop {
        background: #EF4444;
        color: white;
    }
    
    .instructions {
        background: #f0f9ff;
        border: 1px solid #0ea5e9;
        border-radius: 0.8rem;
        padding: 1.5rem;
        margin: 1rem 0;
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
    # Logika sederhana: deteksi wajah = engaged, tidak ada deteksi = compromised
    person_detected = False
    max_confidence = 0

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

    if person_detected:
        return {"status": "Engaged", "confidence": max_confidence, "detected": True}
    else:
        return {"status": "Compromised", "confidence": 0.0, "detected": False}


def create_camera_interface():
    """Buat interface kamera yang simpel"""
    st.markdown(
        """
    <script>
    async function startCamera() {
        try {
            const video = document.getElementById('webcam');
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.play();
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Tidak dapat mengakses kamera. Pastikan izin kamera telah diberikan.');
        }
    }
    
    function stopCamera() {
        const video = document.getElementById('webcam');
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
    }
    </script>
    
    <div class="webcam-container">
        <video id="webcam" width="100%" height="400" style="border-radius: 0.5rem; background: #000;"></video>
        <div style="text-align: center; margin-top: 1rem;">
            <button onclick="startCamera()" class="control-button btn-start">🎥 Mulai Monitoring</button>
            <button onclick="stopCamera()" class="control-button btn-stop">⏹️ Stop Monitoring</button>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
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

        # Instruksi penggunaan
        st.markdown(
            """
        <div class="instructions">
            <h4>📋 Cara Penggunaan:</h4>
            <ol>
                <li>Klik tombol "Mulai Monitoring" untuk mengaktifkan kamera</li>
                <li>Pastikan wajah Anda terlihat jelas di kamera</li>
                <li>Sistem akan mendeteksi apakah Anda fokus atau tidak</li>
                <li>Status akan ditampilkan secara real-time</li>
            </ol>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Interface kamera
        camera_placeholder = st.empty()

        # Simulasi interface kamera web-based
        with camera_placeholder.container():
            st.markdown(
                """
            <div class="webcam-container">
                <div style="background: #1a1a1a; height: 400px; border-radius: 0.5rem; display: flex; align-items: center; justify-content: center; color: white; flex-direction: column;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📹</div>
                    <h3>Camera Feed Area</h3>
                    <p>Kamera akan aktif saat di-deploy di web</p>
                    <p style="font-size: 0.9rem; opacity: 0.7;">Demo: Upload gambar di panel sebelah kanan</p>
                </div>
                <div style="text-align: center; margin-top: 1rem;">
                    <div style="display: inline-block; margin: 0 0.5rem;">
                        <button style="background: #10B981; color: white; padding: 0.8rem 2rem; border: none; border-radius: 0.5rem; font-weight: bold; cursor: pointer;">
                            🎥 Mulai Monitoring
                        </button>
                    </div>
                    <div style="display: inline-block; margin: 0 0.5rem;">
                        <button style="background: #EF4444; color: white; padding: 0.8rem 2rem; border: none; border-radius: 0.5rem; font-weight: bold; cursor: pointer;">
                            ⏹️ Stop Monitoring
                        </button>
                    </div>
                </div>
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

        # Demo upload untuk testing
        st.markdown("### 🖼️ Test Upload")
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

    # Status display area
    st.markdown("---")
    status_col1, status_col2 = st.columns([1, 1])

    with status_col1:
        st.markdown(
            """
        <div class="status-card status-monitoring">
            🔍 SISTEM STANDBY
            <br><small>Menunggu aktivasi kamera...</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with status_col2:
        # Metrics
        st.markdown(
            """
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-value">0</div>
                <div class="metric-label">Sesi Fokus</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">00:00</div>
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
    <div style="text-align: center; color: #6b7280; padding: 2rem;">
        <h4>💡 Informasi Deployment</h4>
        <p>Untuk deployment di web:</p>
        <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
            <li>Gunakan Streamlit Cloud, Heroku, atau platform hosting lainnya</li>
            <li>Pastikan requirements.txt berisi: streamlit, opencv-python-headless, ultralytics, pillow</li>
            <li>Upload model YOLO (best.pt) ke repository</li>
            <li>Kamera akan otomatis terakses melalui browser</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
