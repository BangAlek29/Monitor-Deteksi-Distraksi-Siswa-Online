import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    RTCConfiguration,
    WebRtcMode,
)

# --- Konfigurasi Halaman dan CSS ---
st.set_page_config(
    page_title="Monitor Deteksi Distraksi Siswa",
    page_icon="🧑‍🎓",
    layout="centered",
)

st.markdown(
    """
<style>
    .status-alert {
        padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;
        font-weight: bold; text-align: center; font-size: 1.2rem;
    }
    .status-normal { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .status-engaged { background-color: #cce5ff; color: #004085; border: 2px solid #99d6ff; }
    .status-warning { background-color: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
    .status-danger { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    .metric-card {
        background: #f8f9fa; padding: 1rem; border-radius: 0.5rem;
        border: 1px solid #dee2e6; text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Konfigurasi Status ---
STATUS_CONFIG = {
    "Normal": {"color": "status-normal", "emoji": "✅", "level": "FOKUS"},
    "Engaged": {"color": "status-engaged", "emoji": "👀", "level": "AKTIF BELAJAR"},
    "Distracted": {"color": "status-warning", "emoji": "⚠️", "level": "TERDISTRAKSI"},
    "Compromised": {
        "color": "status-danger",
        "emoji": "🚨",
        "level": "PERLU PERHATIAN",
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
        "emoji": "📱",
        "level": "OBJEK TERDETEKSI",
    },
}


# --- Fungsi Model dan Deteksi ---
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model 'best.pt': {e}")
        return None


MODEL = load_yolo_model()


def get_detection_status(results, confidence_threshold=0.5):
    best_detection = None
    max_confidence = 0
    if results:  # Memastikan results tidak None atau kosong
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    if (
                        confidence >= confidence_threshold
                        and confidence > max_confidence
                    ):
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        max_confidence = confidence
                        best_detection = {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": box.xyxy[0].tolist(),
                        }
    return best_detection


def draw_bounding_box(image_pil, detection_info):
    if detection_info is None:
        return image_pil

    img_array_rgb = np.array(image_pil.convert("RGB"))
    img_cv_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2 = map(int, detection_info["bbox"])
    class_name = detection_info["class"]
    confidence = detection_info["confidence"]

    color_map = {
        "Normal": (0, 255, 0),
        "Engaged": (0, 255, 0),  # Hijau
        "Distracted": (0, 165, 255),
        "Face Covered": (0, 165, 255),  # Oranye
        "Face Concealed": (0, 165, 255),
        "Face Not Visible": (0, 165, 255),
        "Object Detected": (0, 165, 255),
        "Compromised": (0, 0, 255),  # Merah
    }
    box_color = color_map.get(class_name, (255, 0, 0))  # Default Biru

    cv2.rectangle(
        img_cv_bgr, (x1, y1), (x2, y2), box_color, 2
    )  # Ketebalan garis diubah ke 2
    label = f"{class_name}: {confidence:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )  # Ukuran font diubah
    cv2.rectangle(
        img_cv_bgr,
        (x1, y1 - text_height - 10),
        (x1 + text_width, y1 - 5),
        box_color,
        -1,
    )
    cv2.putText(
        img_cv_bgr,
        label,
        (x1, y1 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    img_array_processed_rgb = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_array_processed_rgb)


# --- Video Transformer untuk streamlit-webrtc ---
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self, confidence_threshold_val):
        self.confidence_threshold = confidence_threshold_val
        self.model = MODEL  # Gunakan model yang sudah di-load global
        self.latest_detection_info = {
            "status_name": "Inisialisasi...",
            "status_conf_val": 0.0,
            "status_level": "MEMUAT",
            "status_emoji": "🔄",
            "status_color_class": "status-normal",  # Warna CSS class
            "fps": 0.0,
        }
        self._frame_count_for_fps = 0
        self._start_time_for_fps = time.time()

    def update_fps(self):
        self._frame_count_for_fps += 1
        elapsed_time = time.time() - self._start_time_for_fps
        if elapsed_time > 1.0:  # Update FPS kira-kira setiap 1 detik
            self.latest_detection_info["fps"] = self._frame_count_for_fps / elapsed_time
            self._frame_count_for_fps = 0
            self._start_time_for_fps = time.time()

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image_input = Image.fromarray(img_rgb)

        detection_result = None
        processed_pil_image = pil_image_input  # Default jika model tidak ada

        if self.model:
            results = self.model(
                pil_image_input, verbose=False, half=True
            )  # Menggunakan pil_image_input
            detection_result = get_detection_status(results, self.confidence_threshold)
            processed_pil_image = draw_bounding_box(pil_image_input, detection_result)

        # Update info deteksi terakhir
        if detection_result:
            class_name = detection_result["class"]
            conf_val = detection_result["confidence"]
            self.latest_detection_info["status_name"] = class_name
            self.latest_detection_info["status_conf_val"] = conf_val
            if class_name in STATUS_CONFIG:
                config = STATUS_CONFIG[class_name]
                self.latest_detection_info["status_level"] = config["level"]
                self.latest_detection_info["status_emoji"] = config["emoji"]
                self.latest_detection_info["status_color_class"] = config["color"]
            else:  # Kelas tidak dikenal
                self.latest_detection_info["status_level"] = "TIDAK DIKENAL"
                self.latest_detection_info["status_emoji"] = "❓"
                self.latest_detection_info["status_color_class"] = "status-warning"
        else:
            self.latest_detection_info["status_name"] = "Tidak Ada Deteksi"
            self.latest_detection_info["status_conf_val"] = 0.0
            self.latest_detection_info["status_level"] = "STANDBY"
            self.latest_detection_info["status_emoji"] = "☕"
            self.latest_detection_info["status_color_class"] = "status-normal"

        self.update_fps()

        img_out_rgb = np.array(processed_pil_image.convert("RGB"))
        img_out_bgr = cv2.cvtColor(img_out_rgb, cv2.COLOR_RGB2BGR)
        return frame.from_ndarray(img_out_bgr, format="bgr24")


# --- Aplikasi Streamlit Utama ---
def main_app():
    st.title("🧑‍🎓 Monitor Deteksi Distraksi Siswa")
    st.markdown(
        "**Pantau fokus belajarmu secara real-time!** Aplikasi ini membantu mendeteksi tingkat fokus saat belajar online."
    )
    st.markdown("---")

    if MODEL is None:
        st.error("❌ Model pendeteksi gagal dimuat. Aplikasi tidak dapat berjalan.")
        st.stop()
    st.success("✅ Model pendeteksi berhasil dimuat!")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        confidence_threshold_slider = st.slider(
            "Ambang Batas Kepercayaan (Confidence)",
            0.3,
            0.9,
            0.6,
            0.05,
            help="Semakin tinggi, deteksi semakin selektif.",
        )
        st.markdown("---")
        st.markdown("### 📊 Legenda Status")
        for status_key, config_val in STATUS_CONFIG.items():
            st.markdown(
                f"{config_val['emoji']} **{status_key}** - {config_val['level']}"
            )
        st.markdown("---")
        st.info("Webcam diakses via browser Anda. Izinkan akses jika diminta.")

    # --- Tabs ---
    tab_webcam, tab_upload = st.tabs(["📷 Kamera Real-time", "🖼️ Unggah Gambar"])

    # --- Tab 1: Kamera Real-time ---
    with tab_webcam:
        st.header("🎥 Monitoring Real-time via Kamera")
        st.write(
            "Klik 'START' di bawah untuk mengaktifkan kamera Anda. Izinkan akses di browser jika diminta."
        )

        # Placeholders untuk metrik yang diupdate dari transformer
        status_col, conf_col, fps_col, time_col = st.columns(4)
        ui_status_placeholder = status_col.empty()
        ui_confidence_placeholder = conf_col.empty()
        ui_fps_placeholder = fps_col.empty()
        ui_timestamp_placeholder = time_col.empty()

        # Inisialisasi placeholder dengan nilai default
        ui_status_placeholder.markdown(
            f"""<div class="status-alert status-normal">🔄 Menunggu Kamera...</div>""",
            unsafe_allow_html=True,
        )
        ui_confidence_placeholder.metric("Kepercayaan", "N/A")
        ui_fps_placeholder.metric("FPS", "N/A")
        ui_timestamp_placeholder.metric("Waktu", time.strftime("%H:%M:%S"))

        rtc_config = RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun.cloudflare.com:3478"]},
                    # Anda bisa menambahkan lebih banyak server STUN publik jika ada
                ]
            }
        )

        # Factory untuk membuat instance transformer dengan confidence threshold terbaru
        def video_transformer_factory():
            return YOLOVideoTransformer(
                confidence_threshold_val=confidence_threshold_slider
            )

        webrtc_ctx = webrtc_streamer(
            key="student-distraction-monitor",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_transformer_factory=video_transformer_factory,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            # desired_playing_state=st.session_state.get("play_webrtc", False) # Jika ingin kontrol manual Start/Stop
        )

        # Tombol info (opsional, untuk debug atau info cepat)
        # if st.button("ℹ️ Info Deteksi Terakhir (Webcam)", use_container_width=True):
        #     if webrtc_ctx.video_transformer:
        #         info = webrtc_ctx.video_transformer.latest_detection_info
        #         st.toast(f"Status: {info['status_name']} ({info['status_level']}) @ {info['status_conf_val']:.2f} | FPS: {info['fps']:.1f}", icon=info['status_emoji'])
        #     else:
        #         st.toast("Kamera belum aktif atau transformer belum siap.", icon="⏳")

        # Loop untuk update UI dari transformer jika sedang berjalan
        if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
            while True:  # Loop ini akan berjalan selama webrtc aktif
                info = webrtc_ctx.video_transformer.latest_detection_info
                ui_status_placeholder.markdown(
                    f"""
                    <div class="status-alert {info['status_color_class']}">
                        {info['status_emoji']} {info['status_level']} ({info['status_name']})
                        <br>Kepercayaan: {info['status_conf_val']:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                ui_confidence_placeholder.metric(
                    "Kepercayaan",
                    (
                        f"{info['status_conf_val']:.3f}"
                        if info["status_conf_val"] > 0
                        else "N/A"
                    ),
                )
                ui_fps_placeholder.metric("FPS", f"{info['fps']:.1f}")
                ui_timestamp_placeholder.metric("Waktu", time.strftime("%H:%M:%S"))

                # Periksa apakah streaming masih berjalan, jika tidak, keluar dari loop update UI
                if not webrtc_ctx.state.playing:
                    # Set ke nilai default saat streaming berhenti
                    ui_status_placeholder.markdown(
                        f"""<div class="status-alert status-normal">⏹️ Kamera Berhenti</div>""",
                        unsafe_allow_html=True,
                    )
                    ui_confidence_placeholder.metric("Kepercayaan", "N/A")
                    ui_fps_placeholder.metric("FPS", "N/A")
                    break
                time.sleep(
                    0.1
                )  # Frekuensi update UI, jangan terlalu cepat agar tidak membebani
        # else:
        # Jika tidak streaming, pastikan placeholder menunjukkan status non-aktif
        # ui_status_placeholder.markdown(f"""<div class="status-alert status-normal">🎬 Klik 'START' pada video player di atas</div>""", unsafe_allow_html=True)

    # --- Tab 2: Unggah Gambar ---
    with tab_upload:
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
            image_pil_uploaded = Image.open(uploaded_file)

            st.subheader("🖼️ Pratinjau Gambar")
            col_img_orig, col_img_proc = st.columns(2)
            with col_img_orig:
                st.caption("Gambar Asli")
                st.image(image_pil_uploaded, use_container_width=True)

            with st.spinner(
                "🧠 Menganalisis gambar... Ini mungkin butuh beberapa detik."
            ):
                results_uploaded = MODEL(image_pil_uploaded, verbose=False, half=True)
                detection_uploaded = get_detection_status(
                    results_uploaded, confidence_threshold_slider
                )
                processed_image_uploaded = draw_bounding_box(
                    image_pil_uploaded.copy(), detection_uploaded
                )

            with col_img_proc:
                st.caption("Hasil Deteksi")
                st.image(processed_image_uploaded, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Hasil Analisis Detail")
            if detection_uploaded:
                class_name_up = detection_uploaded["class"]
                conf_val_up = detection_uploaded["confidence"]
                config_up = STATUS_CONFIG.get(class_name_up)

                if config_up:
                    st.markdown(
                        f"""
                        <div class="status-alert {config_up['color']}">
                            {config_up['emoji']} STATUS: {config_up['level']} ({class_name_up})
                            <br>Kepercayaan: {conf_val_up:.3f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:  # Kelas tidak dikenal
                    st.markdown(
                        f"""<div class="status-alert status-warning">❓ STATUS: {class_name_up} (Kepercayaan: {conf_val_up:.3f})</div>""",
                        unsafe_allow_html=True,
                    )

                # Rekomendasi
                st.subheader("💡 Rekomendasi Belajar")
                if class_name_up == "Normal":
                    st.success("✅ Bagus! Kamu terlihat fokus. Pertahankan!")
                elif class_name_up == "Engaged":
                    st.info("👀 Keren! Kamu tampak aktif. Semangat terus!")
                elif class_name_up == "Distracted":
                    st.warning("⚠️ Kamu mulai terdistraksi. Ayo fokus lagi!")
                elif class_name_up in [
                    "Face Covered",
                    "Face Concealed",
                    "Face Not Visible",
                ]:
                    st.warning(
                        "😷 Wajahmu tidak terlihat jelas. Pastikan posisi kamera pas."
                    )
                elif class_name_up == "Object Detected":
                    st.warning("📱 Ada objek (mis. HP) terdeteksi. Singkirkan dulu ya!")
                elif class_name_up == "Compromised":
                    st.error(
                        "🚨 PERLU PERHATIAN! Kamu terlihat sangat tidak fokus. Coba istirahat sebentar."
                    )
            else:
                st.info(
                    "ℹ️ Tidak ada aktivitas atau wajah yang terdeteksi dengan jelas pada gambar ini."
                )

    st.markdown("---")
    st.markdown("© 2025 - Aplikasi Monitor Deteksi Distraksi Siswa")


if __name__ == "__main__":
    main_app()
