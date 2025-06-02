import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Konfigurasi halaman
st.set_page_config(page_title="Webcam Monitor", page_icon="📷", layout="centered")

# CSS sederhana
st.markdown(
    """
<style>
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .status-normal { background-color: #d4edda; color: #155724; }
    .status-detected { background-color: #fff3cd; color: #856404; }
</style>
""",
    unsafe_allow_html=True,
)


# Video Transformer sederhana
class SimpleVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.status = "Normal"

    def recv(self, frame):
        # Konversi frame
        img = frame.to_ndarray(format="bgr24")

        # Deteksi wajah sederhana menggunakan Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Gambar kotak di sekitar wajah
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            self.status = "Face Detected"

        if len(faces) == 0:
            self.status = "No Face"
            cv2.putText(
                img,
                "No Face Detected",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Tambah frame counter
        self.frame_count += 1
        cv2.putText(
            img,
            f"Frame: {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return frame.from_ndarray(img, format="bgr24")


# Aplikasi utama
def main():
    st.title("📷 Simple Webcam Monitor")
    st.write("Aplikasi sederhana untuk mengakses webcam dan deteksi wajah")

    # Status placeholder
    status_placeholder = st.empty()

    # Konfigurasi WebRTC
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="simple-webcam",
        video_transformer_factory=SimpleVideoTransformer,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Update status
    if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
        status = webrtc_ctx.video_transformer.status
        frame_count = webrtc_ctx.video_transformer.frame_count

        if status == "Face Detected":
            status_placeholder.markdown(
                f'<div class="status-box status-detected">✅ Wajah Terdeteksi | Frame: {frame_count}</div>',
                unsafe_allow_html=True,
            )
        else:
            status_placeholder.markdown(
                f'<div class="status-box status-normal">❌ Tidak Ada Wajah | Frame: {frame_count}</div>',
                unsafe_allow_html=True,
            )
    else:
        status_placeholder.markdown(
            '<div class="status-box status-normal">📷 Klik START untuk mengaktifkan kamera</div>',
            unsafe_allow_html=True,
        )

    # Tab untuk upload gambar
    st.markdown("---")
    st.subheader("🖼️ Upload Gambar untuk Test")

    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Konversi ke BGR untuk OpenCV
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        # Deteksi wajah
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Gambar kotak
        for x, y, w, h in faces:
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Konversi kembali ke RGB untuk display
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Tampilkan hasil
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
        with col2:
            st.image(
                img_rgb,
                caption=f"Hasil Deteksi ({len(faces)} wajah)",
                use_container_width=True,
            )

        if len(faces) > 0:
            st.success(f"✅ Terdeteksi {len(faces)} wajah")
        else:
            st.warning("❌ Tidak ada wajah yang terdeteksi")


if __name__ == "__main__":
    main()
