import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os

class RealtimeAttentionMonitor:
    def __init__(self, model_path="best.pt", confidence_threshold=0.6):
        """
        Initialize the realtime attention monitor
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.cap = None
        self.running = False
        
        # Status tracking
        self.current_status = "Standby"
        self.status_history = deque(maxlen=100)  # Keep last 100 status updates
        self.focus_sessions = 0
        self.total_focused_time = 0
        self.session_start_time = None
        self.last_status_change = time.time()
        
        # Performance metrics
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        
        # Colors for different statuses
        self.colors = {
            "Engaged": (0, 255, 0),      # Green
            "Compromised": (0, 0, 255),  # Red
            "Standby": (128, 128, 128)   # Gray
        }
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Trying to load default YOLOv8n model...")
            try:
                self.model = YOLO("yolov8n.pt")
                print("✅ Default YOLOv8n model loaded")
            except Exception as e2:
                print(f"❌ Failed to load any model: {e2}")
                self.model = None
    
    def classify_attention(self, results):
        """Classify attention level based on detection results"""
        person_detected = False
        max_confidence = 0
        face_area = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Class 0 is usually 'person' in YOLO
                    if class_id == 0 and confidence >= self.confidence_threshold:
                        person_detected = True
                        max_confidence = max(max_confidence, confidence)
                        
                        # Calculate face area (as a proxy for attention)
                        x1, y1, x2, y2 = box.xyxy[0]
                        area = (x2 - x1) * (y2 - y1)
                        face_area = max(face_area, area)
        
        # Determine status based on detection
        if person_detected and max_confidence > 0.7:
            return {
                "status": "Engaged",
                "confidence": max_confidence,
                "detected": True,
                "area": face_area
            }
        elif person_detected:
            return {
                "status": "Engaged",
                "confidence": max_confidence,
                "detected": True,
                "area": face_area
            }
        else:
            return {
                "status": "Compromised",
                "confidence": 0.0,
                "detected": False,
                "area": 0
            }
    
    def draw_interface(self, frame, detection_result):
        """Draw monitoring interface on frame"""
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Status panel
        status = detection_result["status"]
        confidence = detection_result["confidence"]
        
        # Draw status panel background
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        status_color = self.colors[status]
        cv2.putText(frame, f"Status: {status}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        if detection_result["detected"]:
            cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Session metrics
        if self.session_start_time:
            elapsed = time.time() - self.session_start_time
            minutes, seconds = divmod(int(elapsed), 60)
            cv2.putText(frame, f"Session: {minutes:02d}:{seconds:02d}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Focus Sessions: {self.focus_sessions}", (250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # FPS counter
        if len(self.fps_counter) > 1:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions panel
        instructions = [
            "Press 'q' to quit",
            "Press 'r' to reset session",
            "Press 's' to save screenshot",
            "Press 'c' to change confidence threshold"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, height - 80 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_detection_boxes(self, frame, results):
        """Draw detection bounding boxes"""
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if confidence >= self.confidence_threshold and class_id == 0:
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Choose color based on confidence
                        if confidence > 0.8:
                            color = (0, 255, 0)  # High confidence - Green
                        elif confidence > 0.6:
                            color = (0, 255, 255)  # Medium confidence - Yellow
                        else:
                            color = (0, 165, 255)  # Low confidence - Orange
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"Person: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def update_status(self, new_status):
        """Update attention status and track changes"""
        current_time = time.time()
        
        if new_status != self.current_status:
            # Status changed
            if new_status == "Engaged" and self.current_status != "Engaged":
                self.focus_sessions += 1
                print(f"📈 Focus session #{self.focus_sessions} started")
            
            self.current_status = new_status
            self.last_status_change = current_time
            
        # Add to history
        self.status_history.append({
            "timestamp": current_time,
            "status": new_status,
            "session": self.focus_sessions
        })
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attention_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
        return filename
    
    def start_monitoring(self, camera_index=0):
        """Start realtime monitoring"""
        if self.model is None:
            print("❌ No model loaded. Cannot start monitoring.")
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎥 Starting realtime monitoring...")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 'r' to reset session")
        print("   - Press 's' to save screenshot")
        print("   - Press 'c' to change confidence threshold")
        print("   - Press SPACE to pause/resume")
        
        self.running = True
        self.session_start_time = time.time()
        paused = False
        
        try:
            while self.running:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("❌ Failed to read frame from camera")
                        break
                    
                    # Update FPS counter
                    current_time = time.time()
                    self.fps_counter.append(current_time)
                    
                    # Run detection
                    results = self.model(frame, verbose=False)
                    detection_result = self.classify_attention(results)
                    
                    # Update status
                    self.update_status(detection_result["status"])
                    
                    # Draw detection boxes
                    frame = self.draw_detection_boxes(frame, results)
                    
                    # Draw interface
                    frame = self.draw_interface(frame, detection_result)
                    
                    # Show frame
                    cv2.imshow('Student Attention Monitor - Realtime', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 Stopping monitoring...")
                    break
                elif key == ord('r'):
                    self.reset_session()
                    print("🔄 Session reset")
                elif key == ord('s'):
                    if not paused:
                        self.save_screenshot(frame)
                elif key == ord('c'):
                    self.change_confidence_threshold()
                elif key == ord(' '):  # Space key
                    paused = not paused
                    status_text = "⏸️ PAUSED" if paused else "▶️ RESUMED"
                    print(f"{status_text}")
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring interrupted by user")
        
        finally:
            self.stop_monitoring()
    
    def change_confidence_threshold(self):
        """Interactive confidence threshold change"""
        print(f"\n🎯 Current confidence threshold: {self.confidence_threshold}")
        try:
            new_threshold = float(input("Enter new threshold (0.1-0.9): "))
            if 0.1 <= new_threshold <= 0.9:
                self.confidence_threshold = new_threshold
                print(f"✅ Confidence threshold updated to: {new_threshold}")
            else:
                print("❌ Invalid threshold. Must be between 0.1 and 0.9")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    def reset_session(self):
        """Reset monitoring session"""
        self.focus_sessions = 0
        self.total_focused_time = 0
        self.session_start_time = time.time()
        self.status_history.clear()
        self.current_status = "Standby"
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        if self.session_start_time:
            total_time = time.time() - self.session_start_time
            minutes, seconds = divmod(int(total_time), 60)
            print(f"\n📊 Session Summary:")
            print(f"   ⏱️  Total time: {minutes:02d}:{seconds:02d}")
            print(f"   🎯 Focus sessions: {self.focus_sessions}")
            print(f"   📈 Status changes: {len(self.status_history)}")
        
        print("👋 Monitoring stopped. Thanks for using Student Attention Monitor!")
    
    def generate_report(self):
        """Generate attention report from history"""
        if not self.status_history:
            print("❌ No data available for report")
            return
        
        # Calculate statistics
        engaged_count = sum(1 for entry in self.status_history if entry["status"] == "Engaged")
        total_count = len(self.status_history)
        engagement_rate = (engaged_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"\n📋 Attention Report:")
        print(f"   📊 Engagement Rate: {engagement_rate:.1f}%")
        print(f"   ✅ Engaged Frames: {engaged_count}/{total_count}")
        print(f"   🎯 Focus Sessions: {self.focus_sessions}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"attention_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("Student Attention Monitoring Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Engagement Rate: {engagement_rate:.1f}%\n")
            f.write(f"Engaged Frames: {engaged_count}/{total_count}\n")
            f.write(f"Focus Sessions: {self.focus_sessions}\n\n")
            
            f.write("Status History:\n")
            f.write("-" * 20 + "\n")
            for entry in self.status_history:
                timestamp = datetime.fromtimestamp(entry["timestamp"]).strftime('%H:%M:%S')
                f.write(f"{timestamp} - {entry['status']} (Session #{entry['session']})\n")
        
        print(f"📄 Detailed report saved: {report_file}")


def main():
    """Main function to run the realtime monitor"""
    print("🎓 Student Attention Monitor - Realtime Version")
    print("=" * 50)
    
    # Initialize monitor
    monitor = RealtimeAttentionMonitor(model_path="best.pt", confidence_threshold=0.6)
    monitor.start_monitoring(camera_index=0)