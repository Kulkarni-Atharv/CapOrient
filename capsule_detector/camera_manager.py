from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import cv2
import os
import time


class CameraManager:
    def __init__(self, save_dir="/media/pi/android/ACE", exposure=1000, gain=3.0, fps=60):
        self.save_dir = save_dir
        self.exposure = exposure
        self.gain = gain
        self.fps = fps
        self.recording = False
        self.start_time = 0
        self.save_path = ""
        self.picam2 = None

    def wait_for_storage(self):
        print("Waiting for storage...")
        while not os.path.exists(self.save_dir):
            print("Storage not ready, retrying...")
            time.sleep(2)
        print("Storage ready!")

    def initialize_camera(self):
        self.wait_for_storage()
        self.picam2 = Picamera2()

        video_config = self.picam2.create_video_configuration(
            main={"size": (1456, 1088), "format": "RGB888"},
            lores={"size": (640, 480), "format": "RGB888"},
            controls={
                "FrameRate": self.fps,
                "ExposureTime": self.exposure,
                "AnalogueGain": self.gain
            }
        )
        self.picam2.configure(video_config)
        self.picam2.start()

    def start_auto_recording(self):
        if self.recording:
            return

        filename = time.strftime("video_%Y%m%d_%H%M%S.h264")
        self.save_path = os.path.join(self.save_dir, filename)

        encoder = H264Encoder(bitrate=8000000)
        output = FileOutput(self.save_path)

        self.picam2.start_recording(encoder, output)
        self.recording = True
        self.start_time = time.time()

        print("Auto recording started")
        print("Saving to:", self.save_path)

    def stop_recording(self):
        if self.recording:
            self.picam2.stop_recording()
            self.recording = False
            print("Recording stopped")
            print("Saved at:", self.save_path)

    def close(self):
        self.stop_recording()
        if self.picam2:
            self.picam2.stop()

    def update_settings(self, exposure=None, gain=None, fps=None):
        if exposure is not None:
            self.exposure = int(exposure)
        if gain is not None:
            self.gain = float(gain)
        if fps is not None:
            self.fps = int(fps)

        if self.picam2:
            self.picam2.set_controls({
                "ExposureTime": self.exposure,
                "AnalogueGain": self.gain,
                "FrameRate": self.fps
            })

    def get_frame_jpeg(self):
        """Captures a frame and encodes it as JPEG for MJPEG streaming."""
        if not self.picam2:
            return None
        try:
            frame = self.picam2.capture_array("lores")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', frame_bgr)
            if ret:
                return jpeg.tobytes()
            return None
        except Exception as e:
            print("Error capturing frame:", e)
            return None

    def get_frame(self):
        """Returns the latest lores frame as a BGR numpy array for the detection pipeline."""
        if not self.picam2:
            return None
        try:
            frame = self.picam2.capture_array("lores")
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("Error capturing frame:", e)
            return None

    def get_recording_time(self):
        if self.recording:
            elapsed = int(time.time() - self.start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            return f"{mins:02d}:{secs:02d}"
        return "00:00"

    def get_status(self):
        return {
            "recording": self.recording,
            "time": self.get_recording_time(),
            "exposure": self.exposure,
            "gain": self.gain,
            "fps": self.fps
        }
