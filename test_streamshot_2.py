import cv2
import time
import threading
import numpy as np
import subprocess
import atexit
from PIL import Image
from pathlib import Path
from datetime import datetime

class FastScreenGrabber:
    # This class variable holds our single, globally shared instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        The Singleton magic: Before creating a new object, check if one 
        already exists. If it does, return that one instead.
        """
        if cls._instance is None:
            cls._instance = super(FastScreenGrabber, cls).__new__(cls)
            # Add a flag so __init__ knows if it has already run
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device_id=9, region="480,288 960x540"):
        # If we already set up the camera, skip initialization entirely
        if self._initialized:
            return

        self.lock = threading.Lock()

        print("Initializing wf-recorder and camera stream...")
        self.device_id = device_id
        self.device_path = f"/dev/video{device_id}"
        self.running = False
        
        # 1. Start wf-recorder automatically in the background
        cmd = [
            "wf-recorder", 
            "-g", region, 
            "-c", "rawvideo", 
            "-m", "v4l2", 
            "-x", "yuv420p", 
            "-f", self.device_path
        ]
        
        # Launch it and hide its terminal spam
        self.recorder_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # Register the cleanup function to run automatically if the script crashes or closes
        atexit.register(self.stop)
        
        # Give wf-recorder a short second to build the video buffer
        time.sleep(1.0)

        # 2. Connect OpenCV
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.stop()
            raise Exception(f"Cannot open {self.device_path}. v4l2loopback might not be loaded.")
            
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        
        if not self.ret:
            self.stop()
            raise Exception("Failed to read initial frame from wf-recorder.")

        # 3. Start the background consumer thread
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
        # Mark as initialized so it never runs this block again
        self._initialized = True
        print("Screen grabber ready!")

    def _update_loop(self):
        """Continuously clears the buffer to maintain sub-millisecond latency."""
        while self.running:
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            
            # Briefly lock the variable just long enough to update it safely
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def grab_frame(self) -> np.ndarray:
        """Instantly and safely returns the most recent frame."""
        with self.lock:
            if not self.ret or self.frame is None:
                raise Exception("Frame stream dropped.")
            # We copy it while locked so the background thread can't touch it

            # Copy the frame while locked
            safe_frame = self.frame.copy()
            
        # Use heavily optimized C++ SIMD instructions to rewrite the memory to RGB
        # return cv2.cvtColor(safe_frame, cv2.COLOR_BGR2GRAY)
        return safe_frame[:, :, ::-1]

    def stop(self):
        """Safely shuts down the thread, camera, and wf-recorder process."""
        if self.running:
            self.running = False
            if hasattr(self, 'thread'):
                self.thread.join(timeout=1.0)
                
            if hasattr(self, 'cap'):
                self.cap.release()
                
            if hasattr(self, 'recorder_process'):
                # Terminate the wf-recorder background process
                self.recorder_process.terminate()
                self.recorder_process.wait(timeout=2.0)
                
            self._initialized = False
            print("Screen grabber cleanly shut down.")

# ==========================================
# Example Usage (Proving the Singleton works)
# ==========================================
if __name__ == "__main__":
    # The first time you call it, it boots up wf-recorder and the thread
    grabber1 = FastScreenGrabber()
    
    # The second time you call it, it just gives you the exact same grabber1 object!
    # No duplicate processes, no double camera initializations.
    grabber2 = FastScreenGrabber()
    
    print(f"Are they the same object in memory? {grabber1 is grabber2}")


    parent = Path("/mnt/SF_NAS/Oliver/auction2")
    parent.mkdir(exist_ok=True)
    # Testing 1000 frames instead of 100 because it's going to be so fast
    while True:
        img = grabber2.grab_frame()
        img_pil = Image.fromarray(img)
        time_now = datetime.now()
        date_time = time_now.strftime("%m_%d_%Y__%H_%M_%S")
        img_pil.save(parent / f"{date_time}.jpg")

        time.sleep(0.01)
    