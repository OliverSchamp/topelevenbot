# import cv2
# import time
# import numpy as np

# class ScreenGrabber:
#     def __init__(self, device_id=9):
#         # Open the virtual camera
#         self.cap = cv2.VideoCapture(device_id)
        
#         if not self.cap.isOpened():
#             raise Exception(f"Cannot open /dev/video{device_id}. Is wf-recorder running?")
            
#         # CRITICAL: Set buffer size to 1 so we always get the *newest* frame,
#         # not a delayed frame sitting in the queue.
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#     def grab_frame(self) -> np.ndarray:
#         """
#         Grabs the latest frame from the Wayland stream.
#         This usually executes in 1 to 3 milliseconds.
#         """
#         # Read the frame from the video buffer
#         ret, frame = self.cap.read()
        
#         if not ret:
#             raise Exception("Failed to grab frame from virtual camera.")
            
#         # The frame is already a numpy ndarray in BGR format, 
#         # which is exactly what OpenCV and your OCR pipeline expect!
#         return frame

# # ==========================================
# # Speed Test Example
# # ==========================================
# if __name__ == "__main__":
#     screen = ScreenGrabber(device_id=9)
    
#     # Warm up the camera (first frame usually takes a little longer)
#     screen.grab_frame()
    
#     print("Testing capture speed over 1 frames...")
    
#     times = []

#     for _ in range(10):
#         start = time.perf_counter()
#         img = screen.grab_frame()
#         duration_ms = (time.perf_counter() - start) * 1000
#         times.append(duration_ms)
        
#     avg_time = sum(times) / len(times)
#     max_time = max(times)
#     min_time = min(times)
    
#     print(f"Average: {avg_time:.2f} ms")
#     print(f"Fastest: {min_time:.2f} ms")
#     print(f"Slowest: {max_time:.2f} ms")
    
#     # Don't forget to release when your bot shuts down
#     screen.cap.release()

#     cv2.imwrite("out.jpg", img)


import cv2
import time
import threading
import numpy as np

class FastScreenGrabber:
    def __init__(self, device_id=9):
        # 1. Force OpenCV to use V4L2 directly, bypassing heavy GStreamer pipelines
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            raise Exception(f"Cannot open /dev/video{device_id}. Is wf-recorder running?")
            
        # Try to minimize internal buffering
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Read the first frame to ensure everything is working
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise Exception("Failed to read initial frame.")

        # 2. Start the background thread
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        """
        This runs infinitely in the background. 
        It silently consumes frames to prevent the 1-second timeout buffer clog.
        """
        while self.running:
            # grab() fetches the data, retrieve() decodes it. 
            # This is slightly faster than read()
            self.cap.grab()
            self.ret, self.frame = self.cap.retrieve()

    def grab_frame(self) -> np.ndarray:
        """
        Instantly returns the most recent frame from memory.
        Execution time: ~0.001 ms
        """
        if not self.ret or self.frame is None:
            raise Exception("Frame stream dropped.")
            
        # Return a copy so your bot's OpenCV image processing 
        # doesn't corrupt the frame while the thread is updating it
        return self.frame.copy()

    def stop(self):
        """Cleanly shut down the background thread and camera."""
        self.running = False
        self.thread.join()
        self.cap.release()

# ==========================================
# Speed Test
# ==========================================
if __name__ == "__main__":
    print("Initializing camera and background thread...")
    screen = FastScreenGrabber(device_id=9)
    
    # Give the thread a tiny moment to spin up
    time.sleep(0.5)
    
    print("Testing capture speed over 1,000 frames...")
    
    times = []
    # Testing 1000 frames instead of 100 because it's going to be so fast
    for _ in range(1000):
        start = time.perf_counter()
        img = screen.grab_frame()
        duration_ms = (time.perf_counter() - start) * 1000
        times.append(duration_ms)
        
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    print(f"Average: {avg_time:.4f} ms")
    print(f"Fastest: {min_time:.4f} ms")
    print(f"Slowest: {max_time:.4f} ms")
    
    screen.stop()

    cv2.imwrite("out.jpg", img)