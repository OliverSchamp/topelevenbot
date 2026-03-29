from pathlib import Path
from utils.ocr import PPOCRv5OpenVINO
import time
from wayland_automation import Mouse
from evdev import UInput, ecodes as e
import cv2
import time
import threading
import numpy as np
import subprocess
import atexit

class FastAutomator:
    KEY_MAP = {
        'esc': e.KEY_ESC,
        'enter': e.KEY_ENTER,
        'space': e.KEY_SPACE,
        'backspace': e.KEY_BACKSPACE,
        'tab': e.KEY_TAB,
        'up': e.KEY_UP,
        'down': e.KEY_DOWN,
        'left': e.KEY_LEFT,
        'right': e.KEY_RIGHT,
        'a': e.KEY_A,
        'b': e.KEY_B,
        # Add any other letters or keys you need here!
    }
    def __init__(self):
        print("Registering virtual hardware with the Linux kernel...")
        
        # We must declare what this virtual device is capable of doing.
        # We tell the kernel it has a mouse (X/Y relative) and standard keys.
        capabilities = {
            e.EV_REL: (e.REL_X, e.REL_Y),
            e.EV_KEY: list(range(1, 128)) + [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE]
        }
        
        # Create the virtual device
        self.ui = UInput(capabilities, name="python-fast-bot", version=1)
        
        # The Wayland compositor needs a tiny fraction of a second to 
        # recognize that a "new USB device" was just plugged in
        time.sleep(0.1) 

    def left_mouse_down(self):
        """0ms latency Left Mouse Down"""
        self.ui.write(e.EV_KEY, e.BTN_LEFT, 1) # 1 means Key Down
        self.ui.syn() # Instantly syncs the event to the kernel

    def left_mouse_up(self):
        """0ms latency Left Mouse Up"""
        self.ui.write(e.EV_KEY, e.BTN_LEFT, 0) # 0 means Key Up
        self.ui.syn()

    def left_click(self):
        self.left_mouse_down()
        self.left_mouse_up()

    def move_relative(self, x, y):
        # Note: Wayland may still apply acceleration to these raw values 
        # unless you disabled it with the XML/INI hack we did earlier
        self.ui.write(e.EV_REL, e.REL_X, x)
        self.ui.write(e.EV_REL, e.REL_Y, y)
        self.ui.syn()
        
    def move_absolute(self, x, y):
        """Re-implementing the Corner Hack natively"""
        self.move_relative(-9999, -9999)
        time.sleep(0.01)
        self.move_relative(x, y)

    def press_key_code(self, key_code):
        """
        Presses a key using evdev ecodes.
        Example: self.press_key_code(e.KEY_ENTER)
        """
        self.ui.write(e.EV_KEY, key_code, 1) # Down
        self.ui.write(e.EV_KEY, key_code, 0) # Up
        self.ui.syn()

    def press_key(self, key_name: str):
        """
        Presses a key using a friendly string name.
        Example: bot.press_key('esc')
        """
        # Look up the string in our dictionary (convert to lowercase just in case)
        key_code = self.KEY_MAP.get(key_name.lower())
        
        if key_code is None:
            raise ValueError(f"Key '{key_name}' is not in the KEY_MAP dictionary.")
            
        # Press down, release, and sync instantly
        self.ui.write(e.EV_KEY, key_code, 1) # Down
        self.ui.write(e.EV_KEY, key_code, 0) # Up
        self.ui.syn()

    def close(self):
        """Cleanly unplugs the virtual device"""
        self.ui.close()

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

    def grab_frame(self, x1=None, y1=None, x2=None, y2=None, mode="RGB") -> np.ndarray:
        """
        Instantly returns the most recent frame in RGB.
        Optionally pass bounding box coordinates to return a cropped region.
        """
        with self.lock:
            if not self.ret or self.frame is None:
                raise Exception("Frame stream dropped.")
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                view = self.frame[y1:y2, x1:x2]
            else:
                view = self.frame
            safe_frame = view.copy()
        if mode == "BGR":
            return safe_frame
        elif mode == "GRAY":
            return cv2.cvtColor(safe_frame, cv2.COLOR_BGR2GRAY)
        else:
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

IMAGE_PATHS = {
    "claim_mystery_choice":Path("img/general/claim_mystery_choice.JPG"),
    "daily_rewards":Path("img/general/daily_rewards.JPG"),
    "mystery_button":Path("img/general/mystery_button.JPG"),
    "mystery_choice":Path("img/general/mystery_button.JPG"),
    # "penalty_clash_template": "img/penalty_clash_template_placeholder.png"  # Placeholder
}

MYSTERY_CHOICE_COORDS = {'x': 960, 'y':697}
# Removed penalty clash config values; now in penalty_clash_config.py

width = 960
height = 540
SCREEN_WIDTH=1920
SCREEN_HEIGHT=1080
taskbar_offset_px = 18

ocr_pipeline = PPOCRv5OpenVINO(
    det_model_paths=["ocr_model/detector/detector.xml", "ocr_model/detector_table/output_det_ft.xml"],
    rec_model_paths=["ocr_model/recognizer/recognizer.xml"],
    dict_path="ocr_model/ppocrv5_en_dict.txt"
)

mouse_keyboard_controller = FastAutomator()

mouse_move_controller = Mouse()

screengrabber = FastScreenGrabber()