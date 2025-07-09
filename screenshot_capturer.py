from pathlib import Path
import time
import pyautogui
from datetime import datetime

# Directory to save screenshots
SAVE_DIR = Path('screenshots')
SAVE_DIR.mkdir(exist_ok=True)

def main():
    print("Waiting 10 seconds before starting screenshot capture...")
    time.sleep(10)
    print("Starting screenshot capture. Press Ctrl+C to stop.")
    try:
        while True:
            start_loop = time.time()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'screenshot_{timestamp}.jpg'
            filepath = SAVE_DIR / filename
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            print(f"Saved {filepath}")
            print(f"Time taken: {time.time() - start_loop} seconds")
            time.sleep(max(0, 0.25 - (time.time() - start_loop)))
    except KeyboardInterrupt:
        print("\nScreenshot capture stopped by user.")

if __name__ == "__main__":
    main() 