import subprocess
import time
import os

# Ensure the script knows where the ydotool socket is
# (Matches the export command we used in the setup)
os.environ["YDOTOOL_SOCKET"] = "/tmp/ydotool_socket"

class YdoAutomator:
    def __init__(self):
        # Quick check to ensure ydotool is accessible
        try:
            subprocess.run(["ydotool", "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            raise Exception("ydotool is not installed or not in the system PATH.")

    def _run_cmd(self, args):
        """Helper to run ydotool commands"""
        subprocess.run(["ydotool"] + args, check=True)

    def move(self, x, y, absolute=False):
        """
        Moves the mouse. 
        absolute=False moves relative to current position.
        absolute=True moves to specific screen coordinates.
        """
        if absolute:
            self._run_cmd(["mousemove", "-a", str(x), str(y)])
        else:
            self._run_cmd(["mousemove", str(x), str(y)])

    def left_click(self):
        """Standard left click (down + up)"""
        # 0xC0 is the hexadecimal code for a left click in ydotool
        self._run_cmd(["click", "0xC0"])

    def left_mouse_down(self):
        """Presses and holds the left mouse button"""
        # 0x40 is the code for left mouse button down
        self._run_cmd(["click", "0x40"])

    def left_mouse_up(self):
        """Releases the left mouse button"""
        # 0x80 is the code for left mouse button up
        self._run_cmd(["click", "0x80"])

    def right_click(self):
        """Standard right click"""
        self._run_cmd(["click", "0xC1"])

    def type_text(self, text):
        """Types out a string of text"""
        self._run_cmd(["type", text])

    def press_key(self, key):
        """
        Presses a specific key or combination. 
        Examples: 'enter', 'esc', 'ctrl+c', 'alt+tab'
        """
        self._run_cmd(["key", key])

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    bot = YdoAutomator()

    print("Starting automation in 3 seconds...")
    time.sleep(3)

    # 1. Move the mouse relative to its current position (e.g., 50px right, 50px down)
    print("Moving mouse...")
    bot.move(50, 50)
    time.sleep(1)

    # 2. Click the left mouse button
    print("Clicking...")
    bot.left_click()
    time.sleep(1)

    # 3. Perform a drag-and-drop (Mouse Down -> Move -> Mouse Up)
    print("Dragging and dropping...")
    bot.left_mouse_down()
    time.sleep(0.5)
    bot.move(100, 0) # Drag 100px to the right
    time.sleep(0.5)
    bot.left_mouse_up()
    time.sleep(1)

    # 4. Use the Keyboard to type
    print("Typing text...")
    bot.type_text("Hello from Wayland!")
    
    # 5. Press the Enter key
    bot.press_key("enter")
    
    print("Automation complete!")