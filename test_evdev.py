import time
from evdev import UInput, ecodes as e

class FastAutomator:
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

    def close(self):
        """Cleanly unplugs the virtual device"""
        self.ui.close()

# ==========================================
# Speed Test
# ==========================================
if __name__ == "__main__":
    bot = FastAutomator()
    
    print("Testing mouse down latency...")
    start = time.perf_counter()
    
    bot.left_mouse_down()
    bot.left_mouse_up()
    
    duration_ms = (time.perf_counter() - start) * 1000
    print(f"Click completed in: {duration_ms:.4f} ms")
    
    bot.close()