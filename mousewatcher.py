# # run_watcher.py
# from wayland_automation import mouse_position

# if __name__ == "__main__":
#     mouse_position.main() # Or whatever the start function is named


# import evdev

# # You may need to change 'event0' to your actual mouse event ID
# # Use 'ls /dev/input/by-id' to find your mouse/pointer
# device = evdev.InputDevice('/dev/input/event0')

# print("Move your mouse... (Ctrl+C to stop)")
# for event in device.read_loop():
#     if event.type == evdev.ecodes.EV_ABS:
#         if event.code == evdev.ecodes.ABS_X:
#             print(f"X: {event.value}")
#         if event.code == evdev.ecodes.ABS_Y:
#             print(f"Y: {event.value}")