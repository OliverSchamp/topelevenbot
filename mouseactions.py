from platform import release
import struct
import wayland_automation as wa

# Assuming wa is your WaylandAutomation instance
# 0x111 is BTN_LEFT

def release_only(mouse: wa.Mouse, button_code=0x111):
    # 1. Send the Release message (state 0)
    mouse.send_message(
        mouse.current_virtual_pointer_id, 
        2, 
        struct.pack(f"{mouse.endianness}III", 0, button_code, 0)
    )
    
    # 2. Send Frame
    mouse.send_message(mouse.current_virtual_pointer_id, 4, b'')
    
    # 3. Flush and sync
    mouse.send_sync_request()
    mouse.handle_events()

def press_only(mouse: wa.Mouse, button_code=0x111):
    # 1. Send the Press message (Opcode 2)
    # The 'III' represents: time, button_code, state (1 for down)
    mouse.send_message(
        mouse.current_virtual_pointer_id, 
        2, 
        struct.pack(f"{mouse.endianness}III", 0, button_code, 1)
    )
    
    # 2. MUST send a Frame (Opcode 4) to "close" the event package
    mouse.send_message(mouse.current_virtual_pointer_id, 4, b'')
    
    # 3. MUST sync and handle events to flush the buffer to the compositor
    mouse.send_sync_request()
    mouse.handle_events()


if __name__ == "__main__":
    import time
    mouse = wa.Mouse()

    start_lc = time.perf_counter()
    press_only(mouse)
    print(time.perf_counter()-start_lc)

    start_lc = time.perf_counter()
    release_only(mouse)
    print(time.perf_counter()-start_lc)