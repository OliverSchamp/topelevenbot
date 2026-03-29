import os
import subprocess

def disable_mouse_acceleration():
    """
    Detects the active Wayland compositor on Raspberry Pi OS (Wayfire or Labwc)
    and programmatically disables mouse acceleration by setting a 'flat' profile.
    """
    home = os.path.expanduser("~")

    # ==========================================
    # 1. WAYFIRE (Older Pi OS Bookworm Default)
    # ==========================================
    wayfire_conf = os.path.join(home, ".config", "wayfire.ini")
    if os.path.exists(wayfire_conf):
        with open(wayfire_conf, "r") as f:
            lines = f.readlines()

        in_input_section = False
        found_accel = False
        new_lines = []

        for line in lines:
            if line.strip().startswith("[input]"):
                in_input_section = True
            elif line.strip().startswith("[") and in_input_section:
                # Leaving the [input] section. Inject setting if we haven't found it.
                if not found_accel:
                    new_lines.append("mouse_accel_profile = flat\n")
                    found_accel = True
                in_input_section = False
            
            # If the setting already exists, overwrite it
            if in_input_section and line.strip().startswith("mouse_accel_profile"):
                new_lines.append("mouse_accel_profile = flat\n")
                found_accel = True
                continue
                
            new_lines.append(line)

        # Catch-all: If [input] was the very last section in the file
        if in_input_section and not found_accel:
            new_lines.append("mouse_accel_profile = flat\n")
            
        # Catch-all: If [input] didn't exist at all
        if not any(l.strip().startswith("[input]") for l in lines):
            new_lines.extend(["\n[input]\n", "mouse_accel_profile = flat\n"])

        with open(wayfire_conf, "w") as f:
            f.writelines(new_lines)
            
        print("Wayfire mouse acceleration disabled. Changes apply instantly.")

    # ==========================================
    # 2. LABWC (Newer Pi OS Default)
    # ==========================================
    labwc_conf = os.path.join(home, ".config", "labwc", "rc.xml")
    if os.path.exists(labwc_conf):
        with open(labwc_conf, "r") as f:
            content = f.read()

        # Simple check to see if we've already patched it
        if "accelProfile" not in content:
            # Inject the libinput block right before the closing XML tag
            injection = """
  <libinput>
    <device category="default">
      <accelProfile>flat</accelProfile>
    </device>
  </libinput>
"""
            new_content = content.replace("</labwc_config>", injection + "</labwc_config>")
            
            with open(labwc_conf, "w") as f:
                f.write(new_content)
                
            # Labwc requires a terminal command to reload its config
            subprocess.run(["labwc", "--reconfigure"], check=False)
            print("Labwc mouse acceleration disabled. Reconfigured successfully.")
        else:
            print("Labwc mouse acceleration already appears to be disabled.")

if __name__ == "__main__":
    disable_mouse_acceleration()