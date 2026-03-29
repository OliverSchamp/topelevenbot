import subprocess
import re

def get_waydroid_geometry():
    # Run wlrctl to list all windows
    result = subprocess.run(['wlrctl', 'window', 'list'], capture_output=True, text=True)
    
    # Look for the line containing 'waydroid'
    # The output usually looks like: "waydroid: Waydroid (0,0 600x900)"
    for line in result.stdout.splitlines():
        if "waydroid" in line.lower():
            # Regex to pull out: x, y, width, height
            match = re.search(r'\((\d+),(\d+)\s+(\d+)x(\d+)\)', line)
            if match:
                x, y, w, h = match.groups()
                return f"{x},{y} {w}x{h}"
    return None

# Usage with your existing grim function:
region = get_waydroid_geometry()
if region:
    # command = ['grim', '-t', 'ppm', '-g', region, '-']
    print(f"Targeting Waydroid at: {region}")