"""
Fire Detection System - Quick Launcher
Run this script to launch the full Integrated Streamlit Fire Detection App.
Usage:
    python run.py
"""

import os
import sys
import subprocess

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(root_dir, "Integrated_Fire_System", "app.py")

    if not os.path.exists(app_path):
        print(f"Error: Could not locate app.py at {app_path}")
        sys.exit(1)

    print("=" * 60)
    print("🔥 Starting Integrated Fire Detection & Risk System...")
    print(f"📂 Application: {app_path}")
    print("=" * 60)

    # Use sys.executable to ensure we run under the current Python environment
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApplication stopped.")

if __name__ == "__main__":
    main()
