"""Dev server entrypoint without the auto-reloader.

main.py runs with debug=True, which makes werkzeug fork a reloader child. On
this machine that left orphaned listeners bound to port 5000 across restarts,
so requests were silently answered by a stale process running old code. Use
this for anything you intend to verify.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import create_app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    create_app().run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
