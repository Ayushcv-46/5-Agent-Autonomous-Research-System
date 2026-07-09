# tests/conftest.py
#
# Your app.py runs as a script (streamlit adds its own directory to
# sys.path automatically), but pytest doesn't do that for you. This file
# makes `from graph.routing import ...` etc. work no matter where you run
# `pytest` from — VS Code, a CI runner, or your terminal from a different
# folder.

import sys
import os

# The root is the parent of this tests/ folder.
DAY24_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DAY24_ROOT not in sys.path:
    sys.path.insert(0, DAY24_ROOT)