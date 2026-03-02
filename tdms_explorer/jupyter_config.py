"""
TDMS Explorer Jupyter Server Proxy Configuration

Provides the entry point for jupyter-server-proxy to auto-discover
and launch the TDMS Explorer Panel application from JupyterLab's launcher.
"""

import os


def setup_tdms_explorer():
    """Return the server-proxy configuration dict for TDMS Explorer.

    This function is registered as a ``jupyter_serverproxy_servers``
    entry point so that ``jupyter-server-proxy`` can discover it
    automatically.
    """
    # Resolve the path to panel_app.py relative to this file
    app_path = os.path.join(os.path.dirname(__file__), "panel_app.py")

    # Optional: SVG icon for the JupyterLab launcher tile
    icon_path = os.path.join(os.path.dirname(__file__), "static", "icon.svg")
    launcher_entry = {
        "title": "TDMS Explorer",
        "enabled": True,
    }
    if os.path.isfile(icon_path):
        launcher_entry["icon_path"] = icon_path

    return {
        "command": [
            "panel",
            "serve",
            app_path,
            "--port",
            "{port}",
            "--allow-websocket-origin",
            "*",
        ],
        "timeout": 20,
        "launcher_entry": launcher_entry,
    }
