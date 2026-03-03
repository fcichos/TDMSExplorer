"""
TDMS Explorer Jupyter Server Proxy Configuration

Provides the entry point for jupyter-server-proxy to auto-discover
and launch the TDMS Explorer Panel application from JupyterLab's launcher.
"""

import os
import shutil
import sys


def setup_tdms_explorer():
    """Return the server-proxy configuration dict for TDMS Explorer.

    This function is registered as a ``jupyter_serverproxy_servers``
    entry point so that ``jupyter-server-proxy`` can discover it
    automatically.
    """
    # Resolve the path to panel_app.py relative to this file
    app_path = os.path.join(os.path.dirname(__file__), "panel_app.py")

    # Find the panel executable — use the same environment as this Python
    panel_cmd = shutil.which("panel") or os.path.join(
        os.path.dirname(sys.executable), "panel"
    )

    # Optional: SVG icon for the JupyterLab launcher tile
    icon_path = os.path.join(os.path.dirname(__file__), "static", "icon.svg")
    launcher_entry = {
        "title": "TDMS Explorer",
        "enabled": True,
    }
    if os.path.isfile(icon_path):
        launcher_entry["icon_path"] = icon_path

    def _mappath(path):
        """Prepend /panel_app so all proxy paths reach the Panel app.

        Server-proxy strips its prefix before forwarding, so the browser
        URL /user/mona/tdms-explorer/ws arrives here as /ws.  Panel
        expects /panel_app/ws, etc.
        """
        if path in ("", "/"):
            return "/panel_app"
        if not path.startswith("/panel_app") and not path.startswith("/static"):
            return "/panel_app" + path
        return path

    return {
        "command": [
            panel_cmd,
            "serve",
            app_path,
            "--port",
            "{port}",
            "--allow-websocket-origin",
            "*",
        ],
        "timeout": 30,
        "mappath": _mappath,
        "launcher_entry": launcher_entry,
        "new_browser_tab": True,
    }
