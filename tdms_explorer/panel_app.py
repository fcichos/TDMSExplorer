"""
TDMS Explorer Panel Application

A Panel-based web application for exploring TDMS files in JupyterLab.
Launch with: panel serve tdms_explorer/panel_app.py --show
"""

import os
import numpy as np
import param
import panel as pn
import holoviews as hv
from holoviews import opts

from tdms_explorer.core import TDMSFileExplorer, list_tdms_files

pn.extension("bokeh")
hv.extension("bokeh")


class TDMSExplorerApp(param.Parameterized):
    """Main Panel application for exploring TDMS files."""

    # --- Sidebar parameters ---
    directory = param.String(default=os.getcwd(), doc="Directory containing TDMS files")
    selected_file = param.Selector(default=None, objects=[], doc="TDMS file to load")
    load_file = param.Action(lambda self: self._load_file(), doc="Load selected file")

    # --- Image Viewer parameters ---
    frame = param.Integer(default=0, bounds=(0, 0), doc="Current frame index")
    colormap = param.Selector(
        default="gray",
        objects=["gray", "viridis", "plasma", "inferno", "magma", "cividis", "hot", "jet"],
        doc="Colormap for image display",
    )

    # --- Channel Data parameters ---
    selected_group = param.Selector(default=None, objects=[], doc="Channel group")
    selected_channel = param.Selector(default=None, objects=[], doc="Channel")

    # --- Image Analysis parameters ---
    analysis_frame = param.Integer(default=0, bounds=(0, 0), doc="Frame for analysis")
    filter_type = param.Selector(
        default="gaussian",
        objects=["gaussian", "median", "bilateral", "sobel", "prewitt", "laplace"],
        doc="Filter type",
    )
    filter_sigma = param.Number(default=1.0, bounds=(0.1, 10.0), doc="Filter sigma/size")
    edge_method = param.Selector(
        default="canny", objects=["canny", "sobel", "prewitt", "laplace"], doc="Edge method"
    )
    profile_direction = param.Selector(
        default="horizontal", objects=["horizontal", "vertical", "diagonal"], doc="Profile direction"
    )
    profile_position = param.Integer(default=0, bounds=(0, 0), doc="Profile line position")
    histogram_bins = param.Integer(default=256, bounds=(16, 1024), step=16, doc="Histogram bins")

    # --- Compare parameters ---
    compare_frame_a = param.Integer(default=0, bounds=(0, 0), doc="Frame A")
    compare_frame_b = param.Integer(default=0, bounds=(0, 0), doc="Frame B")
    compare_method = param.Selector(
        default="difference", objects=["difference", "absolute", "relative"], doc="Comparison method"
    )

    def __init__(self, **params):
        super().__init__(**params)
        self._explorer = None
        self._images = None
        self._status = pn.pane.Alert("No file loaded. Select a directory and file above.", alert_type="info")
        self._refresh_file_list()

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _refresh_file_list(self):
        """Scan directory for .tdms files and populate selector."""
        d = self.directory
        if not os.path.isdir(d):
            self.param.selected_file.objects = []
            self.selected_file = None
            return
        files = sorted(list_tdms_files(d))
        names = [os.path.basename(f) for f in files]
        self._file_map = dict(zip(names, files))
        self.param.selected_file.objects = names
        if names:
            self.selected_file = names[0]
        else:
            self.selected_file = None

    @param.depends("directory", watch=True)
    def _on_directory_change(self):
        self._refresh_file_list()

    def _load_file(self):
        """Load the selected TDMS file."""
        if self.selected_file is None:
            self._status.object = "No file selected."
            self._status.alert_type = "warning"
            return
        path = self._file_map.get(self.selected_file)
        if path is None:
            return
        try:
            self._explorer = TDMSFileExplorer(path)
            self._images = self._explorer.extract_images()
            self._update_bounds()
            self._update_group_list()
            self._status.object = f"Loaded: {self.selected_file}"
            self._status.alert_type = "success"
        except Exception as e:
            self._status.object = f"Error loading file: {e}"
            self._status.alert_type = "danger"

    def _update_bounds(self):
        """Update frame slider bounds after loading."""
        if self._images is not None:
            n = self._images.shape[0] - 1
            self.param.frame.bounds = (0, n)
            self.param.analysis_frame.bounds = (0, n)
            self.param.compare_frame_a.bounds = (0, n)
            self.param.compare_frame_b.bounds = (0, max(n, 1))
            max_pos = max(self._images.shape[1], self._images.shape[2]) - 1
            self.param.profile_position.bounds = (0, max_pos)
            self.frame = 0
            self.analysis_frame = 0
            self.compare_frame_a = 0
            self.compare_frame_b = min(1, n)

    def _update_group_list(self):
        """Populate group/channel selectors from loaded file."""
        if self._explorer is None:
            return
        groups = self._explorer.groups
        self.param.selected_group.objects = groups
        if groups:
            self.selected_group = groups[0]

    @param.depends("selected_group", watch=True)
    def _on_group_change(self):
        if self._explorer is None or self.selected_group is None:
            return
        channels = self._explorer.channels.get(self.selected_group, {}).get("_channels", [])
        self.param.selected_channel.objects = channels
        if channels:
            self.selected_channel = channels[0]

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _sidebar(self):
        dir_input = pn.widgets.TextInput.from_param(self.param.directory, name="Directory")
        file_select = pn.widgets.Select.from_param(self.param.selected_file, name="TDMS File")
        scan_btn = pn.widgets.Button(name="Scan Directory", button_type="default")
        scan_btn.on_click(lambda e: self._refresh_file_list())
        load_btn = pn.widgets.Button(name="Load File", button_type="primary")
        load_btn.on_click(lambda e: self._load_file())

        metadata_pane = pn.bind(self._metadata_view)

        return pn.Column(
            "## TDMS Explorer",
            dir_input,
            scan_btn,
            file_select,
            load_btn,
            self._status,
            pn.layout.Divider(),
            metadata_pane,
            width=320,
            sizing_mode="stretch_height",
        )

    @param.depends("selected_file")
    def _metadata_view(self):
        if self._explorer is None:
            return pn.pane.Markdown("*No file loaded*")
        md = self._explorer.metadata
        groups = self._explorer.groups
        lines = ["### File Metadata\n"]
        for k, v in md.items():
            if k == "file_size":
                v = f"{v / 1024:.1f} KB" if v < 1_048_576 else f"{v / 1_048_576:.1f} MB"
            lines.append(f"- **{k}**: {v}")
        lines.append(f"\n### Groups ({len(groups)})")
        for g in groups:
            ch_list = self._explorer.channels.get(g, {}).get("_channels", [])
            lines.append(f"- **{g}** ({len(ch_list)} channels)")
        if self._images is not None:
            lines.append(f"\n### Image Data")
            lines.append(f"- Frames: {self._images.shape[0]}")
            lines.append(f"- Size: {self._images.shape[1]} x {self._images.shape[2]}")
        return pn.pane.Markdown("\n".join(lines))

    # ------------------------------------------------------------------
    # Tab 1: Image Viewer
    # ------------------------------------------------------------------

    @param.depends("frame", "colormap")
    def _image_viewer(self):
        if self._images is None:
            return pn.pane.Markdown("*No image data available. Load a TDMS file with image data.*")
        img = self._images[self.frame]
        hv_img = hv.Image(
            (np.arange(img.shape[1]), np.arange(img.shape[0]), img),
            kdims=["x", "y"],
        ).opts(
            cmap=self.colormap,
            colorbar=True,
            width=600,
            height=500,
            tools=["hover"],
            title=f"Frame {self.frame}",
            invert_yaxis=True,
        )
        return hv_img

    @param.depends("frame")
    def _frame_stats(self):
        if self._images is None:
            return pn.pane.HTML("")
        img = self._images[self.frame]
        return pn.pane.HTML(
            f"<b>Frame {self.frame}</b> &mdash; "
            f"min: {img.min():.4g}, max: {img.max():.4g}, "
            f"mean: {img.mean():.4g}, std: {img.std():.4g}",
            styles={"font-size": "13px"},
        )

    def _image_tab(self):
        frame_slider = pn.widgets.IntSlider.from_param(self.param.frame, name="Frame")
        cmap_select = pn.widgets.Select.from_param(self.param.colormap, name="Colormap")
        controls = pn.Row(frame_slider, cmap_select)
        return pn.Column(controls, self._image_viewer, self._frame_stats, sizing_mode="stretch_width")

    # ------------------------------------------------------------------
    # Tab 2: Channel Data
    # ------------------------------------------------------------------

    @param.depends("selected_group", "selected_channel")
    def _channel_plot(self):
        if self._explorer is None or self.selected_group is None or self.selected_channel is None:
            return pn.pane.Markdown("*Select a group and channel after loading a file.*")
        data = self._explorer.get_raw_channel_data(self.selected_group, self.selected_channel)
        if data is None:
            return pn.pane.Markdown("*Could not read channel data.*")
        curve = hv.Curve(
            (np.arange(len(data)), data),
            kdims=["Sample"],
            vdims=["Value"],
        ).opts(
            width=700,
            height=400,
            title=f"{self.selected_group} / {self.selected_channel}",
            tools=["hover"],
            line_width=1,
        )
        stats_html = (
            f"<b>Channel:</b> {self.selected_channel} &mdash; "
            f"points: {len(data)}, min: {data.min():.4g}, max: {data.max():.4g}, "
            f"mean: {data.mean():.4g}, std: {data.std():.4g}"
        )
        return pn.Column(curve, pn.pane.HTML(stats_html, styles={"font-size": "13px"}))

    def _channel_tab(self):
        group_sel = pn.widgets.Select.from_param(self.param.selected_group, name="Group")
        channel_sel = pn.widgets.Select.from_param(self.param.selected_channel, name="Channel")
        controls = pn.Row(group_sel, channel_sel)
        return pn.Column(controls, self._channel_plot, sizing_mode="stretch_width")

    # ------------------------------------------------------------------
    # Tab 3: Image Analysis
    # ------------------------------------------------------------------

    def _analysis_tab(self):
        frame_slider = pn.widgets.IntSlider.from_param(self.param.analysis_frame, name="Frame")
        sub_tabs = pn.Tabs(
            ("Histogram", self._histogram_panel()),
            ("Filters", self._filter_panel()),
            ("Edge Detection", self._edge_panel()),
            ("Profile", self._profile_panel()),
        )
        return pn.Column(frame_slider, sub_tabs, sizing_mode="stretch_width")

    # --- Histogram ---

    @param.depends("analysis_frame", "histogram_bins")
    def _histogram_plot(self):
        if self._explorer is None or self._images is None:
            return pn.pane.Markdown("*No image data.*")
        hist_data = self._explorer.get_image_histogram(self.analysis_frame, bins=self.histogram_bins)
        if hist_data is None:
            return pn.pane.Markdown("*Could not compute histogram.*")
        edges = np.array(hist_data["bin_edges"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts = np.array(hist_data["hist"])
        bars = hv.Bars((centers, counts), kdims=["Intensity"], vdims=["Count"]).opts(
            width=700, height=350, title="Intensity Histogram", color="steelblue"
        )
        stats = self._explorer.analyze_image(self.analysis_frame)
        if stats:
            info = (
                f"<b>Stats:</b> min={stats['min']:.4g}, max={stats['max']:.4g}, "
                f"mean={stats['mean']:.4g}, std={stats['std']:.4g}, median={stats['median']:.4g}"
            )
        else:
            info = ""
        return pn.Column(bars, pn.pane.HTML(info, styles={"font-size": "13px"}))

    def _histogram_panel(self):
        bins_slider = pn.widgets.IntSlider.from_param(self.param.histogram_bins, name="Bins")
        return pn.Column(bins_slider, self._histogram_plot, sizing_mode="stretch_width")

    # --- Filters ---

    @param.depends("analysis_frame", "filter_type", "filter_sigma")
    def _filter_view(self):
        if self._explorer is None or self._images is None:
            return pn.pane.Markdown("*No image data.*")
        original = self._images[self.analysis_frame]
        kwargs = {}
        if self.filter_type == "gaussian":
            kwargs["sigma"] = self.filter_sigma
        elif self.filter_type == "median":
            kwargs["size"] = max(3, int(self.filter_sigma) | 1)  # ensure odd
        filtered = self._explorer.apply_image_filter(self.analysis_frame, self.filter_type, **kwargs)
        if filtered is None:
            return pn.pane.Markdown("*Filter not available (missing dependency).*")
        orig_hv = hv.Image(
            (np.arange(original.shape[1]), np.arange(original.shape[0]), original),
            kdims=["x", "y"],
            label="Original",
        ).opts(cmap="gray", colorbar=True, width=380, height=340, invert_yaxis=True)
        filt_hv = hv.Image(
            (np.arange(filtered.shape[1]), np.arange(filtered.shape[0]), filtered),
            kdims=["x", "y"],
            label="Filtered",
        ).opts(cmap="gray", colorbar=True, width=380, height=340, invert_yaxis=True)
        return pn.Row(orig_hv, filt_hv)

    def _filter_panel(self):
        ftype = pn.widgets.Select.from_param(self.param.filter_type, name="Filter")
        sigma = pn.widgets.FloatSlider.from_param(self.param.filter_sigma, name="Sigma / Size")
        return pn.Column(pn.Row(ftype, sigma), self._filter_view, sizing_mode="stretch_width")

    # --- Edge Detection ---

    @param.depends("analysis_frame", "edge_method")
    def _edge_view(self):
        if self._explorer is None or self._images is None:
            return pn.pane.Markdown("*No image data.*")
        original = self._images[self.analysis_frame]
        edges = self._explorer.detect_edges(self.analysis_frame, method=self.edge_method)
        if edges is None:
            return pn.pane.Markdown("*Edge detection not available (missing dependency).*")
        orig_hv = hv.Image(
            (np.arange(original.shape[1]), np.arange(original.shape[0]), original),
            kdims=["x", "y"],
            label="Original",
        ).opts(cmap="gray", colorbar=True, width=380, height=340, invert_yaxis=True)
        edge_hv = hv.Image(
            (np.arange(edges.shape[1]), np.arange(edges.shape[0]), edges),
            kdims=["x", "y"],
            label="Edges",
        ).opts(cmap="gray", colorbar=True, width=380, height=340, invert_yaxis=True)
        return pn.Row(orig_hv, edge_hv)

    def _edge_panel(self):
        method = pn.widgets.Select.from_param(self.param.edge_method, name="Method")
        return pn.Column(method, self._edge_view, sizing_mode="stretch_width")

    # --- Profile ---

    @param.depends("analysis_frame", "profile_direction", "profile_position")
    def _profile_view(self):
        if self._explorer is None or self._images is None:
            return pn.pane.Markdown("*No image data.*")
        pos = self.profile_position if self.profile_position > 0 else None
        profile = self._explorer.get_image_profile(self.analysis_frame, self.profile_direction, pos)
        if profile is None:
            return pn.pane.Markdown("*Could not extract profile.*")
        curve = hv.Curve(
            (profile["x"], profile["y"]),
            kdims=["Position"],
            vdims=["Intensity"],
        ).opts(
            width=700,
            height=350,
            title=f"{self.profile_direction.title()} profile (pos={profile['position']})",
            line_width=1.5,
        )
        return curve

    def _profile_panel(self):
        direction = pn.widgets.Select.from_param(self.param.profile_direction, name="Direction")
        position = pn.widgets.IntSlider.from_param(self.param.profile_position, name="Position")
        return pn.Column(pn.Row(direction, position), self._profile_view, sizing_mode="stretch_width")

    # ------------------------------------------------------------------
    # Tab 4: Compare
    # ------------------------------------------------------------------

    @param.depends("compare_frame_a", "compare_frame_b", "compare_method", "colormap")
    def _compare_view(self):
        if self._explorer is None or self._images is None:
            return pn.pane.Markdown("*No image data available.*")
        n = self._images.shape[0]
        fa, fb = self.compare_frame_a, self.compare_frame_b
        if fa >= n or fb >= n:
            return pn.pane.Markdown("*Frame index out of range.*")
        result = self._explorer.compare_images(fa, fb, method=self.compare_method)
        if result is None:
            return pn.pane.Markdown("*Comparison failed.*")
        img_a = self._images[fa]
        img_b = self._images[fb]

        def _make_hv(arr, label):
            return hv.Image(
                (np.arange(arr.shape[1]), np.arange(arr.shape[0]), arr),
                kdims=["x", "y"],
                label=label,
            ).opts(cmap=self.colormap, colorbar=True, width=320, height=300, invert_yaxis=True)

        hv_a = _make_hv(img_a, f"Frame {fa}")
        hv_b = _make_hv(img_b, f"Frame {fb}")
        hv_diff = _make_hv(result, self.compare_method.title())

        stats_html = (
            f"<b>{self.compare_method.title()}</b> &mdash; "
            f"min: {result.min():.4g}, max: {result.max():.4g}, "
            f"mean: {result.mean():.4g}, std: {result.std():.4g}"
        )
        return pn.Column(
            pn.Row(hv_a, hv_b, hv_diff),
            pn.pane.HTML(stats_html, styles={"font-size": "13px"}),
        )

    def _compare_tab(self):
        fa = pn.widgets.IntSlider.from_param(self.param.compare_frame_a, name="Frame A")
        fb = pn.widgets.IntSlider.from_param(self.param.compare_frame_b, name="Frame B")
        method = pn.widgets.Select.from_param(self.param.compare_method, name="Method")
        controls = pn.Row(fa, fb, method)
        return pn.Column(controls, self._compare_view, sizing_mode="stretch_width")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def view(self):
        """Build the full application layout."""
        tabs = pn.Tabs(
            ("Image Viewer", self._image_tab()),
            ("Channel Data", self._channel_tab()),
            ("Image Analysis", self._analysis_tab()),
            ("Compare", self._compare_tab()),
            sizing_mode="stretch_width",
        )
        template = pn.template.FastListTemplate(
            title="TDMS Explorer",
            sidebar=[self._sidebar()],
            main=[tabs],
            accent_base_color="#2196F3",
            header_background="#1565C0",
        )
        return template


# ------------------------------------------------------------------
# Serve
# ------------------------------------------------------------------

def create_app():
    """Create and return the Panel application."""
    app = TDMSExplorerApp()
    return app.view()


if __name__ == "__main__":
    # Direct execution: python panel_app.py --port 5006
    import argparse

    parser = argparse.ArgumentParser(description="TDMS Explorer Panel App")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--allow-websocket-origin", default="*")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    origins = [args.allow_websocket_origin] if args.allow_websocket_origin != "*" else ["*"]
    pn.serve(
        create_app,
        port=args.port,
        allow_websocket_origin=origins,
        show=args.show,
    )
else:
    # When served via: panel serve panel_app.py
    create_app().servable()
