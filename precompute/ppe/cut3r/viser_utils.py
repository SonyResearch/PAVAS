# -*- coding: utf-8 -*-
"""
Minimal PointCloudViewer for label-colored 3D points using viser.

- Map an `(H, W, 3)` world-space point map to an `(H, W)` segmentation mask.
- Add per-label point clouds to viser using distinct colors.
- Optionally render camera frustums.
"""

import time
import numpy as np
import torch
import viser
import viser.transforms as tf


def generate_pastel_color_map(max_labels=50):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    out = {}
    for i in range(max_labels):
        out[i] = cmap(i % cmap.N)[:3]
    # Reassign label 0 here if you prefer a different background color.
    return out


def map_mask_to_3d_points(pts3d_hw3, segmask_hw, conf_hw=None, conf_thr=None, drop_background=True):
    """
    pts3d_hw3 : (H,W,3) pixel-aligned 3D points (world or camera coords)
    segmask_hw: (H,W)   int labels (0=bg)
    conf_hw   : (H,W)   optional conf map
    conf_thr  : float   optional threshold
    return:
        pts (N,3), labels (N,)
    """
    H, W, _ = pts3d_hw3.shape
    pts = pts3d_hw3.reshape(-1, 3)
    labels = segmask_hw.reshape(-1).astype(np.int32, copy=False)

    valid = np.isfinite(pts).all(axis=1) & ~(np.all(np.isclose(pts, 0.0), axis=1))
    if drop_background:
        valid &= (labels != 0)
    if conf_hw is not None and conf_thr is not None:
        conf = conf_hw.reshape(-1)
        valid &= (conf > conf_thr)

    pts = pts[valid]
    labels = labels[valid]
    return pts.astype(np.float32), labels.astype(np.int32)


class PointCloudViewer:
    def __init__(
        self,
        model,
        state_args,
        pc_list,            # list of (H,W,3) float32
        color_list,         # list of (H,W,3) float32 in [0,1] (unused for seg, but kept for future)
        conf_list,          # list of (H,W) float32 or None
        seg_mask_list,      # list of (H,W) int
        cam_dict,           # dict with focal, pp, R, t
        label_color_map=None,
        device="cpu",
        port=8080,
        show_camera=True,
        vis_threshold=1.5,
        size=512,
    ):
        self.model = model
        self.state_args = state_args
        self.pcs = pc_list
        self.colors = color_list
        self.confs = conf_list
        self.seg_masks = seg_mask_list
        self.cam_dict = cam_dict
        self.num_frames = len(self.pcs)
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")
        self.device = device
        self.show_camera = show_camera
        self.vis_threshold = vis_threshold
        self.size = size
        self.label_color_map = label_color_map or generate_pastel_color_map()

        # gui
        self.fourd = False
        b1 = self.server.gui.add_button("4D (Only Show Current Frame)")
        b2 = self.server.gui.add_button("3D (Show All Frames)")
        @b1.on_click
        def _(_e): self.fourd = True
        @b2.on_click
        def _(_e): self.fourd = False

        self.psize_slider = self.server.add_gui_slider("Point Size", min=0.0005, max=0.05, step=0.0005, initial_value=0.005)
        self.camsize_slider = self.server.add_gui_slider("Camera Size", min=0.02, max=0.5, step=0.01, initial_value=0.1)

        self.pc_handles = []
        self.cam_handles = []
        self.server.on_client_connect(self._connect_client)

    def _connect_client(self, client: viser.ClientHandle):
        # simple HUD (optional)
        wxyz_panel = client.gui.add_text("wxyz:", f"{client.camera.wxyz}")
        position_panel = client.gui.add_text("position:", f"{client.camera.position}")

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            with self.server.atomic():
                wxyz_panel.value = f"{client.camera.wxyz}"
                position_panel.value = f"{client.camera.position}"

        # live sliders apply
        @self.psize_slider.on_update
        def _(_):
            for handle in self.pc_handles:
                handle.point_size = self.psize_slider.value

        @self.camsize_slider.on_update
        def _(_):
            for handle in self.cam_handles:
                handle.scale = self.camsize_slider.value
                handle.line_thickness = 0.03 * handle.scale

    def add_camera(self, step):
        focal = self.cam_dict["focal"][step]
        pp = self.cam_dict["pp"][step]
        R = self.cam_dict["R"][step]
        t = self.cam_dict["t"][step]
        q = tf.SO3.from_matrix(R).wxyz
        # fov/asp (rough)
        fov = 2 * np.arctan(pp[1] / (focal + 1e-6))
        aspect = pp[0] / (pp[1] + 1e-6)
        self.cam_handles.append(
            self.server.add_camera_frustum(
                name=f"/frames/{step}/camera",
                fov=fov,
                aspect=aspect,
                wxyz=q,
                position=t,
                scale=self.camsize_slider.value,
                color=(50, 205, 50),
            )
        )

    def add_segmented_pc(self, step):
        pts_hw3 = self.pcs[step]           # (H,W,3)
        seg_hw = self.seg_masks[step]      # (H,W)
        conf_hw = self.confs[step] if (self.confs is not None and self.confs[step] is not None) else None

        pts, labels = map_mask_to_3d_points(
            pts3d_hw3=pts_hw3,
            segmask_hw=seg_hw,
            conf_hw=conf_hw,
            conf_thr=self.vis_threshold if conf_hw is not None else None,
            drop_background=True,
        )
        if pts.size == 0:
            return

        uniq = np.unique(labels)
        for lbl in uniq:
            if lbl == 0:
                continue
            m = (labels == lbl)
            if not np.any(m):
                continue
            base_color = np.asarray(self.label_color_map.get(int(lbl), (1.0, 0.0, 0.0)), dtype=np.float32)
            colors = np.tile(base_color, (int(m.sum()), 1))
            self.pc_handles.append(
                self.server.add_point_cloud(
                    name=f"/frames/{step}/seg_pts/label_{int(lbl)}",
                    points=pts[m],
                    colors=colors,
                    point_size=self.psize_slider.value,
                )
            )

    def add_full_pc(self, step):
        """Render the full point cloud with its original colors."""
        pts_hw3 = self.pcs[step]        # (H,W,3)
        rgb_hw3 = self.colors[step]     # (H,W,3) in [0,1]

        pts = pts_hw3.reshape(-1, 3)
        rgb = rgb_hw3.reshape(-1, 3)

        valid = np.isfinite(pts).all(axis=1) & ~(np.all(np.isclose(pts, 0.0), axis=1))
        pts = pts[valid]
        rgb = rgb[valid]

        if pts.size == 0:
            return

        self.pc_handles.append(
            self.server.add_point_cloud(
                name=f"/frames/{step}/full_pts",
                points=pts.astype(np.float32),
                colors=rgb.astype(np.float32),
                point_size=self.psize_slider.value * 0.8,  # Slightly smaller than the highlight cloud.
            )
        )

    def animate(self):
        with self.server.add_gui_folder("Playback"):
            gui_timestep = self.server.add_gui_slider(
                "Frame", min=0, max=self.num_frames - 1, step=1, initial_value=0
            )
            gui_next = self.server.add_gui_button("Next")
            gui_prev = self.server.add_gui_button("Prev")
            gui_play = self.server.add_gui_checkbox("Playing", False)
            gui_fps = self.server.add_gui_slider("FPS", min=1, max=60, step=1, initial_value=10)

        @gui_next.on_click
        def _(_): gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

        @gui_prev.on_click
        def _(_): gui_timestep.value = (gui_timestep.value - 1) % self.num_frames

        prev = gui_timestep.value

        self.server.add_frame("/frames", show_axes=False)
        self.frame_nodes = []
        # for i in range(self.num_frames):
        #     self.frame_nodes.append(self.server.add_frame(f"/frames/{i}", show_axes=False))
        #     self.add_segmented_pc(i)
        #     if self.show_camera:
        #         self.add_camera(i)

        for i in range(self.num_frames):
            self.frame_nodes.append(self.server.add_frame(f"/frames/{i}", show_axes=False))
            # 1) Render the full scene first.
            self.add_full_pc(i)
            # 2) Overlay the segmentation highlights.
            self.add_segmented_pc(i)
            if self.show_camera:
                self.add_camera(i)


        @gui_timestep.on_update
        def _(_):
            nonlocal prev
            with self.server.atomic():
                self.frame_nodes[gui_timestep.value].visible = True
                self.frame_nodes[prev].visible = False
            prev = gui_timestep.value
            self.server.flush()

        # Initial visibility: show only the first frame.
        for i, node in enumerate(self.frame_nodes):
            node.visible = (i == 0)

        while True:
            if gui_play.value:
                gui_timestep.value = (gui_timestep.value + 1) % self.num_frames
                # In 4D mode, only show the current frame.
                if self.fourd:
                    for i, node in enumerate(self.frame_nodes):
                        node.visible = (i == gui_timestep.value)
            else:
                # In 3D mode, keep all frames up to the current one visible.
                for i, node in enumerate(self.frame_nodes):
                    node.visible = (i <= gui_timestep.value) if not self.fourd else (i == gui_timestep.value)
            time.sleep(1.0 / gui_fps.value)

    def run(self):
        self.animate()
        while True:
            time.sleep(10.0)
