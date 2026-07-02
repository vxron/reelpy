"""
File: media.py
Classes: ImageLayer, VideoLayer
"""
from __future__ import annotations
from collections.abc import Callable
import numpy as np
from PIL import Image
from reelpy.layers.base import BaseLayer
import os

# Module-level cache for images so we don't reload on every frame or layer (avoid repeated disk reads)
_image_cache: dict[str, Image.Image] = {} # cache key is just im path


class ImageLayer(BaseLayer):
    def __init__(
        self, 
        image_path: str,                                                # path to the image file (PNG, JPG, WebP, etc)
        position: tuple[int,int] | Callable[[float], tuple[int,int]],   # top-left corner of the image on the canvas, callable for animated pos
        size: tuple[int, int] | None,                                   # target (width,height) to resize to. None = use natural image dimensions
        **kwargs
    ):
        super().__init__(**kwargs)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"ImageLayer could not find image file: '{image_path}'")
        self.image_path = image_path
        self.position = position
        self.size = size

    def _copy(self, **overrides):
        image_path = overrides.get("image_path", self.image_path)
        position = overrides.get("position", self.position)
        size = overrides.get("size", self.size)
        name = overrides.get("name", self.name)
        opacity = overrides.get("opacity", self.opacity)
        effects=overrides.get("effects", list(self.effects))
        blend_mode=overrides.get("blend_mode", self.blend_mode)
        t_start=overrides.get("t_start", self.t_start)
        t_end=overrides.get("t_end", self.t_end)
        return ImageLayer(
            image_path, position, size,
            name=name, opacity=opacity, effects=effects,
            blend_mode=blend_mode, t_start=t_start, t_end=t_end,
        )

    def _draw(self, canvas: np.ndarray, t:float):
        # (1) resolve position callable
        position = self.position(t) if callable(self.position) else self.position 
        position = (int(position[0]), int(position[1]))
        
        # (2) load or retrieve im from cache
        im = _image_cache.get(self.image_path, Image.open(self.image_path))
        # make a copy to avoid mutations
        im_copy = im.copy()
        im_copy = im_copy.convert("RGBA")
        
        # (3) resize if self.size is set
        if self.size is not None:
            im_copy = im_copy.resize(self.size, Image.Resampling.LANCZOS)
        
        # (4) compute crop bounds (intersection of image with canvas bounds so we can crop that part of im)
        img_w, img_h = im_copy.size
        canvas_h, canvas_w = canvas.shape[:2]
        x, y = position
        src_x0 = max(0, -x)        # pixels to skip from left of image (if x < 0)
        src_y0 = max(0, -y)        # pixels to skip from top of image (if y < 0)
        dst_x0 = max(0, x)         # where on canvas to start writing (x)
        dst_y0 = max(0, y)         # where on canvas to start writing (y)
        dst_x1 = min(canvas_w, x + img_w)  # where on canvas to stop writing (x)
        dst_y1 = min(canvas_h, y + img_h)  # where on canvas to stop writing (y)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return canvas # image is entirely off-canvas 
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        # (5) crop & write into canvas (RGB/alpha separate)
        img_cropped = im_copy.crop((src_x0, src_y0, src_x1, src_y1))
        img_arr = np.array(img_cropped)
        canvas[dst_y0:dst_y1, dst_x0:dst_x1, :3] = img_arr[:,:,:3] # RGB
        canvas[dst_y0:dst_y1, dst_x0:dst_x1, 3] = img_arr[:,:,3] # alpha (preserves PNG transparency)



