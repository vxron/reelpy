"""
File: media.py
Classes: ImageLayer, VideoLayer
"""
from __future__ import annotations
from collections.abc import Callable
import numpy as np
import cv2
from PIL import Image
from reelpy.layers.base import BaseLayer
from reelpy.clip.base import BaseClip
import os

# Module-level cache for images so we don't reload on every frame or layer (avoid repeated disk reads)
_image_cache: dict[str, Image.Image] = {} # cache key is just im path

# Module-level constants
_SEEK_THRESHOLD_S = 0.5 # time-gap threshold that decides whether to seek or advance frame-by-frame when the target timestamp is ahead of where the generator currently is


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
        im = _image_cache.get(self.image_path)
        if im is None:
            im = Image.open(self.image_path) # acc open on cache misses only
            _image_cache[self.image_path] = im

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
        return canvas


class VideoLayer(BaseLayer):
    def __init__(
        self,
        source: BaseClip,                                               # clip to render as layer
        position: tuple[int,int] | Callable[[float], tuple[int,int]],   # top-left on canvas, callable for animation
        size: tuple[int,int] | None = None,                             # resize dims, none = natural clip dims
        loop: bool = False,                                             # whether to loop the source clip when it ends 
        t_offset: float = 0.0,                                          # start the source clip at this offset 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.source = source
        self.position = position
        self.size = size
        self.loop = loop
        self.t_offset = t_offset
        # private attributes 
        self._frame_gen = None                        # generator, initialized lazily on first _draw call
        self._current_frame: np.ndarray | None = None # last successfully decoded frame
        self._gen_start_t: float = 0.0                # the t_source value at which _frame_gen was last initialized
        self._gen_current_t: float = 0.0              # tracks where gen currently is in layer source time, updated as frames are pulled

    def _copy(self, **overrides):
        source = overrides.get("source", self.source)
        position = overrides.get("position", self.position)
        size = overrides.get("size", self.size)
        loop = overrides.get("loop", self.loop)
        t_offset = overrides.get("t_offset", self.t_offset)
        name = overrides.get("name", self.name)
        opacity = overrides.get("opacity", self.opacity)
        effects=overrides.get("effects", list(self.effects))
        blend_mode=overrides.get("blend_mode", self.blend_mode)
        t_start=overrides.get("t_start", self.t_start)
        t_end=overrides.get("t_end", self.t_end)
        return VideoLayer(
            source, position, size, loop, t_offset,
            name=name, opacity=opacity, effects=effects,
            blend_mode=blend_mode, t_start=t_start, t_end=t_end,
        )
    
    def _init_generator(self, t_source: float) -> None:
        # called from _draw whenever a new generator is needed
        self._frame_gen = self.source.seek_frames(t_source) # starts at t_source
        self._gen_start_t = t_source
        self._gen_current_t = t_source

    def _get_source_t(self, t: float) -> float:
        # computes the source clip time from the parent clip time (passed in t)
        t_source = t + self.t_offset
        # get source duration
        src_dur = self.source.metadata()["duration"]
        if self.loop: 
            # wrap for looping
            t_source = t_source % src_dur
        # clamp t_source
        t_source = max(0.0, min(t_source, src_dur - 0.01)) 
        return t_source
    
    def _draw(self, canvas: np.ndarray, t: float) -> np.ndarray:
        # (1) resolve position
        position = self.position(t) if callable(self.position) else self.position
        position = int(position[0]), int(position[1])
        # (2) compute source time from parent time, t
        t_source = self._get_source_t(t)
        # (3) decide whether to reinitialize or advance the generator 
        if(
            self._frame_gen is None # first call ever, need to init gen
            or t_source < self._gen_current_t # backward seek detected (user scrubbed back, or loop wrapped around)
            or t_source - self._gen_current_t > _SEEK_THRESHOLD_S # large forward jump detected
        ):
            self._init_generator(t_source)
        # (4) advance gen to target frame
        # pulling frames until we reach a frame at or past t_source
        assert self._frame_gen is not None
        try:
            first_arr, frame_t = next(self._frame_gen)
            self._current_frame = first_arr
            self._gen_current_t = frame_t
        except StopIteration:
            pass # fall through to None check: source is genuinely empty
            frame_t = t_source # skip while loop if first pull fails
        
        while frame_t < t_source:
            try: # try pulling next frame 
                arr, frame_t = next(self._frame_gen)
                self._current_frame = arr
                self._gen_current_t = frame_t
            except StopIteration:
                # generator exhausted
                if self.loop:
                    self._init_generator(self.t_offset)
                #else: 
                    # if not looping, just use self.current_frame (freeze last frame)
                    # don't have to do anything, use last stored
                break # exit the while loop upon exhaustion

        if self._current_frame is None:
            # no frame was ever decoded (source clip empty)
            return canvas
        # (5) resize if needed
        frame_h, frame_w = self._current_frame.shape[:2]
        if self.size is not None:
            self._current_frame = cv2.resize(self._current_frame, (self.size[0], self.size[1]))
            frame_h = self.size[1]
            frame_w = self.size[0]
        # (6) compute crop bounds (intersection of video with canvas bounds so we can crop that part of vid)
        canvas_h, canvas_w = canvas.shape[:2]
        x, y = position
        src_x0 = max(0, -x)        # pixels to skip from left of vid (if x < 0)
        src_y0 = max(0, -y)        # pixels to skip from top of vid (if y < 0)
        dst_x0 = max(0, x)         # where on canvas to start writing (x)
        dst_y0 = max(0, y)         # where on canvas to start writing (y)
        dst_x1 = min(canvas_w, x + frame_w)  # where on canvas to stop writing (x)
        dst_y1 = min(canvas_h, y + frame_h)  # where on canvas to stop writing (y)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return canvas # image is entirely off-canvas 
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        # (7) write into canvas
        canvas[dst_y0:dst_y1, dst_x0:dst_x1, :3] = self._current_frame[src_y0:src_y1, src_x0:src_x1, :3]
        canvas[dst_y0:dst_y1, dst_x0:dst_x1, 3] = 255
        return canvas
        


