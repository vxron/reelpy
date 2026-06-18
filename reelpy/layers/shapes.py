from reelpy.layers.base import BaseLayer
import numpy as np
from typing import Callable

class SolidLayer(BaseLayer): 
    def __init__(
        self,
        color: tuple[int, int, int] = (0,0,0),          # RGB fill color
        rect: tuple[int, int, int, int] | None = None,  # (x,y,width,height) of the rectangle to fill. None = fill entire canvas.
        **kwargs                                        # passes name, opacity, effects, blend_mode etc up to BaseLayer
    ):
        super().__init__(**kwargs)
        self.color = color
        self.rect = rect
    
    def _copy(self, **overrides):
        color = overrides.get("color", self.color)
        rect = overrides.get("rect", self.rect)
        name = overrides.get("name", self.name)
        opacity = overrides.get("opacity", self.opacity)
        effects=overrides.get("effects", list(self.effects))
        blend_mode=overrides.get("blend_mode", self.blend_mode)
        t_start=overrides.get("t_start", self.t_start)
        t_end=overrides.get("t_end", self.t_end)
        return SolidLayer(
            color, rect,
            name=name, opacity=opacity, effects=effects,
            blend_mode=blend_mode, t_start=t_start, t_end=t_end,
        )
    
    def _draw(self, canvas: np.ndarray, t: float):
        if self.rect is None:
            #fill entire canvas
            canvas[:, :, :3] = self.color
            canvas[:, :, 3] = 255 # full opacity
        else:
            x, y, w, h = self.rect # unpack
            canvas[y:y+h, x:x+w, :3] = self.color # fill rect
            canvas[y:y+h, x:x+w, 3] = 255
        return canvas
    

class ShapeLayer(BaseLayer):
    """Supported Shapes: circle, rectangle, line, polygon"""
    def __init__(
        self,
        shape: str,                                                                     # required: circle, rectangle, line or polygon
        position: tuple[int,int] | Callable[[float],tuple[int,int]] | None,             # (x,y) center for circle, top-left for rectangle, start point for line, callable if t-dep
        size: tuple[int, int] | Callable[[float],tuple[int,int]] | None,                # (width,height) for rect, (x2,y2) end point for line, None otherwise, callable if t-dep
        radius: int | Callable[[float], int] | None,                                    # radius in pixels, only for circle (None for others), callable if t-dep
        points: list[tuple[int,int]] | Callable[[float], list[tuple[int,int]]] | None,  # vertex list for polygon, None for others, callable if t-dependency                                      
        color: tuple[int, int, int] = (0,0,0),                                          # RGB fill color
        thickness: int = -1,                                                            # stroke thickness in pixels. -1 = filled (default)
        **kwargs                                                                        # passes name, opacity, effects, blend_mode etc up to BaseLayer
    ):
        super().__init__(**kwargs)
        if shape not in ["circle", "rectangle", "polygon", "line"]:
            raise ValueError("Invalid shape passed to ShapeLayer. Must be one of circle, rectangle, polygon, line.")
        self.shape = shape
        self.color = color
        self.position = position
        self.size = size
        self.radius = radius
        self.points = points
        self.thickness = thickness 
