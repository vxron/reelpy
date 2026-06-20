from reelpy.layers.base import BaseLayer
import numpy as np
from typing import Callable
import cv2

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
    
    def _copy(self, **overrides):
        color = overrides.get("color", self.color)
        shape = overrides.get("shape", self.shape)
        position = overrides.get("position", self.position)
        size = overrides.get("size", self.size)
        radius = overrides.get("radius", self.radius)
        points = overrides.get("points", self.points)
        thickness = overrides.get("thickness", self.thickness)
        name = overrides.get("name", self.name)
        opacity = overrides.get("opacity", self.opacity)
        effects=overrides.get("effects", list(self.effects))
        blend_mode=overrides.get("blend_mode", self.blend_mode)
        t_start=overrides.get("t_start", self.t_start)
        t_end=overrides.get("t_end", self.t_end)
        return ShapeLayer(
            shape, position, size, radius, points, color=color, thickness=thickness,
            name=name, opacity=opacity, effects=effects,
            blend_mode=blend_mode, t_start=t_start, t_end=t_end,
        )

    def _draw(self, canvas: np.ndarray, t: float):
        # (1) resolve any callable attributes
        position = self.position(t) if callable(self.position) else self.position
        if position is not None:
            position = (int(position[0]), int(position[1]))
        size = self.size(t) if callable(self.size) else self.size
        radius = self.radius(t) if callable(self.radius) else self.radius
        if radius is not None:
            radius = int(radius)
        points = self.points(t) if callable(self.points) else self.points
        if points is not None:
            points_arr = np.array(points, dtype=np.int32).reshape((-1, 1, 2)) # reshape to (N,1,2) for opencv compatibility
        
        # (2) temp canvases 
        rgb_initial_copy = canvas[:, :, :3].copy() # temp RGB canvas from the RGBA temp canvas
        binary_mask_canvas = np.zeros(canvas.shape[:2], dtype=np.uint8) # mask (starts off blank) so we know which pixels get drawn on
        
        # (3) convert to BGR for openCV (rev dir)
        bgr = cv2.cvtColor(rgb_initial_copy, cv2.COLOR_RGB2BGR)
        bgr_color = (self.color[2], self.color[1], self.color[0])
        
        # (4) draw the shape onto bgr (acc color) as well as onto the binaray mask (white)!
        if self.shape == "circle" and position is not None and radius is not None:
            cv2.circle(bgr, position, radius, bgr_color, self.thickness)
            cv2.circle(binary_mask_canvas, position, radius, (255,255,255), self.thickness)
        elif self.shape == "rectangle" and position is not None and size is not None:
            cv2.rectangle(bgr, position, (position[0]+size[0], position[1]+size[1]), bgr_color, self.thickness)
            cv2.rectangle(binary_mask_canvas, position, (position[0]+size[0], position[1]+size[1]), (255,255,255), self.thickness)
        elif self.shape == "line" and position is not None and size is not None:
            cv2.line(bgr, position, size, bgr_color, max(1, self.thickness))
            cv2.line(binary_mask_canvas, position, size, (255,255,255), max(1, self.thickness))
        elif self.shape == "polygon" and points is not None:
            if self.thickness == -1: #fill
                cv2.fillPoly(bgr, [points_arr], bgr_color)
                cv2.fillPoly(binary_mask_canvas, [points_arr], (255,255,255))
            else:
                cv2.polylines(bgr, [points_arr], isClosed=True, color=bgr_color, thickness=self.thickness)
                cv2.polylines(binary_mask_canvas, [points_arr], isClosed=True, color=(255,255,255), thickness=self.thickness)
        else:
            raise ValueError("Invalid shape configs in ShapeLayer.")

        # (5) convert back to rgb & write back into canvas channels
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas[:, :, :3] = rgb
        
        # (6) use the mask as the alpha channel directly :) (white pixels=opaque, black pixels=transparent)
        canvas[:, :, 3] = binary_mask_canvas

        return canvas
        