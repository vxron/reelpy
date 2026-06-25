"""
File: text.py
Classes: TextLayer
Description: Pillow-backed text rendering layer.
"""
from __future__ import annotations
from collections.abc import Callable
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reelpy.layers.base import BaseLayer

# module-level cache for .ttf files from disk that all textlayers can use
_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {} # keyed by (font_path, font_size)
# _draw will check this cache first, only load from disk on a cache miss 

class TextLayer(BaseLayer):
    def __init__(
        self,
        text: str | Callable[[float], str],                                          # the string to render (callable enables live counters, typewriter effects, etc)
        position: tuple[int, int] | Callable[[float], tuple[int,int]],               # top-left corner (or anchor point depending on anchor attr) of the txt, callable option for animation
        font_path: str | None = None,                                                # path to a .ttf/.otf file. None falls back to a bundled default font
        font_size: int | Callable[[float], int] = 12,                                # font size in pixels
        color: tuple[int,int,int] | Callable[[float], tuple[int,int,int]] = (0,0,0), # RGB text color
        anchor: str = "la",                                                          # pillow anchor string controlling what position refers to (e.g. "la" = left-ascender, "mm" = middle)
        max_width: int | None = None,                                                # pixel width to wrap within (None = single-line, no wrapping unless explicit \n passed)
        line_spacing: int | float = 4,                                               # pixels between wrapped lines
        **kwargs
    ):
        super().__init__(**kwargs)
        self.text = text
        self.position = position
        self.font_path = font_path
        self.font_size = font_size
        self.color = color
        self.anchor = anchor
        self.max_width = max_width
        self.line_spacing = line_spacing
    
    def _copy(
        self,
        **overrides
    ):
        color = overrides.get("color", self.color)
        text = overrides.get("text", self.text)
        position = overrides.get("position", self.position)
        font_path = overrides.get("font_path", self.font_path)
        font_size = overrides.get("font_size", self.font_size)
        anchor = overrides.get("anchor", self.anchor)
        name = overrides.get("name", self.name)
        opacity = overrides.get("opacity", self.opacity)
        effects=overrides.get("effects", list(self.effects))
        blend_mode=overrides.get("blend_mode", self.blend_mode)
        t_start=overrides.get("t_start", self.t_start)
        t_end=overrides.get("t_end", self.t_end)
        return TextLayer(
            text, position, font_path, font_size, color, anchor,
            name=name, opacity=opacity, effects=effects,
            blend_mode=blend_mode, t_start=t_start, t_end=t_end,
        )
    
    def _wrap_text(self, text: str, font) -> str:
        # only called when max_width is set : add necessary \n within text msg to ensure max_width is never exceeded
        if self.max_width is None:
            return text # no auto-wrap
        
        words = text.split()
        candidate_line = ""
        wrapped_text = ""
        
        for word in words:
            # measur current line's width if adding word
            curr = font.getlength(candidate_line + " " + word)
            if curr >= self.max_width:
                # start a new line with word
                wrapped_text = wrapped_text + candidate_line + "\n"
                candidate_line = word # reset 
            else:
                # word fits on curr line
                candidate_line = candidate_line + " " + word
        # when all words r done, check if it was just a single line
        if wrapped_text == "":
            wrapped_text = candidate_line
        else:
            wrapped_text = wrapped_text + candidate_line # wtv left in candidate line
        return wrapped_text

    def _draw(self, canvas: np.ndarray, t: float):
        # (1) resolve ever callable attribute at this t & make everything ints
        text = self.text(t) if callable(self.text) else self.text
        position = self.position(t) if callable(self.position) else self.position
        font_size = int(self.font_size(t) if callable(self.font_size) else self.font_size)
        color = self.color(t) if callable(self.color) else self.color
        color = tuple(int(c) for c in color) 
        position = (int(position[0]), int(position[1]))
        # (2) look up or load the font
        if self.font_path is None:
            # default
            font = ImageFont.load_default()
        else:
            font = _font_cache.get((self.font_path, font_size), None)
            if font is None:
                font = ImageFont.truetype(self.font_path, font_size)
                _font_cache[(self.font_path, font_size)] = font
        # wrapping if applicable
        if self.max_width is not None:
            text = self._wrap_text(text, font)
        # (3) draw
        img = Image.fromarray(canvas, mode="RGBA")
        draw = ImageDraw.Draw(img)
        draw.text(position, text, font=font, fill=(*color,255), anchor=self.anchor, spacing=self.line_spacing) # draw opaque
        canvas = np.array(img) # convert back to NumPy
        return canvas
