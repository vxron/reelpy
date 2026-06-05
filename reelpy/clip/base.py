"""
File: Base.py
Class: BaseClip
Description: Abstract base class for clip types (sequence of frames)
Child Classes: Clip, SyntheticClip
"""
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Self, Dict
import numpy as np

# Abstract Base Class
class BaseClip(ABC):
    # Shared by subclasses by calling super().__init__()
    def __init__(self):
        self.start: float = 0.0
        self.end: float | None = None
        self.layers: list = []
        self.effects: list = []

    @abstractmethod
    # abstract helper to copy object rather than just creating ptr to same obj
    # (purpose is to avoid mutating self)
    def _copy(self, **overrides) -> Self:
        # the ** means "collect all keyword arguments into a dict called overrides"
        pass

    # START LAYER METHODS

    def add_layer(self, layer, index=None) -> Self:
        new_list = list(self.layers) # Copy to avoid mutating original list
        if index is None:
            new_list.append(layer)
        else:
            new_list.insert(index, layer)
        # create new instance to return with updated list
        result = self._copy()
        result.layers = new_list
        return result
    
    def add_layers(self, layers: list) -> Self:
        result = self
        for layer in layers:
            result = result.add_layer(layer) # capture rtn value
        return result

    def move_layer(self, from_index: int, to_index: int) -> Self:
        result = self._copy()
        new_layers = list(self.layers)
        layer = new_layers.pop(from_index)
        new_layers.insert(to_index, layer)
        result.layers = new_layers
        return result

    def swap_layers(self, index_a: int, index_b: int) -> Self:
        result = self._copy()
        # swap items
        tmp = result.layers[index_a]
        result.layers[index_a] = result.layers[index_b]
        result.layers[index_b] = tmp
        return result
    
    def get_layer(self, name: str): # TODO: update return type once BaseLayer class is made
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None
    
    def list_layers(self) -> str:
        n = len(self.layers)
        if n == 0:
            return "Layer stack (0 layers)"
        sep = "─" * 55
        lines = [f"Layer stack ({n} layer{'s' if n != 1 else ''})", sep]
        for i, layer in reversed(list(enumerate(self.layers))):
            tag = "↑ top   " if i == n - 1 else ("↓ bottom" if i == 0 else "        ")
            row = (
                f"[{i}] {tag}  "
                f"{type(layer).__name__:<14} "
                f'"{layer.name}"  '
                f"opacity={layer.opacity}  "
                f"blend={layer.blend_mode}  "
                f"fx={len(layer.effects)}"
            )
            lines.append(row)
        lines.append(sep)
        return "\n".join(lines)
    
    # START EFFECT METHODS
    
    def apply(self, effect) -> Self:
        # Append effect to the effect chain
        new_effect_list = list(self.effects)
        new_effect_list.append(effect)
        result = self._copy()
        result.effects = new_effect_list
        return result
    
    def apply_preset(self, preset):
        # TODO once Preset class is made
        pass

    # START OTHER METHODS

    def trim(self, start: float, end: float | None) -> Self:
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if end is not None and end <= start:
            raise ValueError(f"end must be > start, got end={end}, start={start}")
        result = self._copy()
        result.start = start
        result.end = end
        return result
    
    @abstractmethod
    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        pass

    @abstractmethod
    def export(self, export_path: str, **kwargs) -> None:
        pass

    @abstractmethod
    def preview(self) -> None:
        pass

    @abstractmethod 
    def metadata(self) -> Dict:
        pass
