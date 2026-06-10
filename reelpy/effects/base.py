"""
File: Base.py
Class: BaseEffect
Description: Abstract base class for effect types 
Child Classes: 
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import copy

class BaseEffect(ABC):
    def __init__(self):
        # no shared state at the base level
        pass 
    
    # main API: shared apply method (entirely abstract)
    @abstractmethod
    def apply_frame(self, frame: np.ndarray, t: float) -> np.ndarray:
        pass 

    # method returning a copy of the effect, needed for _copy on clips (for indep copies of its effects instead of shared refs) 
    # DEFAULT: SHALLOW COPY (creates a new object but shares the same internal attribute refs, like duration, etc for FadeIn)
    # NOTE: needs to be overwritten to be a DEEP COPY by subclasses w mutable state that shouldn't share internal attributes (e.g. MotionTrailEffect which keeps a buffer attribute of recent frames, i.e. not stateless)
    def clone(self) -> BaseEffect:
        return copy.copy(self) # shallow
