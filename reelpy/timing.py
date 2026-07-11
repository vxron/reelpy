"""
File: timing.py
Description: Shared time-reference-frame types used across the
clip/effect/layer/preview system.

AbsoluteTime: position in a clip's ORIGINAL, untrimmed timeline.
              t=0 is the start of the full source. Used by seek_frames()
              and PreviewPlayer. Trim has no effect on this clock.

TrimTime:     position relative to a clip's trim start (self.start).
              t=0 is the start of the EXPORTED/edited clip. This is the
              clock effects and layers expect, and what frames() yields.

NewType wrappers around float — zero runtime cost, purely a static-analysis aid.
"""
from typing import NewType

AbsoluteTime = NewType("AbsoluteTime", float)
TrimTime = NewType("TrimTime", float)

def to_trim(t_abs: AbsoluteTime, clip_start: AbsoluteTime) -> TrimTime:
    """Convert an absolute timestamp to trim-relative."""
    return TrimTime(t_abs - clip_start)

def to_absolute(t_trim: TrimTime, clip_start: AbsoluteTime) -> AbsoluteTime:
    """Convert a trim-relative timestamp to absolute."""
    return AbsoluteTime(t_trim + clip_start)