"""
File: player.py
Class: PreviewPlayer
Description: scrubber to view any BaseClip using OpenCV for display
Specs:
     -runs at actual clip FPS via OpenCV wait timing
     -accepts any BaseClip
     -SPACE for play/pause, arrows scrub fwd/bwd, Q for quit
     -half resolution for speed
     -leverage trim/seek primitives so we don't re-decode everything up to target on every scrub
"""
from __future__ import annotations
from collections.abc import Generator
import warnings
import os
import cv2
import numpy as np
from reelpy.clip.base import BaseClip
from reelpy.timing import AbsoluteTime


class PreviewPlayer():
    def __init__(self, clip: BaseClip):
        self.clip: BaseClip = clip
        self.t: AbsoluteTime = AbsoluteTime(0.0)                              # current playhead position in seconds
        self.paused: bool = False
        self._ended: bool = False                                             # true when generator is exhausted: distinguishes 'hit the end' from 'user paused'
        self.scrub_step_s: float = self._compute_scrub_step()                 # seconds fwd/bwd per arrow-key press, 1% of the total clip duration, w/ minimum of 0.5s
        self._current_frame: np.ndarray | None = None                         # last decoded frame
        self._frame_gen: Generator[tuple[np.ndarray, AbsoluteTime], None, None] | None = None
        self._seek(AbsoluteTime(0.0))                                                       # live generator: init _frame_gen at t=0, gets recreated on scrubs

    def _bounds(self) -> tuple[AbsoluteTime, AbsoluteTime]:
        meta = self.clip.metadata()
        duration = meta["duration"]
        start = meta.get("start") or AbsoluteTime(0.0)
        end = meta.get("end") or AbsoluteTime(duration)
        return start, end
    
    def _compute_scrub_step(self) -> float:
        start, end = self._bounds()
        return max(0.5, (end - start) * 0.01) # 1% of EFFECTIVE CLIP DUR (trimmed)

    def _seek(self, t: AbsoluteTime) -> None:
        start, end = self._bounds()
        safe_end = max(start, end - 0.01)  # guards against end-0.01 < start on trims shorter than 0.01s
        self.t = AbsoluteTime(max(start, min(t, safe_end)))  # clamp here so self.t stays accurate
        if self._frame_gen is not None:
            # force cleanup now to avoid having a bunch of gen objs open at once
            self._frame_gen.close()
        self._frame_gen = self.clip.seek_frames(self.t)
        self._ended = False  # fresh generator 

    def _has_display(self) -> bool:
        # Headless detection helper: can we display openCV??
        if os.name == "posix" and "DISPLAY" not in os.environ: # check on WSL/Lin
            return False
        return True # aassume macOS/Windows native always ok

    def run(self) -> None:        
        # Main event loop
        if not self._has_display():
            warnings.warn("No display available — preview() is a no-op in this environment.")
            return
        print("Reelpy Preview | SPACE: play/pause | A/D: scrub (arrow keys may not work on Windows) | Q: quit")

        meta = self.clip.metadata()
        fps = meta["fps"]
        delay_ms = max(1, int(1000/fps)) # min 1ms btwn frames
        start, end = self._bounds()
        self.scrub_step_s = self._compute_scrub_step() # scrub step should be based on effective duration window, not original untrimmed
        cv2.namedWindow("Reelpy Preview", cv2.WINDOW_NORMAL)

        # LOOP
        try:
            stopped = False
            force_redraw = False
            while not stopped:
                try:
                    if not self.paused or force_redraw:
                        assert self._frame_gen is not None
                        try: # try to gen & store next frame
                            arr, t = next(self._frame_gen)
                            if t>= end:
                                # trimmed clips must stop playback at their own end,
                                # even if seek_frames() itself iterates past it
                                self.paused = True
                                self._ended = True
                            else:
                                self.t = t
                                self._current_frame = arr
                        except StopIteration: # once generator is exhausted
                            self.paused = True
                            self._ended = True
                        #print(f"t={self.t:.2f} paused={self.paused} frame_shape={self._current_frame.shape if self._current_frame is not None else None}")

                    force_redraw = False # consumed

                    if self._current_frame is not None: # redraw current frame if paused, otherwise advances to next
                        h, w = self._current_frame.shape[:2]
                        display_frame = cv2.resize(self._current_frame, (w // 2, h // 2)) # half res
                        bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Reelpy Preview", bgr)

                    # key handling happens after imshow since it requires open cv2 window
                    key = cv2.waitKey(delay_ms) & 0xFF     # LET USER PRESS CONTROL KEYS: waitKey returns an integer: num code of wtv keyboard key was pressed by user
                    #if key != 255:
                        #print(f"RAW KEY: {key}, type={type(key)}")
                    if key == ord('q'): # num 113, q = QUIT
                        stopped = True
                        break
                    elif key == ord(' '):                  # space = pause/play
                        if self.paused and self._ended:
                            # space bar replays from beginning 
                            self._seek(start)
                            self.paused = False
                        else:
                            self.paused = not self.paused      # TOGGLE
                    elif key == 81 or key == ord('a'):     # LEFT ARROW
                        #print("reached inside left key elif")
                        #print(f"BEFORE SCRUB: self.t={self.t}")
                        self.t -= self.scrub_step_s
                        #print(f"AFTER SUBTRACT: self.t={self.t}")
                        self._seek(self.t)
                        #print(f"AFTER RESTART: self.t={self.t}")
                        force_redraw = True   # pull+show one frame from the new position, regardless of pause state
                    elif key == 83 or key == ord('d'):     # RIGHT ARROW
                        self.t += self.scrub_step_s
                        self._seek(self.t)
                        force_redraw = True
                except Exception as e:
                    #print(f"LOOP CRASHED: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        finally:
            # guaranteed cleanup
            if self._frame_gen is not None:
                self._frame_gen.close()
            cv2.destroyAllWindows()

