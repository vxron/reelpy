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
import warnings
import os
import cv2
import numpy as np
from reelpy.clip.base import BaseClip


class PreviewPlayer():
    def __init__(self, clip: BaseClip):
        self.clip: BaseClip = clip
        self.t: float = 0.0                                                   # current playhead position in seconds
        self.paused: bool = False
        self.scrub_step_s: float = max(0.5, clip.metadata()["duration"]*0.01) # seconds fwd/bwd per arrow-key press, 1% of the total clip duration, w/ minimum of 0.5s
        self._current_frame: np.ndarray | None = None                         # last decoded frame
        self._seek(0.0)                                                       # live generator: init _frame_gen at t=0, gets recreated on scrubs

    def _seek(self, t: float) -> None:
        duration = self.clip.metadata()["duration"]
        end = self.clip.metadata().get("end") or duration
        start = self.clip.metadata().get("start") or 0.0
        self.t = max(start, min(t, end - 0.01))  # clamp here so self.t stays accurate
        self._frame_gen = self.clip.seek_frames(self.t)

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
        meta = self.clip.metadata()
        fps = meta["fps"]
        delay_ms = max(1, int(1000/fps)) # min 1ms btwn frames
        cv2.namedWindow("Reelpy Preview", cv2.WINDOW_NORMAL)

        # LOOP
        stopped = False
        force_redraw = False
        while not stopped:
            try:
                if not self.paused or force_redraw:
                    try: # try to gen & store next frame
                        arr, t = next(self._frame_gen)
                        self.t = t
                        self._current_frame = arr
                    except StopIteration: # once generator is exhausted
                        self.paused = True
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
            except BaseException as e:
                #print(f"LOOP CRASHED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                break
        cv2.destroyAllWindows()


