"""
File: test_preview_player.py
Tests: PreviewPlayer
Source files: player.py

Test Setup:
 1) Unit tests: PreviewPlayer internals (_bounds, _seek, scrub_step) called
    directly against a small real SyntheticClip
 2) Mocked-loop tests: run() driven headlessly via cv2.waitKey.side_effect,
    against real Clip/SyntheticClip instances. seek_frames() itself is NOT
    faked, only observed via SeekFramesRecorder.
 3) Cross-cutting: seek_frames() timing/fade correctness vs export() which is the
    contract PreviewPlayer's entire scrub-accuracy story depends on.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock
from reelpy.preview.player import PreviewPlayer
from reelpy.clip.video import Clip
from reelpy.clip.synthetic import SyntheticClip
from reelpy.effects.fades import FadeOutEffect
from reelpy.io.reader import VideoReader
from tests.conftest import SAMPLE_3S_320x240_30FPS, make_preview_clip


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def mocked_cv2(monkeypatch):
    """
    Patches every cv2 call used inside run() and forces _has_display() to
    True, so the event loop can be driven headlessly and deterministically
    via cv2.waitKey.side_effect (a scripted list of "keys pressed").
    """
    monkeypatch.setattr(PreviewPlayer, "_has_display", lambda self: True) #hasdisplay always set to true
    mocks = {
        "namedWindow": Mock(),
        "imshow": Mock(),
        # resize mock preserves dtype/channels but honors the requested (w,h)
        "resize": Mock(side_effect=lambda arr, size: np.zeros((size[1], size[0], arr.shape[2]), dtype=arr.dtype)),
        "cvtColor": Mock(side_effect=lambda arr, code: arr),
        "waitKey": Mock(return_value=ord('q')),
        "destroyAllWindows": Mock(),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(cv2, name, mock) #mocking display fxns on openCV
    return mocks


# ── TIER 1: Unit-level tests (direct method calls) ─────

def test_init_state_untrimmed():
    """Fresh player: playhead at 0, unpaused, not ended, generator ready,
    scrub_step computed correctly."""
    player = PreviewPlayer(make_preview_clip(duration=3.0, fps=10.0))
    assert player.t == 0.0
    assert player.paused is False
    assert player._ended is False
    assert player._frame_gen is not None
    assert player.scrub_step_s == 0.5   # 3.0*0.01=0.03, floored to 0.5

def test_scrub_step_uses_trim_window_not_full_duration():
    """Regression test for the bug where scrub_step was computed from the
    FULL source duration instead of the trimmed window. A 100s source
    trimmed to 10s must use 10s for the 1% calculation, not 100s. """
    player = PreviewPlayer(make_preview_clip(duration=100.0, fps=10.0, start=5.0, end=15.0))
    assert player.scrub_step_s == pytest.approx(0.5)  # (15-5)*0.01=0.1, floored to 0.5

def test_bounds_untrimmed():
    player = PreviewPlayer(make_preview_clip(duration=5.0))
    assert player._bounds() == (0.0, 5.0)

def test_bounds_trimmed():
    player = PreviewPlayer(make_preview_clip(duration=5.0, start=1.0, end=3.0))
    assert player._bounds() == (1.0, 3.0)

def test_seek_clamps_trimmed():
    """Clamping must respect the TRIM window, not the full source range."""
    player = PreviewPlayer(make_preview_clip(duration=5.0, start=1.0, end=2.5))
    player._seek(0.0)
    assert player.t == 1.0
    player._seek(100.0)
    assert player.t == pytest.approx(2.5 - 0.01)

def test_seek_resets_ended_flag():
    """A fresh seek always clears _ended, even if the player was previously
    stuck at the end of the clip."""
    player = PreviewPlayer(make_preview_clip(duration=5.0))
    player._ended = True
    player._seek(1.0)
    assert player._ended is False

def test_seek_closes_previous_generator(seek_spy):
    """Every call to _seek() must close the generator it's replacing, so
    resources (e.g. an open VideoReader/av container) don't pile up while
    the old generator waits on garbage collection."""
    clip = make_preview_clip(duration=5.0)
    recorder = seek_spy(clip)          # attach BEFORE construction to catch __init__'s own _seek(0.0)
    player = PreviewPlayer(clip)
    assert recorder.close_count == 0   # nothing to close yet — this is the very first generator
    player._seek(1.0)
    assert recorder.close_count == 1
    player._seek(2.0)
    assert recorder.close_count == 2


# ── TIER 2: Mocked-loop tests (cv2 patched, driven via waitKey scripting) ──

def test_run_quits_immediately(mocked_cv2):
    mocked_cv2["waitKey"].side_effect = [ord('q')]
    player = PreviewPlayer(make_preview_clip(duration=3.0))
    player.run()
    mocked_cv2["namedWindow"].assert_called_once_with("Reelpy Preview", cv2.WINDOW_NORMAL)
    mocked_cv2["destroyAllWindows"].assert_called_once()

def test_run_prints_controls_on_start(mocked_cv2, capsys):
    mocked_cv2["waitKey"].side_effect = [ord('q')]
    PreviewPlayer(make_preview_clip(duration=3.0)).run()
    assert "SPACE: play/pause" in capsys.readouterr().out

def test_run_pause_stops_further_frame_pulls(seek_spy, mocked_cv2):
    clip = make_preview_clip(duration=5.0, fps=10.0)
    recorder = seek_spy(clip)
    mocked_cv2["waitKey"].side_effect = [ord(' '), 255, 255, ord('q')]  # pause, then idle keys
    player = PreviewPlayer(clip)
    player.run()
    assert recorder.next_count == 1
    assert player.paused is True

def test_run_scrub_forward_and_back(seek_spy, mocked_cv2):
    """
    Covers both scrub direction correctness AND that _seek() (and therefore
    generator close) fires exactly once per scrub keypress 
    """
    clip = make_preview_clip(duration=5.0, fps=10.0)   # scrub_step floors to 0.5
    recorder = seek_spy(clip)
    mocked_cv2["waitKey"].side_effect = [ord('d'), ord('a'), ord('q')]
    player = PreviewPlayer(clip)
    player.run()
    assert player.t == pytest.approx(0.0)
    assert recorder.seek_calls == pytest.approx([0.0, 0.5, 0.0])  # init + fwd + back
    assert recorder.close_count == 3   # 2 scrubs + 1 final cleanup on quit

def test_run_scrub_while_paused_still_updates_frame(mocked_cv2):
    """
    Scrubbing while paused is meant to redraw the new frame WITHOUT
    un-pausing (force_redraw bypasses the paused check for exactly one
    frame). Not covered by the play-then-scrub tests above, which never
    scrub from a paused state.
    """
    clip = make_preview_clip(duration=5.0, fps=10.0)
    mocked_cv2["waitKey"].side_effect = [ord(' '), ord('d'), ord('q')]  # pause, then scrub
    player = PreviewPlayer(clip)
    player.run()
    assert player.paused is True            # scrubbing doesn't un-pause
    assert player.t == pytest.approx(0.5)   # but the playhead still moved

def test_run_pause_toggle(mocked_cv2):
    mocked_cv2["waitKey"].side_effect = [ord(' '), ord(' '), ord('q')]
    player = PreviewPlayer(make_preview_clip(duration=3.0))
    player.run()
    assert player.paused is False  # toggled True then False

def test_run_natural_end_pauses_and_marks_ended(mocked_cv2):
    clip = make_preview_clip(duration=0.3, fps=10.0)  # yields t=0.0, 0.1, 0.2 then exhausts
    mocked_cv2["waitKey"].side_effect = [255, 255, 255, 255, ord('q')]
    player = PreviewPlayer(clip)
    player.run()
    assert player.paused is True
    assert player._ended is True
    assert player.t == pytest.approx(0.2)  # frozen on the last valid frame

def test_run_space_after_natural_end_restarts(seek_spy, mocked_cv2):
    """space after hitting the end restarts from trim start,
    rather than uselessly toggling pause on an exhausted generator."""
    clip = make_preview_clip(duration=0.3, fps=10.0)
    recorder = seek_spy(clip)
    mocked_cv2["waitKey"].side_effect = [255, 255, 255, 255, ord(' '), ord('q')]
    player = PreviewPlayer(clip)
    player.run()
    assert player.paused is False
    assert player._ended is False
    assert player.t == pytest.approx(0.0)
    assert recorder.seek_calls == pytest.approx([0.0, 0.0])  # init seek + restart seek

def test_run_natural_playback_stops_at_trim_end_even_if_generator_overruns(monkeypatch, mocked_cv2):
    """
    Defensive check: PreviewPlayer must never DISPLAY a frame past
    clip.end, even if seek_frames() itself doesn't stop there on its own.
    Both real seek_frames() implementations currently DO respect trim
    already, so to actually exercise PreviewPlayer's independent safety net
    here, we monkeypatch seek_frames() on this one real clip to deliberately
    ignore trim — simulating a hypothetical future clip type that doesn't
    self-limit.
    """
    clip = make_preview_clip(duration=5.0, fps=10.0, start=1.0, end=1.3)

    def overrunning_seek_frames(t):
        i = int(t * clip.fps)
        while i / clip.fps < clip.duration:  # deliberately ignores clip.end
            t_abs = i / clip.fps
            yield (np.zeros((clip.height, clip.width, 3), dtype=np.uint8), t_abs)
            i += 1

    monkeypatch.setattr(clip, "seek_frames", overrunning_seek_frames)
    mocked_cv2["waitKey"].side_effect = [255] * 10 + [ord('q')]
    player = PreviewPlayer(clip)
    player.run()
    assert player.t == pytest.approx(1.2, abs=0.01)   # last frame strictly inside [1.0, 1.3)
    assert player.paused is True
    assert player._ended is True

def test_run_closes_generator_on_quit(seek_spy, mocked_cv2):
    clip = make_preview_clip(duration=5.0)
    recorder = seek_spy(clip)
    mocked_cv2["waitKey"].side_effect = [ord('q')]
    PreviewPlayer(clip).run()
    assert recorder.close_count == 1  # the generator opened in __init__ still gets cleaned up

def test_run_headless_is_noop(mocked_cv2, monkeypatch, capsys):
    monkeypatch.setattr(PreviewPlayer, "_has_display", lambda self: False)
    player = PreviewPlayer(make_preview_clip(duration=3.0))
    with pytest.warns(UserWarning, match="No display available"):
        player.run()
    mocked_cv2["namedWindow"].assert_not_called()
    mocked_cv2["waitKey"].assert_not_called()
    assert capsys.readouterr().out == ""  # controls message shouldn't print if we never actually ran


# ── Smoke test against real decoded video (not just synthetic) ─────────────

@pytest.mark.routine
def test_run_smoke_with_real_video_clip(mocked_cv2):
    """Confirms PreviewPlayer works end-to-end against a genuinely
    decode-backed Clip, not just the cheap SyntheticClip used everywhere
    above. Loose assertions — keyframe-based seeking doesn't guarantee exact
    frame timing the way synthetic clips do."""
    clip = Clip(SAMPLE_3S_320x240_30FPS)
    mocked_cv2["waitKey"].side_effect = [ord('d'), ord('a'), ord(' '), ord(' '), ord('q')]
    player = PreviewPlayer(clip)
    player.run()  # should not raise
    assert 0.0 <= player.t <= clip.duration


# ── TIER 3: Cross-cutting — seek_frames() timing must match export() ───────

def _build_trimmed_faded(clip_type: str):
    if clip_type == "video":
        clip = Clip(SAMPLE_3S_320x240_30FPS)
    else:
        clip = SyntheticClip(320, 240, 30.0, 3.0, (200, 200, 200))
    return clip.trim(1.0, 3.0).apply(FadeOutEffect(1.0))  # 2s window, fades over its last 1s

@pytest.mark.routine
@pytest.mark.parametrize("clip_type", ["video", "synthetic"])
def test_seek_frames_matches_export_timing_with_fade(tmp_path, clip_type):
    """
    seek_frames() must yield ABSOLUTE timestamps whose fade math matches
    frames()/export() at the equivalent TRIM-RELATIVE timestamp. This is the
    exact contract PreviewPlayer's scrub accuracy depends on — if this
    fails, what you see while scrubbing won't match what actually exports.
    (FadeIn isn't tested here: its math has no clip_duration dependency, so
    it can't have this specific absolute-vs-relative failure mode.)
    """
    TRIM_START = 1.0
    clip = _build_trimmed_faded(clip_type)

    output = str(tmp_path / f"export_ref_{clip_type}.mp4")
    clip.export(output)
    with VideoReader(output) as reader:
        exported_frames = list(reader.frames())  # t is trim-relative, starts at 0

    for t_rel in [0.2, 0.9, 1.5, 1.9]:  # before, at, and during the fade (fade starts at t_rel=1.0)
        exported = min(exported_frames, key=lambda f: abs(f[1] - t_rel))
        t_abs = TRIM_START + t_rel
        gen = clip.seek_frames(t_abs)
        seek_frame, seek_t = next(gen)
        gen.close()

        assert seek_t == pytest.approx(t_abs, abs=0.05)
        assert np.allclose(seek_frame.mean(), exported[0].mean(), atol=10), (
            f"[{clip_type}] mismatch at t_rel={t_rel}: "
            f"seek_frames mean={seek_frame.mean():.1f} vs export mean={exported[0].mean():.1f}"
        )