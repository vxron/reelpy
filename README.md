# Reelpy

Reelpy is a Python library for programmatic video editing. It is built on PyAV (direct in-process bindings to FFmpeg) rather than spawning FFmpeg as a subprocess the way MoviePy does. This makes it faster, more memory-efficient, and gives it access to the full codec pipeline without the overhead of piping frames across process boundaries.

---

## Background

Python's video editing ecosystem has a gap. The low-level tools (PyAV, OpenCV) give you full control but no editing abstractions — no clip model, no effects, no timeline. The high-level tool (MoviePy) gives you a usable API but runs everything through a slow FFmpeg subprocess, has no interactive preview, and provides only a thin set of effects with no composable pipeline structure.

Reelpy is designed to sit between those two. It uses PyAV directly for I/O so it inherits PyAV's performance characteristics, then builds a compositing and editing layer on top of it that MoviePy does not offer.

---

## Core concepts

**Clip** — wraps a video file. Holds a list of layers and a list of effects. Nothing is decoded until `.preview()` or `.export()` is called. Every method (`trim`, `add_layer`, `apply`) returns a new Clip — the source file is never modified.

**SyntheticClip** — generates frames programmatically rather than reading from a file. The starting canvas is a solid color; layers are composited on top of each frame. Accepts an optional audio file. Fully interchangeable with `Clip` everywhere as both inherit from `BaseClip`.

**Layers** — drawn bottom-to-top onto the frame before effects run. Each layer renders into a private RGBA temp canvas, runs its own per-layer effect chain, then alpha-composites the result onto the main canvas. Available layer types: `SolidLayer`, `ShapeLayer`, `PathLayer`, `TextLayer`, `ImageLayer`, `VideoLayer`, `Group`. Layers support animated position, opacity, and visibility windows via callables.

**Effects** — transform the entire composited frame. Run after all layers have rendered. Available effects: `FadeIn`, `FadeOut`, `ColorGrade`, `Resize`, `TextOverlay`, `MotionTrailEffect`.

**Timeline** — concatenates any mix of `Clip` and `SyntheticClip` into a single output. Handles timestamp offsetting, per-clip audio, and stretch strategies for fitting clip duration to audio length (`freeze`, `slow`, `freeze_slow`).

**Animation** — any layer attribute that accepts a position, size, rotation, or opacity also accepts a callable `f(t) -> value`, where `t` is seconds elapsed. The `animation` module provides `Tween`, `TweenVec2`, and `TweenSequence` for common interpolation patterns. The `utils` module provides `Joint` for skeleton-based character animation and `repeat()` / `burst_positions()` for particle systems.

**Preview** — opens an OpenCV window showing the clip at half resolution. `SPACE` plays/pauses, `←` `→` scrubs by one second, `Q` closes. Works on both `Clip` and `SyntheticClip`.

---

## Install

```bash
pip install reelpy
```

Requires Python 3.11+. PyAV ships with FFmpeg bundled — no separate FFmpeg install needed.

For development:

```bash
git clone git@gitlab.com:yourusername/reelpy.git
cd reelpy
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

---

## Status

Active development. API is unstable until v1.0.