"""
File: test_clip.py
Tests: BaseClip, Clip, SyntheticClip
Source files: clip/
"""
import pytest
import numpy as np
import av
from reelpy.io.reader import VideoReader
from reelpy.io.writer import VideoWriter
from reelpy.clip.base import BaseClip
from reelpy.clip.synthetic import SyntheticClip
from reelpy.clip.video import Clip
from reelpy.exceptions import InvalidVideoError
from reelpy.config import config
from tests.conftest import (
    VIDEO_FIXTURES, SAMPLE_3S_320x240_30FPS, SAMPLE_VERT_AUDIO, AUDIO_FIXTURES,
    assert_frame_valid, assert_timestamps_valid, INVALID_CLIP_CONFIGS, SYNTHETIC_FIXTURES,
    make_synthetic_from_fixture, make_video_from_fixture
)

# --------------------------------------- CLIP TESTS ---------------------------------------------------------------

def test_clip_nonexistent_file():
    # source of this raise should be VideoReader on attempting to use av.open(path)
    with pytest.raises(InvalidVideoError):
        Clip("tests/fake_path.mp4")

@pytest.mark.parametrize("file_path,width,height,fps,dur",VIDEO_FIXTURES)
def test_clip_init(file_path: str, width: int, height: int, fps: float, dur: float):
    clip = Clip(file_path)
    assert clip.fps == fps
    assert clip.width == width
    assert clip.height == height
    assert clip.duration == pytest.approx(dur, abs=0.1)
    assert clip.start == 0.0 # default
    assert clip.end is None # default
    assert clip.layers == [] # default
    assert clip.effects == [] # default

@pytest.mark.parametrize("file_path,width,height,fps,dur",VIDEO_FIXTURES)
def test_clip_trim(file_path: str, width: int, height: int, fps: float, dur: float):
    clip = Clip(file_path)
    trimmed = clip.trim (1.0, 2.0)
    assert trimmed.start == 1.0
    assert trimmed.end == 2.0
    assert trimmed.path == clip.path # source path preserved
    assert trimmed.fps == clip.fps # metadata preserved
    # non-destructive checks (original unchanged)
    assert clip.start == 0.0 
    assert clip.end is None
    assert clip.layers is not trimmed.layers # lists should be indep copies, not same ref
    assert clip.effects is not trimmed.effects

def test_clip_invalid_trims():
    clip = Clip(SAMPLE_3S_320x240_30FPS)
    with pytest.raises(ValueError):
        clip.trim(-1.0, 2.0)
    with pytest.raises(ValueError):
        clip.trim(2.0, 1.0)
    with pytest.raises(ValueError):
        clip.trim(1.0, 1.0)

@pytest.mark.parametrize("file_path,width,height,fps,dur",VIDEO_FIXTURES)
def test_clip_frames(file_path: str, width: int, height: int, fps: float, dur: float):
    clip = Clip(file_path)
    frames = clip.frames()
    # get first frame to validate metadata
    frame1 = next(frames)
    assert_frame_valid(frame1[0], width, height)
    # check all frames & timestamps
    all_frames = [frame1] + list(frames)
    timestamps = [t for (arr,t) in all_frames]
    assert_timestamps_valid(timestamps, dur)

@pytest.mark.parametrize("file_path,width,height,fps,dur",VIDEO_FIXTURES)
def test_clip_frames_with_trim(file_path: str, width: int, height: int, fps: float, dur: float):
    clip = Clip(file_path)
    trimmed = clip.trim(0.8, 1.65)
    all_frames_trimmed = list(trimmed.frames())
    timestamps_trimmed = [t for (arr,t) in all_frames_trimmed]
    assert timestamps_trimmed[0] == pytest.approx(0.0, abs=0.1) # t is relative to trim start
    assert timestamps_trimmed[-1] == pytest.approx(1.65-0.8, abs=0.1)
    assert all(t<(1.65-0.8) for t in timestamps_trimmed) # nothn outside trim window

def test_clip_export(tmp_path):
    clip = Clip(SAMPLE_3S_320x240_30FPS).trim(0.5,2.5)
    output = str(tmp_path / "output.mp4")
    clip.export(output)
    with VideoReader(output) as reader:
        assert reader.width == 320
        assert reader.height == 240
        assert reader.fps == 30.0
        assert reader.duration_s == pytest.approx(2.0, abs=0.2)
        frames = list(reader.frames())
        # make sure frame count adds up with fps and duration_s
        assert len(frames) == pytest.approx(round(30.0*2.0), abs=1)

def test_clip_export_preserves_audio(tmp_path):
    clip = Clip(SAMPLE_VERT_AUDIO)
    output = str ( tmp_path / "output.mp4" )
    clip.export(output)
    with av.open(output) as container:
        # audio stream exists with valid properties
        assert len(container.streams.audio) > 0
        audio_stream = container.streams.audio[0]
        assert audio_stream.channels > 0
        # audio duration is roughly expected 
        assert audio_stream.time_base is not None and audio_stream.duration is not None
        audio_dur = float(audio_stream.duration * audio_stream.time_base)
        assert audio_dur == pytest.approx(4.131678, abs=0.3)

@pytest.mark.parametrize("file_path,width,height,fps,dur",VIDEO_FIXTURES)
def test_clip_copy(file_path: str, width: int, height: int, fps: float, dur: float):
    clip1 = Clip(file_path)
    clip2 = clip1._copy()
    clip3 = clip2._copy(start=1.0) # test overrides
    assert clip1 is not clip2 and clip1 is not clip3 and clip2 is not clip3
    assert clip1.width == clip2.width == clip3.width
    assert clip1.height == clip2.height == clip3.height
    assert clip1.fps == clip2.fps == clip3.fps
    assert clip1.duration == clip2.duration == clip3.duration # ALL SAME DURATION, because _copy ONLY CHANGES START
    assert clip1.start == clip2.start == 0.0 == clip3.start-1
    assert clip1.layers == clip2.layers == clip3.layers
    assert clip1.effects == clip2.effects == clip3.effects 
    assert clip1.has_audio == clip2.has_audio == clip3.has_audio

@pytest.mark.parametrize("file_path,width,height,fps,dur", VIDEO_FIXTURES)
def test_clip_metadata(file_path: str, width: int, height: int, fps: float, dur: float):
    clip = Clip(file_path)
    meta = clip.metadata()
    # all keys present
    for key in ["duration", "fps", "width", "height", "start", "end", "layer_count", "effect_count"]:
        assert key in meta
    # values correct
    assert meta["fps"] == fps
    assert meta["width"] == width
    assert meta["height"] == height
    assert meta["duration"] == pytest.approx(dur, abs=0.1)
    assert meta["start"] == 0.0
    assert meta["end"] is None
    assert meta["layer_count"] == 0
    assert meta["effect_count"] == 0


# --------------------------------------- SYNTHETIC CLIP TESTS ---------------------------------------------------------------

def build_synthetic_clip(width, height, fps, duration, background, audio_source, start, end) -> tuple[SyntheticClip, float]:
    syn = SyntheticClip(width, height, fps, duration, background, audio_source)
    effective_duration = duration
    if start is None:
        start = 0.0
    if start is not None or end is not None:
        syn = syn.trim(start, end)
        if end is not None:
            effective_duration = duration - (start) - (duration - end)
        else:
            effective_duration = duration - (start)
    return syn, effective_duration

def test_synthetic_clip_default_init():
    syn = SyntheticClip()
    assert syn.width == 1920
    assert syn.height == 1080
    assert syn.fps == 30.0
    assert syn.duration == 10.0
    assert syn.background == (0,0,0)
    assert syn.audio_source is None
    assert syn.start == 0.0
    assert syn.end is None
    assert syn.layers == []
    assert syn.effects == []

@pytest.mark.parametrize("width, height, fps, duration, background, audio_source, start, end", SYNTHETIC_FIXTURES)
def test_synthetic_clip_init(width, height, fps, duration, background, audio_source, start, end):
    syn, _ = build_synthetic_clip(width, height, fps, duration, background, audio_source, start, end)
    assert syn.width == width
    assert syn.height == height
    assert syn.fps == fps
    assert syn.duration == duration
    assert syn.background == background
    assert syn.audio_source == audio_source
    assert syn.start == (0.0 if start is None else start)
    assert syn.end == end
    assert syn.layers == []
    assert syn.effects == []

@pytest.mark.parametrize("width, height, fps, duration, background", INVALID_CLIP_CONFIGS)
def test_synthetic_clip_invalid_inputs(width, height, fps, duration, background):
    with pytest.raises(ValueError):
        syn = SyntheticClip(width, height, fps, duration, background)

@pytest.mark.parametrize("width, height, fps, duration, background, audio_source, start, end", SYNTHETIC_FIXTURES)
def test_synthetic_clip_frames_properties(width, height, fps, duration, background, audio_source, start, end):
    syn, effective_duration = build_synthetic_clip(width, height, fps, duration, background, audio_source, start, end)
    # (1) open generator
    all_frames = list(syn.frames())
    # (2) test shape (check 1st frame, pos 1) since frames are (arr,t)
    assert all_frames[0][0].shape == (height, width, 3)
    assert all_frames[0][0].dtype == np.uint8
    assert len(all_frames) == pytest.approx(int(fps*effective_duration), abs=1) # correct frame count within 1 frame
    # (3) test background color (all pixels in frame belong to background color & entire canvas is filled)
    # all <-> all frames (i) and np.all <-> all pixels in arr all_frames[?][0]
    assert all(np.all(all_frames[i][0] == background) for i in range(len(all_frames)))
    # (4) test timestamps
    timestamps = list(t for (arr,t) in all_frames)
    assert_timestamps_valid(timestamps, effective_duration)
    # (5) test independence btwn frames
    # modify the first frame's array
    all_frames[0][0][0,0] = [min(background[0]+2, 255),int(background[1]/2 + 1),min(background[2]+10,255)]
    assert all_frames[1][0][0,0].tolist() != all_frames[0][0][0,0].tolist() # modifying one frame shouldn't affect the others

@pytest.mark.parametrize("width, height, fps, duration, background, audio_source, start, end", SYNTHETIC_FIXTURES)
def test_synthetic_clip_export(tmp_path, width, height, fps, duration, background, audio_source, start, end):
    syn, effective_duration = build_synthetic_clip(width, height, fps, duration, background, audio_source, start, end)
    output = str(tmp_path / "output.mp4")
    syn.export(output)
    # read what we've exported
    with VideoReader(output) as reader:
        assert reader.width == width
        assert reader.height == height
        assert reader.fps == fps
        assert reader.duration_s == pytest.approx(effective_duration, abs=0.2)
        frames = list(reader.frames())
        # make sure frame count adds up with fps and duration_s within 1frame
        assert len(frames) == pytest.approx(round(fps*effective_duration), abs=1)
        # check background color, read color in frames, make sure its same everywhere
        assert all(
            np.all(
                np.abs(frames[i][0].astype(int) - np.array(background)) <= 10 # tolerance rather than strict equality due to differences in H.264 vs YUV420p conversions
            ) 
            for i in range(len(frames))
        )
    # check audio
    if audio_source is None:
        with av.open(output) as container:
            # audio stream should not exist
            assert len(container.streams.audio) == 0
    if audio_source is not None:
        with av.open(output) as container:
            # audio stream exists with valid properties
            assert len(container.streams.audio) > 0
            audio_stream = container.streams.audio[0]
            assert audio_stream.channels > 0
            # audio duration is roughly expected 
            assert audio_stream.time_base is not None and audio_stream.duration is not None
            audio_dur = float(audio_stream.duration * audio_stream.time_base)
            assert audio_dur == pytest.approx(effective_duration, abs=0.3) # default behavior makes audio clip at video length ('trim' mode)

@pytest.mark.parametrize("width, height, fps, duration, background, audio_source, start, end", SYNTHETIC_FIXTURES)
def test_synthetic_clip_copy(width, height, fps, duration, background, audio_source, start, end):
    clip1, _ = build_synthetic_clip(width, height, fps, duration, background, audio_source, start, end)
    clip2 = clip1._copy()
    clip3 = clip2._copy(start=1.0) # test overrides
    assert clip1 is not clip2 and clip1 is not clip3 and clip2 is not clip3
    assert clip1.width == clip2.width == clip3.width
    assert clip1.height == clip2.height == clip3.height
    assert clip1.fps == clip2.fps == clip3.fps
    assert clip1.duration == clip2.duration == clip3.duration # ALL SAME DURATION, because _copy ONLY CHANGES START
    assert clip1.start == clip2.start 
    assert clip3.start == 1.0
    assert clip1.layers == clip2.layers == clip3.layers
    assert clip1.effects == clip2.effects == clip3.effects 
    assert clip1.audio_source == clip2.audio_source == clip3.audio_source
    assert clip1.background == clip2.background == clip3.background

@pytest.mark.parametrize("width, height, fps, duration, background, audio_source, start, end", SYNTHETIC_FIXTURES)
def test_synthetic_clip_metadata(width, height, fps, duration, background, audio_source, start, end):
    syn, _ = build_synthetic_clip(width, height, fps, duration, background, audio_source, start, end)
    meta = syn.metadata()
    # all keys present
    for key in ["background", "audio_source", "duration", "fps", "width", "height", "start", "end", "layer_count", "effect_count"]:
        assert key in meta
    # values correct
    assert meta["fps"] == fps
    assert meta["width"] == width
    assert meta["height"] == height
    assert meta["duration"] == pytest.approx(duration, abs=0.2)
    assert meta["start"] == (0.0 if start is None else start)
    assert meta["end"] == end
    assert meta["layer_count"] == 0
    assert meta["effect_count"] == 0

# --------------------------------------- AUDIO TESTS (BOTH CLIP TYPES) ---------------------------------------------------------------

@pytest.mark.parametrize("clip_type", ["clip", "synthetic"])
@pytest.mark.parametrize("audio_source, audio_dur", AUDIO_FIXTURES)
def test_clips_mute(tmp_path, clip_type, audio_source, audio_dur):
    if audio_source is None and clip_type == "synthetic":
        pytest.skip("no audio source to mute")
    
    if clip_type == "synthetic":
        clip = make_synthetic_from_fixture(320, 240, 30.0, 2.0, audio_source=audio_source, mute=True)
    elif clip_type == "clip":
        clip = make_video_from_fixture(SAMPLE_VERT_AUDIO, audio_source=audio_source, mute=True)
    
    output = str(tmp_path / "output.mp4")
    clip.export(output)
    with av.open(output) as container:
        assert len(container.streams.audio) == 0 # no audio

@pytest.mark.parametrize("clip_type", ["clip", "synthetic"])
@pytest.mark.parametrize("audio_source, audio_dur", AUDIO_FIXTURES)
@pytest.mark.parametrize("audio_mode, expect_trimmed_audio", [
    ("trim", True),
    ("full", False),
    #("extend", False) TODO
])
def test_all_config_audio_modes(tmp_path, clip_type, audio_source, audio_dur, audio_mode, expect_trimmed_audio):
    if audio_source is None and clip_type == "synthetic":
        pytest.skip("this test case doesn't have audio: skipping audio tests")
    if clip_type == "synthetic":
        clip = make_synthetic_from_fixture(320, 240, 30.0, 2.0, audio_source=audio_source)
    elif clip_type == "clip":
        clip = make_video_from_fixture(SAMPLE_VERT_AUDIO, audio_source=audio_source)
    output = str(tmp_path / "output.mp4")
    clip.export(output, audio_mode=audio_mode)

    with av.open(output) as container:
        assert len(container.streams.audio) > 0
        audio_stream = container.streams.audio[0]
        assert audio_stream.duration is not None and audio_stream.time_base is not None
        export_audio_dur = float(audio_stream.duration * audio_stream.time_base)
        
        uses_own_audio = (audio_source is None and clip_type == "clip")
        if expect_trimmed_audio or uses_own_audio: 
            assert container.streams.video[0] is not None and container.streams.video[0].duration is not None and container.streams.video[0].time_base is not None
            # audio should match video duration, not full source OR audio is video (audio_source == None)
            video_dur = float(container.streams.video[0].duration * container.streams.video[0].time_base)
            assert export_audio_dur == pytest.approx(video_dur, abs=0.3)
        else:
            # audio should be the full audio source duration, completely external to clip
            assert export_audio_dur == pytest.approx(audio_dur, abs=0.3)