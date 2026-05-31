import numpy as np

def test_import():
    import reelpy
    assert reelpy is not None

def test_numpy_frame_operations():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (128, 64, 32)
    assert frame[0, 0, 0] == 128
    assert frame.shape == (100, 100, 3)