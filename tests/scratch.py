
from reelpy.clip.video import Clip
from reelpy.effects.fades import FadeInEffect, FadeOutEffect
clip = Clip("tests/fixtures/Sample_320x240_30fps.mp4")
clip = clip.apply(FadeInEffect(1.0))
clip = clip.apply(FadeOutEffect(1.0))
clip.preview()


"""
import cv2
print("cv2 imported first")
import av
print("av imported second")

print("about to namedWindow")
cv2.namedWindow("test", cv2.WINDOW_NORMAL)
print("namedWindow returned")
frame = np.zeros((300, 300, 3), dtype=np.uint8) if False else __import__("numpy").zeros((300,300,3), dtype="uint8")
frame[:] = (0, 0, 255)
cv2.imshow("test", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("done")
"""
