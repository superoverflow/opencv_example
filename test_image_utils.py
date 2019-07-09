from image_utils import diff_images, get_contours, get_centroid
import cv2
import numpy as np
from snapshottest.file import FileSnapshot

RESULT_COLOR = (0, 255, 0)

def test_diff_images(snapshot, tmpdir):
    before = cv2.imread('test_fixtures/background00.jpg')
    after = cv2.imread('test_fixtures/float00.jpg')
    diff = diff_images(before, after)
    contour = get_contours(diff)
    centroid = get_centroid(contour)

    result = after.copy()
    cv2.drawContours(result, [contour], 0, RESULT_COLOR)
    cv2.circle(result, centroid, radius=3, color=RESULT_COLOR, thickness=-1)

    tmpfile = tmpdir.join('test_diff_images_result.bmp')
    cv2.imwrite(str(tmpfile), result)
    snapshot.assert_match(FileSnapshot(str(tmpfile)))
