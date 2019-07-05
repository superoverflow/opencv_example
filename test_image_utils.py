from image_utils import diff_images, get_contours
import cv2
import numpy as np
from snapshottest.file import FileSnapshot

def test_diff_images(snapshot, tmpdir):
    img1 = cv2.imread('test_fixtures/background00.jpg')
    img2 = cv2.imread('test_fixtures/float00.jpg')
    result = diff_images(img1, img2)
    tmpfile = tmpdir.join('test_diff_images_result.bmp')
    cv2.imwrite(str(tmpfile), result)
    snapshot.assert_match(FileSnapshot(str(tmpfile)))

# def test_get_contours():
#     img = cv2.imread('test_fixtures/test_diff_images_result.bmp',
#                     cv2.IMREAD_GRAYSCALE)
#     bg = cv2.imread('test_fixtures/float00.jpg')
#     contour = get_contours(img)
#     cv2.drawContours(bg, [contour], 0, (0, 255, 255))
#     cv2.imshow("contour-with-bg", bg)
#     cv2.waitKey(0)



