from image_utils import diff_images, get_contours
import cv2
import numpy as np

def test_diff_images():
    img1 = cv2.imread('test_fixtures/background00.jpg')
    img2 = cv2.imread('test_fixtures/float00.jpg')
    result = diff_images(img1, img2)
    #cv2.imwrite('test_fixtures/test_diff_images_result.jpg', result)
    expected = cv2.imread('test_fixtures/test_diff_images_result.jpg')
    np.testing.assert_array_equal(result, expected)

def test_get_contours(snapshot):
    img = cv2.imread('test_fixtures/float00.jpg')
    contour = get_contours(img)
    snapshot.assert_match(contour)

if __name__ == '__main__':
    test_diff_images()



