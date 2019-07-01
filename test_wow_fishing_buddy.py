from wow_fishing_buddy import compute_diff_from_images
import cv2
from skimage.measure import compare_ssim

def test_compute_diff_from_images():
    img1 = cv2.imread("test_fixtures/image1.png")
    img2 = cv2.imread("test_fixtures/image2.png")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(gray1, gray2, full=True)
    cv2.imshow("diff", diff)
    cv2.waitKey(1000)

test_compute_diff_from_images()
