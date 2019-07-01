from wow_fishing_buddy import compute_diff_from_images
import cv2

def test_compute_diff_from_images():
    img1 = cv2.imread("text_fixtures/image1.png")
    img2 = cv2.imread("text_fixtures/image2.png")
    cv2.imshow("img1", img1)
    cv2.waitKey(0)

test_compute_diff_from_images()
