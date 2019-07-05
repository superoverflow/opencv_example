import cv2
from skimage.measure import compare_ssim

def diff_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(gray1, gray2, full=True)

    if score > 0.99:
        raise Exception("No observable diff found from images")

    diff = (diff * 255).astype("uint8")
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresh

def get_contours(img):
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    largest_contours = contours[0]
    return largest_contours

def get_centroid(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

def crop_required_obj(img, contour):
    pass

def get_descriptor_from_template(tmpl):
    pass

def feature_matching(img, tmpl):
    pass

def get_angle_of_rotation(img, tmpl):
    pass
