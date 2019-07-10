import cv2
import numpy as np
from skimage.measure import compare_ssim
from matplotlib import pyplot as plt

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
    WHITE = (255, 255, 255)
    mask = np.zeros(img.shape,dtype='uint8')
    mask = cv2.drawContours(mask, [contour], -1, WHITE, thickness=cv2.FILLED)
    img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def get_descriptor_from_img(img):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def feature_matching(img, kp, des):
    img_kp, img_des = get_descriptor_from_img(img)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des, img_des)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def get_angle_of_rotation(img, tmpl):
    pass

if __name__ == '__main__':
    background = cv2.imread('test_fixtures/background00.jpg')
    scene = cv2.imread('test_fixtures/float00.jpg')
    diff = diff_images(background, scene)
    contour = get_contours(diff)
    obj = crop_required_obj(scene, contour)

    obj_kp, obj_des = get_descriptor_from_img(obj)
    img_kp, img_des = get_descriptor_from_img(scene)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(obj_des, img_des)
    print(matches)
    img3 = background.copy()
    cv2.drawMatches(obj, obj_kp, scene, img_kp, matches[:10], flags=2, outImg=img3)
    plt.imshow(img3),plt.show()
