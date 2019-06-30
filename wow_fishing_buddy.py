import logging
import pyautogui
from skimage.measure import compare_ssim
import numpy as np
import pyautogui
import cv2
import time
from datetime import datetime

import pyaudio
import math
import audioop
from collections import deque


LOGFMT = ("%(asctime)-15s [%(levelname)-5s] %(filename)-10s:%(lineno)-3d "
            "%(message)s")
logging.basicConfig(level=logging.DEBUG, format=LOGFMT)

def get_focus_region():
    UPPER_CUT = 0.2
    LOWER_CUT = 0.75
    SCREEN_SIZE = (1280, 768)
    top_left_x = int(SCREEN_SIZE[0] * UPPER_CUT)
    top_left_y = int(SCREEN_SIZE[1] * UPPER_CUT)
    bot_right_x = int(SCREEN_SIZE[0] * LOWER_CUT)
    bot_right_y = int(SCREEN_SIZE[1] * LOWER_CUT)
    return top_left_x, top_left_y, bot_right_x, bot_right_y

def get_screenshot(region):
    img = pyautogui.screenshot(region=region)
    return img

def compute_diff_from_images(img1, img2):
    diff = img2 - img1
    return diff

def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def listen():
	logging.debug('Well, now we are listening for loud sounds...')
	CHUNK = 1024  # CHUNKS of bytes to read each time from mic
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 18000
	THRESHOLD = 1600  # The threshold intensity that defines silence
	                  # and noise signal (an int. lower than THRESHOLD is silence).
	SILENCE_LIMIT = 1  # Silence limit in seconds. The max ammount of seconds where
	                   # only silence is recorded. When this time passes the
	                   # recording finishes and the file is delivered.
	#Open stream
	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                frames_per_buffer=CHUNK)
	cur_data = ''  # current chunk  of audio data
	rel = int(RATE/CHUNK)
	slid_win = deque(maxlen=SILENCE_LIMIT * rel)


	success = False
	listening_start_time = time.time()
	while True:
		try:
			cur_data = stream.read(CHUNK)
			slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
			sound = [x > THRESHOLD for x in slid_win]
			if max(slid_win) > 1500:
			   logging.debug("sound level  [{}]".format(max(slid_win)))
			if(sum(sound) > 0):
				logging.debug('I heard something!')
				success = True
				break
			if time.time() - listening_start_time > 20:
				logging.debug('I dont hear anything already 20 seconds!')
				break
		except IOError:
			break

	# print "* Done recording: " + str(time.time() - start)
	stream.close()
	p.terminate()
	return success


def main():
    logging.info("Waiting 2 Sec before switching to WOW")
    time.sleep(2)

    region = get_focus_region()
    logging.info("Focus region : {}".format(region))

    logging.info("Taking inital screenshot")
    before = pyautogui.screenshot(region=region)

    logging.info("Sending a float")
    pyautogui.press("1")

    logging.info("Wait for 2 sec before capture new snapshot")
    time.sleep(2)
    after = pyautogui.screenshot(region=region)

    logging.info("running comparison")
    img1 = np.array(before)
    img2 = np.array(after)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255,
    	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    biggest_cnt = contours[0]
    biggest_cnt_shifted = biggest_cnt + (region[0], region[1])

    # find center of mass
    M = cv2.moments(biggest_cnt_shifted)
    cur_x = int(M["m10"] / M["m00"])
    cur_y = int(M["m01"] / M["m00"])
    logging.info("Float center at: {}, {}".format(cur_x, cur_y))

    pyautogui.moveTo(cur_x, cur_y, 0.5)

    # log the screen image
    fullscrn = pyautogui.screenshot()
    fullscrn_img = np.array(fullscrn)
    cv2.rectangle(fullscrn_img,
                  (region[0], region[1]),
                  (region[2], region[3]),
                  (0, 255, 255))

    pyautogui.moveTo(cur_x, cur_y, 0.5)

    cv2.drawContours(fullscrn_img, [biggest_cnt_shifted], 0, (0,0,255), 0)
    cv2.circle(fullscrn_img, (cur_x, cur_y), 7, (255, 255, 255), -1)
    cv2.imwrite("debug/{}.png".format(now_str()), fullscrn_img)

    window_name = 'fishing'
    window_width = 1920 - 1280
    window_height = int(1280 * (1920 -1280)/1920)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 1280, 0)
    cv2.resizeWindow(window_name, window_width, window_height)
    fullscrn_img_small = cv2.resize(fullscrn_img, (window_width, window_height))
    cv2.imshow(window_name, fullscrn_img_small)
    cv2.waitKey(1000)

    if listen():
        logging.debug('now click the float')
        pyautogui.click()
        logging.debug('sleep for 2 sec to reset the pic')
        time.sleep(2)
    else:
        logging.debug('trying again!')


if __name__ == '__main__':
    for i in range(500):
        logging.debug("trial: # {}".format(i))
        main()
