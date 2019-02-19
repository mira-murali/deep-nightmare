import cv2
import argparse
from speed_dream import dream

def cropToFit(param):
    overlay_height, overlay_width, channels = overlay.shape
    bg_height, bg_width, bg_channels = bg.shape
    h = min(overlay_height, bg_height)
    w = min(overlay_width, bg_width)
    param[0] = bg[0:h,0:w]
    param[1] = overlay[0:h,0:w]
    param[2] = original[0:h,0:w]

def blend(event, x, y, flags, param): 
    bg = param[0]
    model = param[3]
    step = 20 # decid how large each paint brush is

    if event == cv2.EVENT_MOUSEMOVE:
        bg_height, bg_width, channels = bg.shape
        x1, y1 = max(0, x-step), max(0, y-step)
        x2, y2 = min(x+step, bg_height), min(y+step, bg_width)

        # assume returned "overlay" image has the same size of the img passed in
        param[1] = dream(model, bg[y1:y2, x1:x2])
        overlay = param[1]

        # make composite
        if overlay != None:
            composite = cv2.addWeighted(bg[y1:y2, x1:x2],
                                        alpha1,
                                        overlay,
                                        1 - alpha1,
                                        0)
        else:
            print("Returned image by dream() is none.")

        bg[y1:y2, x1:x2] = composite
        param[0] = bg

def fadeOut(param):
    bg = param[0]
    original = param[2]
    param[0] = cv2.addWeighted(bg,
                               alpha2,
                               original,
                               1 - alpha2,
                               0)

# e.g. python GUI.py -p /path/to/image -m /path/to/model
parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True, type=str, help='Specify path to the base image')
parser.add_argument('-m', required=True, type=str, help='Specify path to the model')
args = parser.parse_args()

p = cv2.imread(args.p, 1) # read images
model = args.m
bg, original = p.copy(), p.copy()
param = [bg, None, original, model]
alpha1 =0.9 # for blend
alpha2 = 0.96 # for fadeOut
# cropToFit(param) - not necessary
while True:
    cv2.namedWindow("Deep-Nightmare")
    cv2.setMouseCallback("Deep-Nightmare", blend, param)
    cv2.imshow("Deep-Nightmare", param[0]) # display bg
    if cv2.waitKey(100) & 0xFF == ord("q"): # press key q to exit
        break
    fadeOut(param)
    cv2.waitKey(1)
cv2.destroyAllWindows()