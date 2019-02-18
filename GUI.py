import cv2
import argparse

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
    overlay = param[1]
    overlay_height, overlay_width, channels = overlay.shape
    step = 20 # decid how large each paint brush is

    if event == cv2.EVENT_MOUSEMOVE:
        x1, y1 = max(0, x-step), max(0, y-step)
        x2, y2 = min(x+step, overlay_height), min(y+step, overlay_width)
        
        # make composite
        composite = cv2.addWeighted(bg[y1:y2, x1:x2],
                                    alpha1,
                                    overlay[y1:y2, x1:x2],
                                    1 - alpha1,
                                    0)
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

# e.g. python GUI.py --p1 image1.png --p2 /dir/image2.jpg
parser = argparse.ArgumentParser()
parser.add_argument('--p1', required=True, type=str, help='Specify path to the base image')
parser.add_argument('--p2', required=True, type=str, help='Specify path to the second image')
# parser.add_argument('-m', required=True, type=str, help='Specify path to the model')
args = parser.parse_args()

# read images
p1, p2 = cv2.imread(args.p1, 1), cv2.imread(args.p2, 1)
original, bg, overlay = p1.copy(), p1.copy(), p2.copy()
param = [bg, overlay, original]
alpha1 =0.9 # for blend
alpha2 = 0.96 # for fadeOut
cropToFit(param)

while True:
    cv2.namedWindow("Deep-Nightmare")
    cv2.setMouseCallback("Deep-Nightmare", blend, param)
    cv2.imshow("Deep-Nightmare", param[2])
    # press key q to exit
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break
    fadeOut(param)
    cv2.waitKey(1)
cv2.destroyAllWindows()