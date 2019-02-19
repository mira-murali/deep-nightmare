import cv2
import argparse, sys, time
from speed_dream import dream
from model import Model
import torch
import hyperparameters as hyp
from PIL import Image
import numpy as np

def cropToFit(param):
    overlay_height, overlay_width, channels = overlay.shape
    bg_height, bg_width, bg_channels = bg.shape
    h = min(overlay_height, bg_height)
    w = min(overlay_width, bg_width)
    param[0] = bg[0:h,0:w]
    param[1] = overlay[0:h,0:w]
    param[2] = original[0:h,0:w]

def blend(event, x, y, flags, param): 
    try:
        bg = param[0]
        model = param[3]
        step = 112 # decid how large each paint brush is

        if event == cv2.EVENT_MOUSEMOVE:
            bg_height, bg_width, channels = bg.shape
            x1, y1 = max(0, x-step), max(0, y-step)
            x2, y2 = min(x+step, bg_height), min(y+step, bg_width)

            # assume returned "overlay" image has the same size of the img passed in
            start = time.time()
            img = np.asarray(Image.fromarray(cv2.cvtColor(bg[y1:y2, x1:x2],cv2.COLOR_BGR2RGB)), np.uint8)
            param[1] = dream(model, img)
            param[1] = cv2.cvtColor(param[1],cv2.COLOR_RGB2BGR)
            end = time.time()
        #    print("{:.1f} FPS".format(1/(end-start)))
            overlay = param[1]
            # make composite
            #cv2.imwrite("{}.jpg".format(time.time()),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
            composite = cv2.addWeighted(bg[y1:y2, x1:x2],
                                            alpha1,
                                            overlay,
                                            1-alpha1,
                                            0)

            bg[y1:y2, x1:x2] = composite
            param[0] = bg
    except Exception as e:
        pass

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
parser.add_argument('-m', default='pre', type=str, help='Specify path to the model')
args = parser.parse_args()

p = cv2.imread(args.p, 1) # read images
model = args.m
bg, original = p.copy(), p.copy()
alpha1 =0.9 # for blend
alpha2 = 0.96 # for fadeOut
# cropToFit(param) - not necessary
mdl = Model()
if model != "pre":
    mdl.load_state_dict(torch.load(model))
model = mdl.cuda()
param = [bg, None, original, model]
window_name = "Deep-Nightmare"

while True:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, blend, param)
    cv2.imshow(window_name, param[0]) # display bg
#    fadeOut(param)
    cv2.waitKey(1)
cv2.destroyAllWindows()