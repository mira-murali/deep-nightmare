import cv2
import argparse

# e.g. python GUI.py --p1 image1.png --p2 /dir/image2.jpg
parser = argparse.ArgumentParser()
parser.add_argument('--p1', required=True, type=str, help='Specify path to the first image')
parser.add_argument('--p2', required=True, type=str, help='Specify path to the second image')
args = parser.parse_args()

def blend(event, x, y, flags, param): 
    overlay_height, overlay_width, channels = overlay.shape
    alpha =0.9 # decide which image is more dominant
    step = 20 # decid how large each paint brush is

    if event == cv2.EVENT_MOUSEMOVE:
        x1, y1 = max(0, x-step), max(0, y-step)
        x2, y2 = min(x+step, overlay_height), min(y+step, overlay_width)
        
        # make composite
        composite = cv2.addWeighted(bg[y1:y2, x1:x2],
                                 alpha,
                                 overlay[y1:y2, x1:x2],
                                 1 - alpha,
                                 0)
        bg[y1:y2, x1:x2] = composite

# read images
p1, p2 = cv2.imread(args.p1, 1), cv2.imread(args.p2, 1)
bg, overlay = p1.copy(), p2.copy()

while True:
    cv2.namedWindow("Deep-Nightmare")
    cv2.setMouseCallback("Deep-Nightmare", blend) # param = None
    cv2.imshow("Deep-Nightmare", bg)
    # press key q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()