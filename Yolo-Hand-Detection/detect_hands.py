import argparse
import glob
import os
from pdb import set_trace

import cv2
from yolo import YOLO

def detect_hands(mat, showim=False):

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images', default="images", help='Path to images or image file')
    ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
    ap.add_argument('-d', '--device', default=0, help='Device to use')
    ap.add_argument('-s', '--size', default=640, help='Size for yolo')
    ap.add_argument('-c', '--confidence', default=0.25, help='Confidence for yolo')
    args = ap.parse_args()

    if args.network == "normal":
        #print("loading yolo...")
        yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
    elif args.network == "prn":
        #print("loading yolo-tiny-prn...")
        yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
    else:
        #print("loading yolo-tiny...")
        yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

    yolo.size = int(args.size)
    yolo.confidence = float(args.confidence)

    conf_sum = 0
    detection_count = 0

    width, height, inference_time, results = yolo.inference(mat)

    return_list = []

    if showim:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 848, 640)

    for detection in results:
        id, name, confidence, x, y, w, h = detection
        return_list.append([x, y, w, h, confidence])

        if showim:
            # draw a bounding box rectangle and label on the image
            color = (255, 0, 255)
            cv2.rectangle(mat, (x, y), (x + w, y + h), color, 1)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

            cv2.imwrite("export.jpg", mat)

            # show the output image
            cv2.imshow('image', mat)
            #cv2.imwrite('images/test.png', mat)

            cv2.waitKey(0)

    #print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))
    #cv2.destroyAllWindows()
    
    return return_list
    
if __name__ == "__main__":
    mat = cv2.imread('test_images/test5.png')
    detect_hands(mat, showim=True)
