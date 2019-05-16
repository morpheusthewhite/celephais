import cv2 as cv
import os
import argparse

from celephais import face_detection

# creating top-level parser
parser = argparse.ArgumentParser(prog="Celephais")
subparsers = parser.add_subparsers(help="sub-command help")

# creating parser for the detection command
parser_detection = subparsers.add_parser('detect')
parser_detection.add_argument("--image", required=True, type=str, help="the image file in which to detect faces")
parser_detection.add_argument("--show", action="store_true", help="show the detected faces in a window, "
                                                                  "otherwise just prints the number")

def main():
    parsed_args = parser.parse_args()

    img_path = os.path.join(os.getcwd(), parsed_args.image)

    img = cv.imread(img_path)

    if parsed_args.show:
        face_detection.face_detection_draw_rectangles(img)
        cv.imshow('sample image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("detected " + str(face_detection.face_detection_count(img)) +" faces")


if __name__ == '__main__':
    main()