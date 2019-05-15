import cv2 as cv
import os

from celephais import utils
from celephais import face_detection

def main():
    parser = utils.CelephaisParser()
    parsed_args = parser.parse_args()

    img_path = os.path.join(os.getcwd(), parsed_args.image)

    img = cv.imread(img_path)

    face_detection.face_detection_draw_rectangles(img)
    cv.imshow('sample image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()