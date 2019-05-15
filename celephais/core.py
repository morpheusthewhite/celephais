import cv2 as cv
import celephais.common
import celephais.face_detection as face_detection

def main():
    img = cv.imread('../class.jpg')

    face_detection.face_detection_draw_rectangles(img)
    celephais.common.CelephaisParser()
    cv.imshow('sample image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()