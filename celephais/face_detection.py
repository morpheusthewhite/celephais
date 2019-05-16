import cv2 as cv
import os
import pkg_resources

CLASSIFIERS_FOLDER = "classifiers"
CLASSIFIER_FILENAME_FRONTAL = pkg_resources.resource_filename(__name__,
                                                            os.path.join(CLASSIFIERS_FOLDER + '/haarcascade_frontalface_default.xml'))
CLASSIFIER_FILENAME_PROFILE = pkg_resources.resource_filename(__name__,
                                                            os.path.join(CLASSIFIERS_FOLDER + '/haarcascade_profileface.xml'))
print(CLASSIFIER_FILENAME_PROFILE)

face_cascade_frontal = cv.CascadeClassifier(os.path.join(CLASSIFIERS_FOLDER, CLASSIFIER_FILENAME_FRONTAL))
face_cascade_profile= cv.CascadeClassifier(os.path.join(CLASSIFIERS_FOLDER, CLASSIFIER_FILENAME_PROFILE))


def face_detection_draw_rectangles(img_raw):
    # convert image to gray scale in order to detect faces
    gray = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)

    # detect profile faces and draw them on the image
    faces = face_cascade_profile.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(img_raw, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # detect frontal faces and draw them on the image
    faces = face_cascade_frontal.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(img_raw, (x, y), (x + w, y + h), (255, 0, 0), 2)
