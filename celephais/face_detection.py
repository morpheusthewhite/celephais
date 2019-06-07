import cv2 as cv
import os
import pkg_resources

CLASSIFIERS_FOLDER = "classifiers"

CLASSIFIER_FILENAME_FRONTAL = pkg_resources.resource_filename(__name__,
                                                              os.path.join(CLASSIFIERS_FOLDER +
                                                                           '/haarcascade_frontalface_default.xml'))
CLASSIFIER_FILENAME_PROFILE = pkg_resources.resource_filename(__name__,
                                                              os.path.join(CLASSIFIERS_FOLDER +
                                                                           '/haarcascade_profileface.xml'))

OVERLAPPING_THRESHOLD = 0.9
IMAGE_PREFIX = "detected"

face_cascade_frontal = cv.CascadeClassifier(os.path.join(CLASSIFIERS_FOLDER, CLASSIFIER_FILENAME_FRONTAL))
face_cascade_profile = cv.CascadeClassifier(os.path.join(CLASSIFIERS_FOLDER, CLASSIFIER_FILENAME_PROFILE))


def face_detection_draw_rectangles(img_raw):
    # convert image to gray scale in order to detect faces
    gray = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)

    faces = face_detection_get_faces(gray)

    # draw non overlapping faces
    for (x, y, w, h) in faces:
        cv.rectangle(img_raw, (x, y), (x + w, y + h), (255, 0, 255), 4)


def get_overlapping_area(x1, y1, w1, h1, x2, y2, w2, h2):
    """
    Calculates the overlapping area given the two rectangles respectively defined by (x1, y1, w1, h1)
    and (x2, y2, w2, h2)
    """
    if x1 < x2 and x1 + w1 < x2:
        return 0
    elif x2 <= x1 and x2 + w2 < x1:
        return 0

    if y1 < y2 and y1 + h1 < y2:
        return 0
    elif y2 <= y1 and y2 + h2 < y1:
        return 0

    overlapping_height = min(x1 + w1, x2 + w2) - max(x1, x2)
    overlapping_width = min(y1 + h1, y2 + h2) - max(y1, y2)

    return overlapping_height * overlapping_width


def face_detection_get_faces(img_gray):
    """
    returns a list of rectangles detected in the given image
    :param img_gray: the image (in gray scale)
    :return: a list of rectangles in the form (x, y, w, h)
    """
    # detect faces
    faces_profile = face_cascade_profile.detectMultiScale(img_gray, 1.3, 5)
    faces_frontal = face_cascade_frontal.detectMultiScale(img_gray, 1.3, 5)

    # merge faces detected
    faces = list(faces_profile) + list(faces_frontal)

    valid_faces = []
    # filter overlapping faces
    for i in range(len(faces)):
        (x1, y1, w1, h1) = faces[i]
        other_faces = faces[i + 1:]

        valid_faces.append(faces[i])

        for (x2, y2, w2, h2) in other_faces:
            overlapping_area = get_overlapping_area(x1, y1, w1, h1, x2, y2, w2, h2)

            if overlapping_area / w1 * h1 > OVERLAPPING_THRESHOLD:
                valid_faces.pop()
                break

    return valid_faces


def face_detection_count(img_raw):
    # convert image to gray scale in order to detect faces
    gray = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)

    return len(face_detection_get_faces(gray))


def detect_show_image(img):
    face_detection_draw_rectangles(img)
    cv.imshow('sample image', img)
    cv.waitKey(0)


def detect_save_image(img, img_folder, img_name):
    folder_path = os.path.join(os.getcwd(), img_folder)

    if not os.path.exists(folder_path):
        print(folder_path + " does not exists, image cannot be saved")

    # save the image in the given folder
    img_path = os.path.join(os.getcwd(), img_folder, img_name)

    face_detection_draw_rectangles(img)
    (dir, filename) = os.path.split(img_path)
    new_filename = os.path.join(dir, IMAGE_PREFIX + filename)
    cv.imwrite(new_filename, img)
