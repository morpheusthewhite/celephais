import cv2 as cv
import os
import argparse
import json

from celephais import face_detection
from celephais import metadata
from celephais import data_parse
from celephais import train

# creating top-level parser
parser = argparse.ArgumentParser(prog="Celephais")

# creating parser for the detection command
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument("--image", type=str, help="the image file in which to detect faces")
source_group.add_argument("--xml", type=str, help="the xml file which contains required metadata")
source_group.add_argument("--json", type=str, help="the json file which contains dataset for training (--ojson and "
                                                   "--show will have no effect)")

out_group = parser.add_mutually_exclusive_group(required=False)
out_group.add_argument("--show", action="store_true", help="show the detected faces in a window, "
                                                                  "otherwise just prints the number")
out_group.add_argument("--ojson", help="if specified, and if --xml is given, metadata are saved "
                                              "with the given filename")
train_group = parser.add_mutually_exclusive_group(required=True)
train_group.add_argument("--no-train", help="will exit before training the model", action="store_true")
train_group.add_argument("--predict-xml", help="predict parsing the given xml data (folder or file)")


def main():
    parsed_args = parser.parse_args()

    img_paths = []
    if parsed_args.image is not None:
        img_paths = data_parse.parse_images(parsed_args.image)
    elif parsed_args.xml is not None:  # xml is given instead
        xml_paths = data_parse.parse_xmls(parsed_args.xml)

        # for each xml the path to the associated image is calculated
        dicts_parsed = []
        for xml_path in xml_paths:
            parsed_dict = metadata.xml_parse(xml_path)
            dicts_parsed.append(parsed_dict)

            img_filename = parsed_dict["photo"]
            img_paths.append(os.path.join(os.path.split(xml_path)[0], img_filename))

    dicts_detected = []

    for img_path in img_paths:
        img = cv.imread(img_path)

        if parsed_args.show:
            face_detection.face_detection_draw_rectangles(img)
            cv.imshow('sample image', img)
            cv.waitKey(0)
            continue
        else:  # if show is not specified the number of face detected are printed on stdout
            face_detected = face_detection.face_detection_count(img)
            print("detected " + str(face_detected) +" faces in " + img_path)

        # save the final json file
        if parsed_args.xml is not None:
            parsed_dict = dicts_parsed.pop(0)
            parsed_dict.pop("photo")

            parsed_dict["students"] = face_detected

            dicts_detected.append(parsed_dict)

    if parsed_args.json is None and parsed_args.ojson is not None:
        with open(os.path.join(os.getcwd(), parsed_args.ojson), "w") as f:
            f.write(json.dumps(dicts_detected))

    cv.destroyAllWindows()

    # if an image has been passed then no training is possible
    if parsed_args.image is not None or parsed_args.no_train:
        return
    elif parsed_args.json is not None:
        json_file = os.path.join(os.getcwd(), parsed_args.json)

        with open(json_file, "r") as f:
            dicts_detected = json.load(f)

    model = train.StudentsEstimator(dicts_detected)

    print("Training the model..")
    model.train()
    print("Training completed")

    xml_predict_paths = data_parse.parse_xmls(parsed_args.predict_xml)

    # read xlms which contains data to predict
    dicts_prediction_parsed = []
    for xml_path in xml_predict_paths:
        dict_prediction_parsed = metadata.xml_parse(xml_path)
        dicts_prediction_parsed.append(dict_prediction_parsed)

    # predict and print the result
    print(model.predict(dicts_prediction_parsed))


if __name__ == '__main__':
    main()
