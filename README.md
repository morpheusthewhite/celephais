# Celephais

An application for counting students in photos, predicting their flow
from data and assigning them to a set of classrooms.

## Installation

Just run

```bash
pip install --upgrade .
```

### Dependencies

Celephais depends on `keras`, `scikit-learn`, `cplex`, `pandas` and
`opencv-python` (all packages are automatically installed).

# Usage

Celephais takes its argument from command line, you can list them by
runnning `celephais -h`
```
usage: Celephais [-h] (--image IMAGE | --xml XML | --json JSON)
                 [--show | --ojson OJSON]
                 (--no-train | --predict-xml PREDICT_XML)
                 [--rooms-json ROOMS_JSON] [--print-score]
                 [--save-imgs SAVE_IMGS]

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE         the image file or folder in which to detect faces
  --xml XML             the xml file or folder which contains required
                        metadata
  --json JSON           the json file which contains dataset for training
                        (--ojson and --show will have no effect)
  --show                show the detected faces in a window, otherwise just
                        prints the number
  --ojson OJSON         if specified, and if --xml is given, metadata are
                        saved with the given filename
  --no-train            exit before training the model
  --predict-xml PREDICT_XML
                        the xml data (folder or file) of lessons to predict
  --rooms-json ROOMS_JSON
                        the json containing the rooms in which classes will be
                        allocated
  --print-score         use a part of the dataset to calculate the score of
                        the net used for the prediction
  --save-imgs SAVE_IMGS
                        the folder in which images with detected faces will be
                        saved as 'detected_ORIGINAL_FILENAME' (cannot be used
                        with --ojson)
```

## Required files

If `--xml` is specified, `celephais` will look for xml files in the following form
```xml
<data>
    <subject>Machine Learning</subject>
    <hour>10</hour>
    <day>sunday</day>
    <photo>relative/path/to/image</photo>
</data>
```
while `--json` needs a list of dictionaries, for example

```javascript
[{"day": "tuesday", "subject": "Calculus", "hour": 9, "students": 69} ...
```

`--predict-xml` expects the same of `--xml` apart from the `<photo>` child.

`--rooms-json` needs a list of dictionaries containing the name of the
room and its capacity
```javascript
[{"cap": 40, "name": "r00"} ...
```

### About

This application takes its name after one of Lovecraft's tales,
[Celephais](http://www.hplovecraft.com/writings/texts/fiction/c.aspx)