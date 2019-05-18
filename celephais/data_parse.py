import os


def parse_images(image_path):
    """
    returns a list of the absolute paths to the images (more than 1 if image_path is a folder, 1 if is a file)
    """
    images = []

    # if a directory is passed
    if os.path.isdir(image_path):
        images_listed = os.listdir(image_path)

        # then all the files in the dir will be analyzed
        for img in images_listed:
            images.append(os.path.join(os.getcwd(), image_path, img))
    else:
        # only the given image is analyzed
        images.append(os.path.join(os.getcwd(), image_path))

    return images


def parse_xmls(xml_path):
    """
    returns a list of the absolute paths to the xmls (more than 1 if xml_path is a folder, 1 if is a file)
    """
    xml_paths = []

    xml_path_abs = os.path.join(os.getcwd(), xml_path)

    # if a directory is passed
    if os.path.isdir(xml_path_abs):
        xmls_listed = filter(lambda x: x.endswith(".xml"), os.listdir(xml_path))

        # then all the files in the dir will be analyzed
        for xml in xmls_listed:
            xml_paths.append(os.path.join(xml_path_abs, xml))
    else:
        # only the given xml is analyzed
        xml_paths.append(xml_path_abs)

    return xml_paths
