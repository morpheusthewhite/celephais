import xml.etree.ElementTree as ET

def xml_parse(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    parsed_dict = {}
    for child in root:
        parsed_dict[child.tag] = child.text

    return parsed_dict
