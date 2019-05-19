import xml.etree.ElementTree as ET


def xml_parse(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    parsed_dict = {}
    for child in root:
        parsed_dict[child.tag] = child.text

    return parsed_dict


def group_by_time(events_list):
    """
    :param events_list: a list of dictionaries in the form {"day":..., "hour":..., ...}
    :return: a dictionary which contains as key the different time and as values a list of events, i.e.
    {"day hour":[event_1, event_2 ...], ...} where event_i are dictionary obtained removing the "day"
    and "hour" pair from events_list
    """
    grouped_events = {}

    for event in events_list:
        event = event.copy()  # to preserve original dictionary

        time = str(event["day"]) + " " + str(event["hour"])

        event.pop("day")
        event.pop("hour")
        try:
            grouped_events[time].append(event)
        except KeyError:
            grouped_events[time] = [event]

    return grouped_events
