import xml.etree.ElementTree as ET
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subject", required=True, help="subject name")
parser.add_argument("--hour", required=True, help="The hour (as integer)")
parser.add_argument("-d", "--day", required=True, help="The day (full lowercase name)")
parser.add_argument("-p", "--photo_filename", required=True, help="the filename of the photo associated to this xml")
parser.add_argument("-o", help="the name of the output file", default="data.xml")

parsed_args = parser.parse_args()

subject_name = parsed_args.subject
hour_value = parsed_args.hour
day_value= parsed_args.day
photo_filename = parsed_args.photo_filename
output_filename = parsed_args.o

# create the file structure
data = ET.Element('data')
subject = ET.SubElement(data, 'subject')
hour = ET.SubElement(data, 'hour')
day = ET.SubElement(data, 'day')
photo = ET.SubElement(data, 'photo')

subject.text = subject_name
hour.text = hour_value
day.text = day_value
photo.text = photo_filename

# create a new XML file with the results
data_string = ET.tostring(data)

output_file = open(output_filename, "w")
output_file.write(data_string)
output_file.write(os.linesep)
output_file.close()
