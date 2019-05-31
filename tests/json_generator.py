import json
import random

SUBJECTS = ["SC2", "PSW", "LAB", "ALG", "ANA", "STA", "PFP", "SC1"]
DATASET_SIZE = 1000
LESSON_HOURS = 14
LESSON_STARTING_HOUR = 6
OUTPUT_FILENAME = 'input_data.json'
DAYS = ["monday", "sunday", "thursday", "wednesday", "tuesday"]

assert(LESSON_HOURS + LESSON_STARTING_HOUR <= 24)
hour_central = LESSON_STARTING_HOUR + LESSON_HOURS / 2

dataset = []

# calculate randomly a dataset
for i in range(DATASET_SIZE):
    s_index = int(random.random() * len(SUBJECTS))
    hour = int(random.random() * LESSON_HOURS + LESSON_STARTING_HOUR)
    d_index = int(random.random() * len(DAYS))

    # just a fancy math relationship
    students = s_index * 5 - (hour - hour_central)**2 + 100 + d_index % 3 + int(random.random()*16 - 8)

    entry = {"day": DAYS[d_index], "hour": hour, "subject": SUBJECTS[s_index], "students": students}
    dataset.append(entry)

# print(dataset)

# write the dataset to the output file with the given filename
with open(OUTPUT_FILENAME, "w") as f:
    f.write(json.dumps(dataset))
