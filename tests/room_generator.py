import json

N_ROOMS = 10
ROOM_NAMES = ["r" + str(i).zfill(2) for i in range(N_ROOMS)]
ROOM_CAPS = [40 + 10 * i for i in range(N_ROOMS)]
OUT_FILE = "rooms.json"

rooms = []

for (name, cap) in zip(ROOM_NAMES, ROOM_CAPS):
    room = {"name": name, "cap":cap}
    rooms.append(room)

with open(OUT_FILE, "w") as f:
    f.write(json.dumps(rooms))
