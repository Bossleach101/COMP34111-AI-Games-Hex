import json
import numpy as np

with open('processed_11x11_games.json', 'r') as f:
    data = json.load(f)

print("First posistion loaded from JSON:")
np_array = np.array(data[0]['position'])
print(np_array)

print("Second posistion loaded from JSON:")
np_array = np.array(data[1]['position'])
print(np_array)

