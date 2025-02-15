import json

# Open the JSON file and load it as a dictionary
with open(r"old_script\pksp\fusiondex_links2.json", "r") as file:
    data = json.load(file)

# Swap keys and values
swapped_data = {value: key for key, value in data.items()}

# Save the swapped dictionary to a new JSON file
with open(r"old_script\pksp\swapped_fusiondex_links.json", "w") as file:
    json.dump(swapped_data, file, indent=4)
