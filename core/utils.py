import json
import os
import re


def read_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def write_json(data, file_name, save_path="./", mode="r+"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_path = os.path.join(save_path, f"{file_name}.json")

    if not os.path.exists(full_path):
        with open(full_path, 'w') as file:
            pass
    elif mode == "w":
        with open(full_path, 'w') as file:
            json.dump(data, file, indent=4)
        return

    with open(full_path, 'r+') as file:
        for key, value in data.items():
            if os.path.getsize(full_path) == 0:
                file.write('{}')
                file.seek(0, os.SEEK_END)
                file.seek(file.tell() - 1, os.SEEK_SET)
                file.write('\n')
                file.write(json.dumps(key, indent=4))
                file.write(': ')
                file.write(json.dumps(value, indent=4))
                file.write('\n')
                file.write('}')
                continue

            file.seek(0, os.SEEK_END)
            file.seek(file.tell() - 2, os.SEEK_SET)
            file.write(',')
            file.write('\n')
            file.write(json.dumps(key, indent=4))
            file.write(': ')
            file.write(json.dumps(value, indent=4))
            file.write('\n')
            file.write('}')
    return
