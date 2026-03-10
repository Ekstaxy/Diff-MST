import yaml
import os

input_file = "data/musdb18-aug.yaml"
output_file = "data/musdb18-aug-names.yaml"

def extract_names():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    new_data = {}

    for split in ['val', 'train']:
        if split in data:
            new_data[split] = []
            for path_key in data[split].keys():
                # path_key is like "test/AM Contra - Heart Peripheral" or "train/A Classic Education - NightOwl"
                # We want just the song name part after the slash
                song_name = path_key.split('/')[-1]
                new_data[split].append(song_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(new_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Successfully extracted song names to {output_file}")

if __name__ == "__main__":
    extract_names()
