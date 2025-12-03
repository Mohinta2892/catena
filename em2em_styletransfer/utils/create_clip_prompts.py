import json

prompt = "Unprocessed electron microscopy images capturing the intricate ultrastructure of insect brain tissue at a cellular level." \
         " These high-resolution images reveal the complex architecture of neural networks," \
         " showcasing the fine details of synaptic connections, cell membranes," \
         " and intracellular organelles. The raw data provides a foundational view for detailed analysis " \
         "and interpretation of brain morphology and function in insects."

# Create the dictionary with image filenames as keys and descriptions as values
image_dict = {
    f"img_{str(i).zfill(5)}.png": prompt
    for i in range(96, 120)
}

# Define the file path to save the JSON file
file_path = '/media/samia/DATA/mounts/cephfs/img2img-turbo/data/test_prompts.json'

# Dump the dictionary to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(image_dict, json_file, indent=4)

print(f"json saved here: {file_path}")
