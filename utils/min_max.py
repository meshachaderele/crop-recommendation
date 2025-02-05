import json

min_max_values = {}
for column in df.select_dtypes(include=[np.number]).columns:
    min_max_values[column] = {
        'min': df[column].min(),
        'max': df[column].max()
    }

# Save the dictionary to a JSON file
output_file = 'min_max_values.json'
with open(output_file, 'w') as f:
    json.dump(min_max_values, f, indent=4)