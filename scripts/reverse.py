
import csv
import os
import pandas as pd

file_paths = [
    './virtual_database/DBCz.csv',
    './virtual_database/PA.csv',
    './virtual_database/ICz.csv'
]
save_directory = './result/reverse_design' 

search_configs = {
    'config1': {
        'search_type': 'max',
        'target_column_range': (242, 275),
        'value_range': (0.6, 0.8)
    },
    # 'config2': {
    #     'search_type': 'max',
    #     'target_column_range': (242, 275),
    #     'value_range': (0.8, 1.0)
    # },
    # 'config3': {
    #     'search_type': 'max',
    #     'target_column_range': (242, 275),
    #     'value_range': (1.0, 1.2)
    # },
    # 'config4': {
    #     'search_type': 'max',
    #     'target_column_range': (242, 275),
    #     'value_range': (1.2, 1.4)
    # },
    # 'config5': {
    #     'search_type': 'max',
    #     'target_column_range': (242, 275),
    #     'value_range': (1.4, 1.6)
    # },
    # 'config6': {
    #     'search_type': 'max',
    #     'target_column_range': (67, 100),
    #     'value_range': (1.4, 1.6)
    # },
    # 'config7': {
    #     'search_type': 'max',
    #     'target_column_range': (233, 266),
    #     'value_range': (1.4, 1.6)
    # },
    # 'config8': {
    #     'search_type': 'max',
    #     'target_column_range': (400, 433),
    #     'value_range': (1.4, 1.6)
    # },
    # 'config9': {
    #     'search_type': 'max',
    #     'target_column_range': (233, 266),
    #     'value_range': (0.9, 1.1)
    # },
    # 'config10': {
    #     'search_type': 'min',
    #     'target_column_range': (0, 17),
    #     'value_range': (-1.1, -0.9)
    # },
    # 'config11': {
    #     'search_type': 'max',
    #     'target_column_range': (317, 350),
    #     'value_range': (0.7, 0.9)
    # },
    # 'config12': {
    #     'search_type': 'min',
    #     'target_column_range': (317, 350),
    #     'value_range': (-1.3, -1.1)
    # },
    # 'config13': {
    #     'search_type': 'max',
    #     'target_column_range': (150, 183),
    #     'value_range': (0.9, 1.1)
    # },
    # 'config11': {
    #     'search_type': 'max',
    #     'target_column_range': (400, 433),
    #     'value_range': (0.7, 0.9)
    # },
    # 'config12': {
    #     'search_type': 'min',
    #     'target_column_range': (400, 433),
    #     'value_range': (-1.3, -1.1)
    # },
    # 'config13': {
    #     'search_type': 'max',
    #     'target_column_range': (317, 350),
    #     'value_range': (1.4, 1.6)
    # }
}

stretch_start, stretch_end = 80, 120
stretch_step = 1
stretch_size = (stretch_end - stretch_start) // stretch_step + 1  # 41
thickness_list = [30, 48, 80]  
dye_list = ["red", "blue", "yellow"] 

group_size = stretch_size 
dye_change_interval = group_size * len(thickness_list)  

def get_parameter_group(index):
    stretch = stretch_start + (index % stretch_size) * stretch_step
    thickness = thickness_list[(index // group_size) % len(thickness_list)]
    dye = dye_list[(index // dye_change_interval) % len(dye_list)]
    return dye, thickness, stretch

for config_name, config in search_configs.items():
    search_type = config['search_type']
    target_column_range = config['target_column_range']
    value_range = config['value_range']

    column_range_min = target_column_range[0]
    column_range_max = target_column_range[1]
    value_lower_bound = value_range[0]
    value_upper_bound = value_range[1]

    matching_records = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)

            for index, row in enumerate(reader):
                try:
                    values = [float(val) for val in row]

                    if search_type == 'max':
                        target_value = max(values)
                        target_index = values.index(target_value)
                    else:
                        target_value = min(values)
                        target_index = values.index(target_value)

                    if column_range_min <= target_index <= column_range_max and value_lower_bound <= target_value <= value_upper_bound:
                        dye, thickness, stretch = get_parameter_group(index) 
                        file_name = os.path.splitext(os.path.basename(file_path))[0] 
                        matching_records.append([file_name, dye, thickness, stretch, target_value])

                except (ValueError, IndexError):
                    continue

    
    csv_filename = f"{config_name}_{search_type}_{column_range_min}_{column_range_max}_{value_lower_bound}_{value_upper_bound}_.csv"
    xlsx_filename = f"{config_name}_{search_type}_{column_range_min}_{column_range_max}_{value_lower_bound}_{value_upper_bound}_.xlsx"
    csv_save_path = os.path.join(save_directory, csv_filename)
    xlsx_save_path = os.path.join(save_directory, xlsx_filename)

    with open(csv_save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Phosphorescent molecule", "Dye", "thickness", "stretch", f"{search_type}_value"])
        writer.writerows(matching_records)
