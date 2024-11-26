import csv
import datetime
import pytz
from utils import xlsxWriter, DataManager

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def transform_data(csv_data, fps, site_location, names, local_tz):
    # Reverse the names dictionary to get the key from the value
    names = {v: k for k, v in names.items()}

    transformed_data = {
        'site_location': site_location,
        'start_datetime': datetime.datetime.strptime(csv_data[0][0], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=pytz.utc).astimezone(local_tz),
        'CROSSED': {}
    }

    for row in csv_data:
        timestamp_str, direction, vehicle_type = row
        timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=pytz.utc).astimezone(local_tz).timestamp()
        frame = int((timestamp - transformed_data['start_datetime'].timestamp()) * fps )  # Assuming 30 FPS
        transformed_data['CROSSED'][timestamp] = (frame, names[vehicle_type.strip()], direction.strip())

    return transformed_data

def write_to_excel(transformed_data, output_path):
    data_manager = DataManager()
    data_manager.site_location = transformed_data['site_location']
    data_manager.start_datetime = transformed_data['start_datetime']
    data_manager.CROSSED = transformed_data['CROSSED']
    
    writer = xlsxWriter()
    writer.write_to_excel(output_path, data_manager)

if __name__ == "__main__":
    # Set variables
    fps = 30
    site_location = 'Example Location'  
    csv_file_path = '2024-10-21T055715.854Z.csv.csv'  # Path to your CSV file
    output_excel_path = 'output.xlsx'  # Path to save the Excel file
    names = {0: 'Car', 1: 'Van', 2: 'Bus', 3: 'Motorcycle', 4: 'Lorry'}  # Mapping of vehicle types as model.names dictionary format
    local_tz = pytz.timezone('Singapore')  # Set your local timezone here

    # Read, transform, and write data
    csv_data = read_csv(csv_file_path)
    transformed_data = transform_data(csv_data, fps, site_location, names, local_tz)
    write_to_excel(transformed_data, output_excel_path)

    print(f"Excel file has been created at {output_excel_path}")