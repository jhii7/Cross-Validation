import csv
import os
# for converting txt to csvs 

input_file = "/workspaces/Cross-Validation/TestingSets/NAMEHERE.txt"
output_file = "/workspaces/Cross-Validation/TestingSets/NAMEHERE.csv"
if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' does not exist.")
else:   
    with open(input_file, "r") as txt_file, open(output_file, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        for line in txt_file:  
            row = line.split() 
            csv_writer.writerow(row)
    print(f"Cconverted '{input_file}' to '{output_file}'.")
