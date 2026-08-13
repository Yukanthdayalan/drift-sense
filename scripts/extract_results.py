import csv
import shutil
import os

input_file = r"C:\Users\YUKANTH\drift_sense\results\final_submission_50_results.csv"
hard_case_file = r"C:\Users\YUKANTH\drift_sense\results\hard_case_results.csv"
gen_file = r"C:\Users\YUKANTH\drift_sense\results\generalization_results.csv"

# Generalization results is just the 50 samples evaluation which tests randomly generated conditions
shutil.copy(input_file, gen_file)

with open(input_file, 'r', newline='') as f_in, open(hard_case_file, 'w', newline='') as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        # A hard case is defined as having an error >= 2.0 pixels
        if float(row['error_px']) >= 2.0:
            writer.writerow(row)

print("Created hard_case_results.csv and generalization_results.csv")
