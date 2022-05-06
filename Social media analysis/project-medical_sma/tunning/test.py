import csv
with open(r"znanylekarz_dataset_all_clean.csv", 'r') as f:
    reader = csv.reader(f)
    linenumber = 1
    try:
        for row in reader:
            linenumber += 1
    except Exception as e:
        print (linenumber,e)