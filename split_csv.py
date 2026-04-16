import csv
import os
import sys

input_file = r"data\enron\emails.csv"
output_dir = r"data\enron\chunks"
rows_per_chunk = 30000

os.makedirs(output_dir, exist_ok=True)

# Increase CSV field size limit for very large email bodies
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = max_int // 10

with open(input_file, "r", newline="", encoding="utf-8", errors="replace") as infile:
    reader = csv.reader(infile)
    header = next(reader)

    chunk_num = 1
    row_count = 0
    outfile = None
    writer = None

    for row in reader:
        if row_count % rows_per_chunk == 0:
            if outfile:
                outfile.close()
            chunk_path = os.path.join(output_dir, f"emails_chunk_{chunk_num:03d}.csv")
            outfile = open(chunk_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(outfile)
            writer.writerow(header)
            chunk_num += 1

        writer.writerow(row)
        row_count += 1

    if outfile:
        outfile.close()

print(f"Done. Created {chunk_num - 1} chunk files in {output_dir}")