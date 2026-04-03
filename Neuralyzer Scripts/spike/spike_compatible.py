import csv

input_path = r"C:\Users\wanglab\Desktop\TG_PT\0112_2\spikes.csv"
output_path = r"C:\Users\wanglab\Desktop\TG_PT\0112_2\spikes_compatible.csv"

with open(input_path, "r") as infile, open(output_path, "w", newline="") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        value = float(line)
        # Format to 5 decimal places (e.g. 0.11110)
        formatted = f"{value:.5f}"
        # Match the spacing style: "     15.61363,   1,   1"
        outfile.write(f"     {formatted},   1,   1\n")

print(f"Done! Saved to: {output_path}")