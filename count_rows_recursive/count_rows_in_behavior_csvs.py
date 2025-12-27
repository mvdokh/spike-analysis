import os
import csv

def find_behavior_csvs(root_dir):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('behavior.csv'):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files

def count_rows(csv_path):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1  # subtract 1 for header
    except Exception as e:
        return f'ERROR: {e}'

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Count rows in all *behavior.csv files recursively in a folder.')
    parser.add_argument('--root', type=str, default=r'C:\Users\wanglab\Desktop\PCRt_TeLC', help='Root directory to search for behavior.csv files')
    parser.add_argument('--csv', type=str, default='behavior_csv_row_counts.csv', help='Output CSV file name')
    args = parser.parse_args()

    csv_files = find_behavior_csvs(args.root)
    if not csv_files:
        print('No behavior.csv files found.')
        return
    print(f'Found {len(csv_files)} behavior.csv file(s):\n')
    results = []
    for csv_file in csv_files:
        row_count = count_rows(csv_file)
        print(f'{csv_file}: {row_count} rows')
        results.append({'csv_path': csv_file, 'row_count': row_count})

    with open(args.csv, 'w', newline='', encoding='utf-8') as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=['csv_path', 'row_count'])
        writer.writeheader()
        writer.writerows(results)
    print(f'\nResults written to {args.csv}')

if __name__ == '__main__':
    main()
