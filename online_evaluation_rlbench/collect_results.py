import argparse
import json
import os


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    parser.add_argument('--folder', type=str)
    parser.add_argument('--output_file', type=str, default=None)

    return parser.parse_args()


args = parse_arguments()
FOLDER = args.folder
output_file = args.output_file or os.path.join(FOLDER, 'all_results.txt')

sum_ = 0
tasks = sorted(os.listdir(FOLDER))
results = []
lines = []
for folder in tasks:
    if not os.path.exists(f'{FOLDER}/{folder}/eval.json'):
        continue
    with open(f'{FOLDER}/{folder}/eval.json') as fid:
        res = 100 * json.load(fid)[folder]["mean"]
    results.append(res)
    line = f'{folder} {res}'
    print(line)
    lines.append(line)
    sum_ += res

if results:
    mean_line = f'Mean on {len(results)} tasks {sum_ / len(results)}'
else:
    mean_line = 'Mean on 0 tasks nan'

print(mean_line)
lines.append(mean_line)

with open(output_file, 'w') as fid:
    fid.write('\n'.join(lines))
    fid.write('\n')
