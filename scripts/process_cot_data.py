import os
import argparse
import shutil
import subprocess, json

parser = argparse.ArgumentParser()
parser.add_argument('directory', default='.', help='directory')
SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    args = parser.parse_args()

    def num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)

    stats = []

    for x in os.listdir(args.directory): 
        if x.endswith('_processed'):
            continue

        if not os.path.isdir(x):
            continue

        out = subprocess.check_output(
            [
                "python", 
                f"{SCRIPT_DIR}/../pretraining_analysis/process_pretrain_labelled.py",
                '--dataset_name', x
            ],
        )
        out = subprocess.check_output(
            [
                "python", 
                f"{SCRIPT_DIR}/../pretraining_analysis/get_stats.py",
                '--dataset_name', f'{x}_processed'
            ],
        )

        # data
        Y = [x.split(':') for x in out.decode().split('\n')]
        Y = [x for x in Y if len(x) == 2]
        Y = {k:num(v.split()[0]) for k,v in Y}
        Y['file'] = x

        print (json.dumps(Y))

        with open(f'{x}_stats.json', 'w') as f:
            json.dump(Y,f)
