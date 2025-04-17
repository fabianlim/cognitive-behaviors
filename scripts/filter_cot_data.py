#!bin/python


import os
import subprocess
import argparse

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

parser = argparse.ArgumentParser()
parser.add_argument('--shard_index', type=int, default=1, help='Shard index')
parser.add_argument('--shard_size', type=int, default=1000000, help='the shard size per node')
parser.add_argument('--gpus', type=int, default=2, help='total number of gpus for this shard')
parser.add_argument('--tp_degree', type=int, default=2, help='tp_degree')
parser.add_argument('--save_every', type=int, default=10000, help='Save every N examples')
parser.add_argument('--dataset_name', type=str, default='open-web-math', help='Dataset to process')

SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':

    args = parser.parse_args()
    N = args.gpus // args.tp_degree
    start = args.shard_index * args.shard_size
    mini_shard_size = args.shard_size // N

    all_processes, files = [], []

    for i in range(N):

        s = start + i * mini_shard_size
        e = start + (i+1) * mini_shard_size

        out_file = open(
            f"out_{args.dataset_name}_s_{s}_e_{e}_{i}.log", "wb", 0
        )

        print ('run process', i)
        process = subprocess.Popen(
            [
                "python", 
                "pretraining_analysis/relabel_pretrain.py",
                "--start", str(s),
                "--end", str(e),
                "--dataset_name", args.dataset_name,
                "--save_every", str(args.save_every),
            ],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": ",".join([
                    str(x) for x in range(
                        args.tp_degree * i,
                        args.tp_degree * (i+1)
                    )
                ]),
            },
            cwd=f'{SCRIPT_DIR}/..',
            stdout=out_file,
            stderr=out_file,
        ) 

        all_processes.append(process)
        files.append(out_file)

    for process in all_processes:
        process.wait()
        process.kill()
        print ('process', i, 'finished')

    for f in files:
        f.close()

