import argparse
import h5py
import os

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, 
    default="data_dump", help="Directory to the data folder!")
parser.add_argument('--num_idxs_per_job', type=int,
    default=1200, help="f5")
# parser.add_argument('--case', type=int,
#     default=0, help="f5")
parser.add_argument("--alter", type=str, 
    default="aw00", help="aw00, aw04, aw08,aw-40, aw-44, aw-48, aw40, aw44, aw48")

args = parser.parse_args()
num_idxs_per_job = args.num_idxs_per_job
alter = args.alter

def main(args):
    jobs_dir = f"jobs_{alter}"
    jobs_dir = os.path.join(jobs_dir, "todo")
    if not os.path.exists(jobs_dir):
        os.makedirs(jobs_dir)

    cases = [0,1,2,3,4,5,6,7,8]
    cases = [1]
    # num_jobs
    num_cores = 24
    num_idxs = 1160
    num_jobs = num_idxs // num_idxs_per_job + 1
    script_name = f"model3_50years.py"

    # starter for job scripts
    header = "#!/bin/bash" + "\n"
    # header += "module load httpproxy" + "\n"
    header += "eval \"$(conda shell.bash hook)\"" + "\n"
    header += "conda activate" + "\n"
    header += "cd /home/$USER/scratch/eco_model/population-dynamic-model" + "\n"

    # generate jobs
    for i in range(num_jobs):
        start_idx = i * num_idxs_per_job
    
        if i== num_jobs:
            end_idx = num_idxs
        else:
            end_idx = (i+1) * num_idxs_per_job

        for j in cases:
            paras = f"--start_idx={start_idx} --end_idx={end_idx} --num_cores={num_cores} --alter={alter} --case={j}"
            cmd1 = f"python3 -u R03_create_folders_50years.py --case={j} --alter={alter}" + "\n"
            cmd2 = f"python3 -u {script_name} {paras}"
            bash_fn = f"{alter}_{i}_{j}.sh"
            bash_fn = os.path.join(jobs_dir, bash_fn)
            with open(bash_fn, "w") as f:
                f.write(header)
                f.write(cmd1)
                f.write(cmd2)

if __name__ == "__main__":
    main(args)
