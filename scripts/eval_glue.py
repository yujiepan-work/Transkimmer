import os
from pathlib import Path
import tempfile

import toytools
from toytools.batchrun import Launcher, Task, avail_cuda_list
from toytools.iterext import product
from toytools.misc import today_cipher, json_dump
from toytools.snapshot.log_python_env import log_python_env_status

root = Path('.').absolute().parent

env = os.environ.copy()
env["WANDB_DISABLED"] = "true"
env["WANDB_PROJECT"] = "debug"
env["WANDB_WATCH"] = "false"

cfgs = product(
    # skim_factor=[0.05, 0.1, 0.15, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.3]
    # skim_factor=[0.35, 0.4, 0.45, 0.5]
    skim_factor=[-1.]
)




tasks = []
for cfg in list(cfgs)[:]:
    folder = Path('/nvme2/yujiepan/workspace/moe-bert/moe-bert-learning/Transkimmer/model/glue', '0613-eval-ft-start-benchmark', f'factor{cfg.skim_factor}')
    task = Task(
        cmd=[f"""python -u run_glue_no_trainer.py \
            --model_type baseline \
            --skim_coefficient {cfg.skim_factor} \
            --model_name_or_path  /nvme2/yujiepan/workspace/moe-bert/moe-bert-learning/Transkimmer/model/glue/0608-ft-start/factor{cfg.skim_factor} \
            --tokenizer_name bert-base-uncased \
            --do_evaluate \
            --task_name qnli \
            --max_length 128 \
            --per_device_train_batch_size 32 \
            --gradient_accumulation_steps 2 \
            --per_device_eval_batch_size 1 \
            --learning_rate 2e-5 \
            --seed 42 \
            --num_train_epochs 3 \
            --output_dir {folder}
            """],
        cwd='/nvme2/yujiepan/workspace/moe-bert/moe-bert-learning/Transkimmer/src',
        io_folder=folder,
        identifier=folder.name,
        env=env,
        cuda_quantity=1,
        # prepare_fn=prepare_fn,
        # prepare_fn_args=(token_dropping_json, folder),
    )
    tasks.append(task)

Launcher([0]).run(tasks)
