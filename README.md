# QMUL-final-year-project

Github links:https://github.com/simone2401/QMUL-final-year-project/tree/main
This repository contains the code and experimental results for the undergraduate final-year project on query generation for automated fact-checking. The project explores reinforcement learning (GRPO) for improving retrieval-oriented query generation in automated fact-checking.


## Dataset

This project uses the FEVER dataset:

https://fever.ai/dataset/fever.html

The dataset was downloaded in January 2026. To run the code, please download the dataset manually and place it in a folder named:FEVER/
    

The folder should contain the following files:

- shared_task_dev_public.jsonl  
- shared_task_dev.jsonl  
- shared_task_test.jsonl  
- train.jsonl  
- wiki_license.html  
- wiki-pages/ (directory)

---

## Environment Setup

This project uses the `verl` framework (GRPO implementation).

If you encounter issues related to `verl`, please refer to:

https://verl.readthedocs.io/en/latest/algo/grpo.html

The version of `verl` used in this project corresponds to releases around March–April 2026.

### Key requirements:

- Python = 3.10  
- setuptools < 82  
- tensorboard  
- spacy  
- accelerate  

It is recommended to uninstall `torchao` if installed.

Additional dependencies can be found in: verl_requirements.txt


---

## Model

Please manually download:

- Qwen2.5-1.5B-Instruct  
- verl  

These are not included in the repository.

---

## Running the Code

Before running the code:

- Update all file paths in the code to match your local environment  
- Ensure the FEVER dataset is placed correctly  

This project was developed and executed on QMUL GPU servers.

The file `before_run.txt` provides reference steps for setting up the conda environment after installing Qwen2.5-1.5B-Instruct and verl.
The file `run.txt` provides reference steps for starting the experiment after `before_run.txt`.

---

## Project Structure

- `src/` – source code  
- `outputs/` – experimental results  
- `FEVER/` – FEVER dataset (must be downloaded manually)  

---

## Notes

- Some environment configurations may vary depending on your system  
- This repository is intended for research and reproducibility purposes  
