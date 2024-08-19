## Llama Guard Toxicity Analysis

This repository contains the code for our experiments with the Llama-Guard-3 model on the multilingual toxicity detection task. During our experiments, we found that the Guard model largely underperforms as compared to very slightly prompted Llama-3-8B and Llama-3.1-8B models. 

## Setting up your environment

This repository has only three dependencies. To install these, first create a virtual environment (we recommend python version >= 3.10) and activate it. Then run the following command:

```bash
pip install -r requirements.txt
```

This repository uses Together AI for querying the Llama-3-8B and Llama-3.1-8B models so please ensure to set your `TOGETHER_API_KEY`:

```bash
export TOGETHER_API_KEY="your_key_here"
```

## Executing the code

To run all experiments, please execute the following command:

```bash
bash run.sh
```

This script will create an `outputs/` directory which contains all the output CSV results. The script will also display the accuracy results for every dataset after the execution for that dataset is finished.

Feel free to change the `run.sh` parameters. You can use `--help` for more information on available commands.