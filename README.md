# Does syntactic parsing improve eye tracking prediction?

In this preliminary word we investigate whether training an LLM on a parsing objective improves the predictiveness of its output embeddings with respect to eye-tracking metrics.

The code is mostly based on [IncPar](https://github.com/anaezquerro/incpar).

# Prerequisites

The list of prerequisites can be found at https://github.com/anaezquerro/incpar. The eye-tracking data is provided as part of the [CMCL 2021 Shared Task](https://competitions.codalab.org/competitions/28176). You can still register to acquire the data.

# Results

You can find a description of our approach and the results of our experiments in the [technical report](Technical_Report.pdf).

# Usage

In order to reproduce our experiments, follow these steps:

### 1. Train parser
Train an Attach-Juxtapose parser with GPT-2 as the base with [IncPar](https://github.com/anaezquerro/incpar). Finetune GPT-2 as part of the training process. This can be done with the code provided as part of this repository using the following command:

```
python -u -m supar.cmds.const.aj --device 0 train -b -c configs/config-mgpt.ini     -p ../results/models-con/ptb/model/parser.pt     --delay=0 --use_vq     --train ../treebanks/ptb-gold/train.trees     --dev ../treebanks/ptb-gold/dev.trees     --test ../treebanks/ptb-gold/test.trees
```

### 2. Predict eye-tracking metrics
Train a linear regressor by running the experiment.py script. The arguments are structured as follows

```
python experiment.py MODEL METRIC FROM_TOKEN MODEL_PATH NEW_MODEL_NAME
```

where 
- MODEL is either 'vanilla' (non-finetuned GPT-2), 'aj' (finetuned as part of the Attach-Juxtapose parser) or 'both' (concatenation of both representations)
- METRIC is one of these metrics 'FixProp', 'TRT', 'GPT', 'nFix', 'FFD'
- FROM_TOKEN is either 'current', 'previous' or 'both' (from which token's representation to predict the current token's eye tracking measures)
- MODEL_PATH: path to the fine-tuned Attach-Juxtapose parser model; here '../results/models-con/ptb/model/'
- NEW_MODEL_NAME: name to use for saving the final model and results
