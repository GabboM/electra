#!/bin/bash

#MULTILINGUAL

DATA_DIR='data_fullwiki_multilingual_endefrit'
SIZE='small'
MODEL='melectra_'$SIZE'_fullwiki'
TASK='hate'

python3 run_finetuning.py \
    --data-dir $DATA_DIR \
    --model-name $MODEL \
    --hparams '{"model_size": "'$SIZE'", "task_names": ["emotion","hate","sentiment"], "do_eval": true, "learning_rate" : 1e-4, "model_hparam_overrides": {"vocab_size": 50000}}'


#!/bin/bash

#MULTILINGUAL

# DATA_DIR='data_fullwiki_multilingual_endefrit'
# SIZE='small'
# MODEL='melectra_'$SIZE'_fullwiki'
# TASK='eng_ger_hate'

# python3 run_finetuning.py \
#     --data-dir $DATA_DIR \
#     --model-name $MODEL \
#     --hparams '{"model_size": "'$SIZE'", "task_names": ["'$TASK'"], "do_eval": true, "learning_rate" : 1e-4, "model_hparam_overrides": {"vocab_size": 50000}}'
#MULTILINGUAL

# DATA_DIR='data_fullwiki_multilingual_endefrit'
# MODEL='melectra_small_fullwiki'

# python3 run_finetuning.py \
#     --data-dir $DATA_DIR \
#     --model-name $MODEL \
#     --hparams '{"model_size": "small", "task_names": ["german_hate"], "do_eval": true, "model_hparam_overrides": {"vocab_size": 50000}, "learning_rate" : 1e-4}'


#PRETRAINED small

# DATA_DIR='electra_pretrained'
# MODEL='electra_small'

# python run_finetuning.py \
#     --data-dir $DATA_DIR \
#     --model-name $MODEL \
#     --hparams '{"model_size": "small", "task_names": ["hate"], "do_eval": true, "learning_rate" : 1e-4}'


#PRETRAINED base

# DATA_DIR='electra_pretrained'
# MODEL='electra_base'

# python run_finetuning.py \
#     --data-dir $DATA_DIR \
#     --model-name $MODEL \
#     --hparams '{"model_size": "base", "task_names": ["emotion"], "do_eval": true, "learning_rate" : 1e-4}'