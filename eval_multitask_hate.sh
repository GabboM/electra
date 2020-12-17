#!/bin/bash

#MULTILINGUAL

DATA_DIR='data_fullwiki_multilingual_endefrit'
SIZE='small'
MODEL='melectra_'$SIZE'_fullwiki'
# TASK='sentiment'
OUTPUTDIR=$DATA_DIR'/models/'$MODEL'/finetuning_models/hate,german_hate_model_1'
EXPORTDIR=$DATA_DIR'/exported_models/'$MODEL

python3 run_finetuning.py \
    --data-dir $DATA_DIR \
    --model-name $MODEL \
    --hparams '{"model_size": "'$SIZE'", "task_names": ["hate","german_hate"], "do_train": false, "do_eval": true, "init_checkpoint": "'$OUTPUTDIR'", "learning_rate" : 1e-4, "model_hparam_overrides": {"vocab_size": 50000}, "use_tfrecords_if_existing" : false}'


#GERMAN

# DATA_DIR='data_fullwiki_multilingual_endefrit'
# MODEL='melectra_small_fullwiki'
# TASK='german_hate'


# DATA_DIR='data_multilingual'
# MODEL='melectra_small'
# TASK='german_hate'


# python3 run_finetuning.py \
#     --data-dir $DATA_DIR \
#     --model-name $MODEL \
#     --hparams '{"model_size": "small", "task_names": ["'$TASK'"], "do_train": false, "do_eval": true, "init_checkpoint": "'$DATA_DIR'/models/'$MODEL'/finetuning_models/'$TASK'_model_1", "model_hparam_overrides": {"vocab_size": 50000}, "learning_rate" : 1e-4}'


#PRETRAINED

# DATA_DIR='electra_pretrained'
# MODEL='electra_small'
# TASK='german_hate'

# python3 run_finetuning.py \
#     --data-dir $DATA_DIR \
#     --model-name $MODEL \
#     --hparams '{"model_size": "small", "task_names": ["'$TASK'"], "do_train": false, "do_eval": true, "init_checkpoint": "'$DATA_DIR'/models/'$MODEL'/finetuning_models/'$TASK'_model_1", "learning_rate" : 1e-4}'
