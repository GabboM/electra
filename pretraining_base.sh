#!/bin/bash

DATADIR='data_fullwiki_multilingual_endefrit'
MODEL='melectra_base_fullwiki'

python run_pretraining.py \
    --data-dir $DATADIR \
    --model-name $MODEL \
    --hparams '{"vocab_size": 50000, "model_size": "base", "learning_rate": 1e-4, "train_batch_size": 32, "eval_batch_size": 32}'