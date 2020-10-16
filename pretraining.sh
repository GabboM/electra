#!/bin/bash

DATADIR='data_fullwiki_multilingual_endefrit'
MODEL='melectra_small_fullwiki'

python run_pretraining.py \
    --data-dir $DATADIR \
    --model-name $MODEL \
    --hparams '{"vocab_size": 50000, "model_size": "small"}'