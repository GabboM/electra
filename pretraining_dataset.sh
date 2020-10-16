#!/bin/bash

CORPUS='data_fullwiki_multilingual_endefrit/corpus'
VOCAB='data_fullwiki_multilingual_endefrit/multilingual_vocab.txt'
DESTDIR='data_fullwiki_multilingual_endefrit/pretrain_tfrecords'
MAXLEN=128
NUMPROC=8

python build_pretraining_dataset.py \
    --corpus-dir $CORPUS \
    --vocab-file $VOCAB \
    --output-dir $DESTDIR \
    --max-seq-length $MAXLEN \
    --num-processes $NUMPROC \
    