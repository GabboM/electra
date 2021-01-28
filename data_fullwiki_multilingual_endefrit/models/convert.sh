# # Link from the official google repo
# wget https://storage.googleapis.com/electra-data/electra_small.zip      
# unzip electra_small.zip
# cd electra_small

# # If you're converting a different model you should make your own config.json file
# wget https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-generator/config.json

# Use the conversion script
MODEL=melectra_small_fullwiki
python convert_electra_original_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=$MODEL/model.ckpt-1000000 \
    --config_file=$MODEL/config.json \
    --pytorch_dump_path=$MODEL/pytorch_model.bin \
    --discriminator_or_generator=discriminator