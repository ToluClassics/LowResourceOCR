text_path=raw_data/trdg/eng_target.txt
dir_path=raw_data/trdg/eng_image/
model_name=microsoft/trocr-base-printed
split_size=0.2
max_target_length=256
no_epochs=10
training_batch_size=16
learning_rate=5e-5
max_length=64
processor_tokenizer=None
num_beams=5
model_outputdir=logs/${model_name}
no_repeat_ngram_size=3
length_penalty=2.0

CUDA_VISIBLE_DEVICE=6 python trocr_trainer.py --text_path ${text_path} \
    --dir_path ${dir_path} \
    --model_name ${model_name} \
    --split_size ${split_size} \
    --max_target_length ${max_target_length} \
    --no_epochs ${no_epochs} \
    --training_batch_size ${training_batch_size} \
    --learning_rate ${learning_rate} \
    --max_length ${max_length} \
    --no_repeat_ngram_size ${no_repeat_ngram_size} \
    --length_penalty ${length_penalty} \
    --num_beams ${num_beams} \
    --model_outputdir ${model_outputdir} \