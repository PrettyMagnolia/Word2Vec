#!/bin/bash

# 固定negative_sampling_size为5，调整window_size为2、4、6、8
negative_sampling_size=5
for window_size in 2 4 6 8
do
    python train.py --type cbow --log_dir runs/CBOW_experiment --window_size $window_size --negative_sampling_size $negative_sampling_size
done

# 固定window_size为4，调整negative_sampling_size为1、3、7
window_size=4
for negative_sampling_size in 1 3 7
do
    python train.py --type cbow --log_dir runs/CBOW_experiment --window_size $window_size --negative_sampling_size $negative_sampling_size
done

window_size=8
negative_sampling_size=5
# Skip实验
python train.py --type skipgram --log_dir runs/SkipGram_experiment --window_size $window_size --negative_sampling_size $negative_sampling_size
# text8语料库实验
python train.py --type cbow --log_dir runs/CBOW_experiment --window_size $window_size --negative_sampling_size $negative_sampling_size --file_path /home/yifei/code/word2vec/data/text8 --vocab_path /home/yifei/code/word2vec/data/vocab_text8.json
# 搜狗新闻语料库实验
python train.py --type cbow --log_dir runs/CBOW_experiment --window_size $window_size --negative_sampling_size $negative_sampling_size --file_path /home/yifei/code/word2vec/data/sougou.txt --vocab_path /home/yifei/code/word2vec/data/vocab_sougou.json

