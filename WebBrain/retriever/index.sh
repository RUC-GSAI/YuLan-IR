CUDA_VISIBLE_DEVICES=0,1,2,3 python retriever_index.py -plm splade -ck data/checkpoint.pt \
                -si data/index -ibs 128 -cp data/corpus.jsonl \
                -st data/splade_tmp
