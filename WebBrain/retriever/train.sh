CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python retriever_train.py -plm splade  \
                -tf data/train.tsv \
                -df data/dev.tsv \
                -sop data/splade_out -op data/splade_out\
                -qs 5e-4 -ds 5e-3 \
                -lr 2e-5 -bs 32 -e 1 -ql 16 -sl 256\
                -cp data/corpus.jsonl \
