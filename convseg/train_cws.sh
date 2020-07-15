corpus=$1
model_root=model-${corpus}
if [ ! -d ${model_root} ];then
    mkdir ${model_root}
fi
python train.py   --task cws \
                        --training_path data/datasets/sighan2005-${corpus}/train.txt \
                        --dev_path data/datasets/sighan2005-${corpus}/dev.txt \
                        --test_path data/datasets/sighan2005-${corpus}/test.txt \
                        --pre_trained_emb_path data/embeddings/news_tensite.w2v200 \
                        --model_root ${model_root} \
                        --word_window 0 \
                        >>${model_root}/stdout.txt 2>>${model_root}/stderr.txt &

echo "Model and log are saved in ${model_root}."