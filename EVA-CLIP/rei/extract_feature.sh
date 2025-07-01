export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MODEL=EVA02-CLIP-bigE-14-plus
export PRETRAINED='/llm_reco_ssd/yangzhuoran/model/EVA02_CLIP_E_psz14_plus_s9B/EVA02_CLIP_E_psz14_plus_s9B.pt'
export LAION_2B_DATA_PATH="/llm_reco_ssd/yangzhuoran/dataset/laion-conceptual-captions-12m-webdataset/data/00000.tar"

IMG_EMB_PATH="/llm_reco_ssd/yangzhuoran/dataset/laion-conceptual-captions-12m-webdataset/embedding/image_embedding"
TEXT_EMB_PATH="/llm_reco_ssd/yangzhuoran/dataset/laion-conceptual-captions-12m-webdataset/embedding/text_embedding"

cd /llm_reco_ssd/yangzhuoran/code/EVA/EVA-CLIP/rei

export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

source /llm_reco_ssd/yangzhuoran/anaconda3/etc/profile.d/conda.sh
conda activate rei_2

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK \
	--master_addr=$MASTER_ADDR --master_port=12355 --use_env training/main.py \
        --val-data=${LAION_2B_DATA_PATH} \
        --val-num-samples 2000000000 \
        --batch-size 1024 \
        --model ${MODEL} \
        --force-custom-clip \
        --pretrained ${PRETRAINED} \
        --extract-features \
        --img-emb-path ${IMG_EMB_PATH} \
        --text-emb-path ${TEXT_EMB_PATH} \
        --save-interval 10 \
        --enable-deepspeed
