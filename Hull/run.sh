GPU=$1
MODEL=$2
HIDDEN=$3
BATCH=$4
LR=$5
DR=$6
LAYERS=$7

TCN_CHANNELS=$8
TCN_LEVELS=$9
TCN_KERNEL=${10}

TX_HEADS=${11}
TX_FF=${12}

NUM_ESTIMATOR=${13}
MAX_DEPTH=${14}
SUBSAMPLE=${15}
COLSAMPLE=${16}

SEED=${17}
DROP=${18}
DATA=${19}

EXTRA_ARGS="${@:20}" 

CUDA_VISIBLE_DEVICES=$GPU python -u ./main.py --seed $SEED --data $DATA \
    --model $MODEL \
    --hidden $HIDDEN \
    --batch $BATCH \
    --lr $LR \
    --dropout $DR \
    --layers $LAYERS \
    --tcn_channels $TCN_CHANNELS \
    --tcn_levels $TCN_LEVELS \
    --tcn_kernel $TCN_KERNEL \
    --tx_heads $TX_HEADS \
    --tx_ff $TX_FF \
    --n_est $NUM_ESTIMATOR \
    --max_depth $MAX_DEPTH \
    --subsample $SUBSAMPLE \
    --colsample $COLSAMPLE \
    --drop $DROP \
    $EXTRA_ARGS
