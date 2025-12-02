bash run.sh 0 baseline_ols \
        128 128 0.001 0.5 3 \
        128 5 5 \
        8 256 \
        1000 16 1 1 \
        42 None snp \
        --do_test

bash run.sh 0 baseline_enet \
        128 128 0.001 0.5 3 \
        128 5 5 \
        8 256 \
        1000 16 1 1 \
        42 None snp \
        --do_test

bash run.sh 0 rf \
        128 128 0.001 0.5 3 \
        128 5 5 \
        8 256 \
        1000 16 1 1 \
        42 None snp \
        --do_test

bash run.sh 0 lgbm \
        128 128 0.001 0.5 3 \
        128 5 5 \
        8 256 \
        20000 16 1 1 \
        42 None snp \
        --do_test

bash run.sh 0 xgb \
        128 128 0.001 0.5 3 \
        128 5 5 \
        8 256 \
        20000 2 1 1 \
        42 None snp \
        --do_test

bash run.sh 0 lstm \
        128 128 0.001 0.8 2 \
        128 5 5 \
        8 256 \
        20000 16 1 1 \
        42 None snp \
        --do_test

bash run.sh 0 tcn \
        128 128 0.00005 0.0 3 \
        32 5 5 \
        8 256 \
        20000 16 1 1 \
        42 None snp \
        --do_test

bash run.sh 0 gru \
        128 128 0.001 0.6 1 \
        128 5 5 \
        8 256 \
        20000 2 1 1 \
        42 None snp \
        --do_test

bash run.sh 0 tx \
        128 128 0.001 0.0 3 \
        128 5 5 \
        16 64 \
        20000 2 1 1 \
        42 None snp \
        --do_test