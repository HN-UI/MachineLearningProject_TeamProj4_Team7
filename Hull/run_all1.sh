data=btc
GPU=0
for drop in None; do
        for seed in 10 20 30; do
        bash run.sh $GPU baseline_ols \
                        128 128 0.001 0.5 3 \
                        128 5 5 \
                        8 256 \
                        1000 16 1 1 \
                        $seed $drop $data \
                        --do_test

        bash run.sh $GPU baseline_enet \
                        128 128 0.001 0.5 3 \
                        128 5 5 \
                        8 256 \
                        1000 16 1 1 \
                        $seed $drop $data \
                        --do_test

        bash run.sh $GPU rf \
                        128 128 0.001 0.5 3 \
                        128 5 5 \
                        8 256 \
                        1000 16 1 1 \
                        $seed $drop $data \
                        --do_test

        bash run.sh $GPU lgbm \
                        128 128 0.001 0.5 3 \
                        128 5 5 \
                        8 256 \
                        20000 16 1 1 \
                        $seed $drop $data \
                        --do_test

        bash run.sh $GPU xgb \
                        128 128 0.001 0.5 3 \
                        128 5 5 \
                        8 256 \
                        20000 2 1 1 \
                        $seed $drop $data \
                --do_test

        bash run.sh $GPU lstm \
                128 128 0.001 0.8 2 \
                128 5 5 \
                8 256 \
                20000 16 1 1 \
                $seed $drop $data \
                --do_test

        bash run.sh $GPU tcn \
                        128 128 0.00005 0.0 3 \
                        32 5 5 \
                        8 256 \
                        20000 16 1 1 \
                        $seed $drop $data \
                        --do_test

        bash run.sh $GPU gru \
                        128 128 0.001 0.6 1 \
                        128 5 5 \
                        8 256 \
                        20000 2 1 1 \
                        $seed $drop $data \
                        --do_test

        bash run.sh $GPU tx \
                128 128 0.001 0.0 3 \
                128 5 5 \
                16 64 \
                20000 2 1 1 \
                $seed $drop $data \
                --do_test
        done
done