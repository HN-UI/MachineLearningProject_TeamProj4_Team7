# 'baseline_ols','baseline_enet',
#         'rf','lgbm','xgb',
#         # sequence models
#         'lstm','gru','tcn','tx'
GPU=1
# for seed in 10; do
#     for h in 1 2 4 8 16; do
#         for ff in 32 64 128 256; do 
#             bash run.sh $GPU tx \
#             128 128 0.001 0.0 3 \
#             128 5 5 \
#             $h $ff \
#             20000 2 1 1 \
#             $seed \
#             --do_cross
#         done
#     done
# done


# bash run.sh 1 lstm \
# 128 128 0.001 0.5 3 \
# 128 5 5 \
# 8 256 \
# 1000 1 1 1 \
# 0 \
# --do_cross
# D M E I P V S MOM1 MOM5
data=snp
for drop in None; do
        for seed in 120; do
        # bash run.sh $GPU baseline_ols \
        #                 128 128 0.001 0.5 3 \
        #                 128 5 5 \
        #                 8 256 \
        #                 1000 16 1 1 \
        #                 $seed $drop $data \
        #                 --do_test

        # bash run.sh $GPU baseline_enet \
        #                 128 128 0.001 0.5 3 \
        #                 128 5 5 \
        #                 8 256 \
        #                 1000 16 1 1 \
        #                 $seed $drop $data \
        #                 --do_test

        # bash run.sh $GPU rf \
        #                 128 128 0.001 0.5 3 \
        #                 128 5 5 \
        #                 8 256 \
        #                 1000 16 1 1 \
        #                 $seed $drop $data \
        #                 --do_test

        # bash run.sh $GPU lgbm \
        #                 128 128 0.001 0.5 3 \
        #                 128 5 5 \
        #                 8 256 \
        #                 20000 16 1 1 \
        #                 $seed $drop $data \
        #                 --do_test

        # bash run.sh $GPU xgb \
        #                 128 128 0.001 0.5 3 \
        #                 128 5 5 \
        #                 8 256 \
        #                 20000 2 1 1 \
        #                 $seed $drop $data \
        #         --do_test

        bash run.sh $GPU lstm \
                128 128 0.001 0.8 2 \
                128 5 5 \
                8 256 \
                20000 16 1 1 \
                $seed $drop $data \
                --do_test

        # bash run.sh $GPU tcn \
        #                 128 128 0.00005 0.0 3 \
        #                 32 5 5 \
        #                 8 256 \
        #                 20000 16 1 1 \
        #                 $seed $drop $data \
        #                 --do_test

        # bash run.sh $GPU gru \
        #                 128 128 0.001 0.6 1 \
        #                 128 5 5 \
        #                 8 256 \
        #                 20000 2 1 1 \
        #                 $seed $drop $data \
        #                 --do_test

        # bash run.sh $GPU tx \
        #         128 128 0.001 0.0 3 \
        #         128 5 5 \
        #         16 64 \
        #         20000 2 1 1 \
        #         $seed $drop $data \
        #         --do_test
        done
done