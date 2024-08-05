KERNEL_INITIALIZER=glorot-uniform
OPTIMIZER=adam
EPOCHS=5
EXPERIMENTS_COUNT=10
SEQUENCE_SCHEME=auto
OUT_DIR=out

mkdir -p $OUT_DIR

# CV subset
for ((experiment_id=1; experiment_id<=EXPERIMENTS_COUNT; experiment_id++))
do
  for dataset in "mnist" "cifar10"
  do
    for model_id in "baseline-ann" "baseline-cnn"
    do
      for initializer_type in "pseudo-random" "quasi-random"
      do
        python src/train_and_eval.py \
            --experiment_id=${experiment_id} \
            --sequence_scheme=$SEQUENCE_SCHEME \
            --dataset_id=${dataset} \
            --kernel_initializer=$KERNEL_INITIALIZER \
            --initializer_type=${initializer_type} \
            --units=64 \
            --model_id=$model_id \
            --batch_size=64 \
            --epochs=$EPOCHS \
            --optimizer=$OPTIMIZER \
            --max_features=20000 \
            --out_path="${OUT_DIR}/results_${initializer_type}_${dataset}_${model_id}_${KERNEL_INITIALIZER}_${OPTIMIZER}.${experiment_id}.json"
      done
    done
  done
done

# NLP subset
for ((experiment_id=1; experiment_id<=EXPERIMENTS_COUNT; experiment_id++))
do
  for dataset in "imdb_reviews"
  do
    for model_id in "baseline-lstm" "baseline-transformer"
    do
      for initializer_type in "pseudo-random" "quasi-random"
      do
        python src/train_and_eval.py \
            --experiment_id=${experiment_id} \
            --sequence_scheme=$SEQUENCE_SCHEME \
            --dataset_id=${dataset} \
            --kernel_initializer=$KERNEL_INITIALIZER \
            --initializer_type=${initializer_type} \
            --units=64 \
            --model_id=$model_id \
            --batch_size=64 \
            --epochs=$EPOCHS \
            --optimizer=$OPTIMIZER \
            --max_features=20000 \
            --out_path="${OUT_DIR}/results_${initializer_type}_${dataset}_${model_id}_${KERNEL_INITIALIZER}_${OPTIMIZER}.${experiment_id}.json"
      done
    done
  done
done

# aggregate raw stats
python src/data/aggregate_data_files.py -i "$OUT_DIR" -m "results_*.json" -o "aggregate.parquet"
