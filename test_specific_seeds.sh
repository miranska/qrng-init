KERNEL_INITIALIZER=glorot-uniform
OPTIMIZER=adam
EPOCHS=5
EXPERIMENTS_COUNT=10
OUT_DIR=out
STARTING_DIM_ID=1
SEQUENCE_SCHEME=increment-start

mkdir -p $OUT_DIR

# CV subset
for initializer_type in  "pseudo-random" "quasi-random"
do
  for dataset in "mnist" "cifar10"
  do
    for model_id in "baseline-ann" "baseline-cnn"
    do
      for ((experiment_id=1; experiment_id<=EXPERIMENTS_COUNT; experiment_id++))
      do
        python src/train_and_eval.py \
            --experiment_id=${experiment_id} \
            --sequence_scheme=$SEQUENCE_SCHEME \
            --min_dim_id=$STARTING_DIM_ID \
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
# Note that seed setting does not match old LSTM setup exactly: it has 2 initializers that use different number of dimensions
for initializer_type in "pseudo-random" "quasi-random"
do
  for dataset in "imdb_reviews"
  do
    for model_id in "baseline-lstm" "baseline-transformer"
    do
      for ((experiment_id=1; experiment_id<=EXPERIMENTS_COUNT; experiment_id++))
      do
        python src/train_and_eval.py \
            --experiment_id=${experiment_id} \
            --sequence_scheme=$SEQUENCE_SCHEME \
            --min_dim_id=$STARTING_DIM_ID \
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
