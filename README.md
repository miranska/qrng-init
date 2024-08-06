# On Using Quasirandom Sequences in Machine Learning for Model Weight Initialization
![CI status badge](https://github.com/miranska/qrng-init/actions/workflows/ci.yml/badge.svg?branch=main)
[![Coverage badge](https://github.com/miranska/qrng-init/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/miranska/qrng-init/tree/python-coverage-comment-action-data)

This repository is the official implementation of "[On Using Quasirandom Sequences in Machine Learning for Model Weight Initialization](https://arxiv.org/abs/2408.02654)". 
<p align="center">    
    <br />
    <a href="src/demo.ipynb">View Demo</a>
  </p>

## Requirements

To install requirements:

```shell
conda create -n rnd-init python==3.11
conda activate rnd-init
pip install -r requirements.txt
```
Note that a correct Tensorflow install can be tricky, check the [instructions](https://www.tensorflow.org/install/pip) for your OS.


## Training & Evaluation

For details (on how to choose a model, dataset, etc.), run this command:
```shell
python src/train_and_eval.py --help
```

Here is a sample command to train and evaluate a baseline CNN model trained on 
the CIFAR-10 dataset using a quasi-random Glorot Uniform initializer: 
```shell
python src/train_and_eval.py \
    --experiment_id=1 \
    --sequence_scheme=auto \
    --dataset_id=cifar10 \
    --kernel_initializer=glorot-uniform \
    --initializer_type=quasi-random \
    --model_id=baseline-cnn \
    --units=64 \
    --batch_size=64 \
    --epochs=30 \
    --optimizer=adam \
    --learning_rate=0.0001 \
    --out_path=results.json
```

Sample execution harness is given in `test.sh`.

To set specific starting seed value, try `test_specific_seeds.sh`.

## Structure
### Distributions-related files

* `src/distributions_qr.py` contains implementations of `random_normal`, `random_uniform`, and `truncated_normal` distributions mimicking the API of the corresponding `keras.backend` functions. Our implementations are driven by quasirandom Sobol sequences rather than a pseudorandom number generator. 
* `tests/test_distributions_qr.py` contains additional examples of usage of these functions.

### Initializers
* `src/custom_random_generator.py` contains custom random generator that invokes quasirandom distributions from `src/distributions_qr.py`.
* `src/custom_initializer.py` contains ten custom initialization schemes that use quasirandom distributions as the underlying source of randomness. These initialization schemes adhere to the API `keras.initializers` classes and can be readily used to initialize Keras layers. We add the suffix `QR` (quasi-random) to each custom initialization schema name.

## Test
### Automatic unit tests

Run test cases (from the root directory of the project) and produce code coverage data using 
```shell
pytest
```
Pytest parameters are set in `pyproject.toml`.

Produce HTML code coverage report using
```shell
coverage html
```
Note that Github CI pipeline generates the latest code coverage report automatically and places it into 
`python-coverage-comment-action-data` branch.

## Citation
If you use the paper, algorithm, or code, please cite them as follows. 
```bibtex
@misc{qrng_init_2024,
      title={On Using Quasirandom Sequences in Machine Learning for Model Weight Initialization}, 
      author={Andriy Miranskyy and Adam Sorrenti and Viral Thakar},
      year={2024},
      eprint={2408.02654},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.02654},
}
```

## Contributing

> To contribute to the project, please feel free to submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

> This project is licensed under the Apache License 2.0 -- see the [LICENSE](LICENSE) file for details.
