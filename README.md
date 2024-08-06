# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/miranska/qrng-init/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/\_\_init\_\_.py                            |        0 |        0 |        0 |        0 |    100% |           |
| src/custom\_initializers.py                    |       89 |        8 |       20 |        5 |     88% |65-76, 256, 269-270, 603 |
| src/datapipeline.py                            |       76 |       65 |       10 |        0 |     13% |12-40, 44-79, 88-114, 118-124 |
| src/distributions\_qr.py                       |       61 |        0 |        6 |        0 |    100% |           |
| src/global\_config.py                          |       24 |        0 |       10 |        1 |     97% |    48->64 |
| src/model/baseline\_ann.py                     |       14 |        0 |        2 |        1 |     94% |     9->12 |
| src/model/baseline\_ann\_one\_layer.py         |       14 |        0 |        2 |        1 |     94% |    13->16 |
| src/model/baseline\_cnn.py                     |       18 |        1 |        2 |        1 |     90% |        10 |
| src/model/baseline\_lstm.py                    |       18 |        0 |        2 |        1 |     95% |    14->17 |
| src/model/baseline\_transformer.py             |       22 |        0 |        2 |        1 |     96% |    26->29 |
| src/models.py                                  |       28 |        0 |       10 |        0 |    100% |           |
| src/train\_and\_eval.py                        |      171 |      113 |       68 |        1 |     32% |31-48, 52-57, 61-231, 243-276, 282-317, 494-607, 624-654, 658-664 |
| src/trainer.py                                 |       87 |       26 |       48 |        3 |     79% |43, 68-70, 83-88, 95, 98, 101, 104-126, 145-168 |
| src/utils.py                                   |       47 |       22 |        8 |        0 |     49% |     40-92 |
| tests/model/test\_baseline\_ann.py             |       55 |        0 |       18 |        0 |    100% |           |
| tests/model/test\_baseline\_ann\_one\_layer.py |       51 |        0 |       18 |        0 |    100% |           |
| tests/model/test\_baseline\_cnn.py             |       65 |        0 |       18 |        0 |    100% |           |
| tests/model/test\_baseline\_lstm.py            |       61 |        0 |       18 |        0 |    100% |           |
| tests/model/test\_baseline\_transformer.py     |       61 |        0 |       18 |        0 |    100% |           |
| tests/test\_auto\_seed\_selector.py            |       30 |        0 |       12 |        0 |    100% |           |
| tests/test\_custom\_initializers.py            |       88 |        1 |        6 |        1 |     98% |       153 |
| tests/test\_distributions\_qr.py               |      177 |        1 |       28 |        1 |     99% |       424 |
| tests/test\_global\_config.py                  |       44 |        1 |       12 |        1 |     96% |        86 |
| tests/test\_global\_config\_2.py               |       57 |        0 |        6 |        0 |    100% |           |
| tests/test\_manual\_increment\_schemes.py      |       29 |        0 |        4 |        0 |    100% |           |
| tests/test\_models.py                          |       35 |        4 |       18 |        0 |     92% |60-61, 93-94 |
| tests/test\_utils.py                           |       33 |        1 |        2 |        1 |     94% |        53 |
| tests/test\_weight\_order.py                   |       34 |        0 |        0 |        0 |    100% |           |
|                                      **TOTAL** | **1489** |  **243** |  **368** |   **19** | **82%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/miranska/qrng-init/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/miranska/qrng-init/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/miranska/qrng-init/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/miranska/qrng-init/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmiranska%2Fqrng-init%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/miranska/qrng-init/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.