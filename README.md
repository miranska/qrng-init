# Repository Coverage



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
| tests/model/test\_baseline\_ann.py             |       55 |        0 |       18 |        9 |     88% |23->25, 24->23, 25->24, 44->46, 45->44, 46->45, 66->68, 67->66, 68->67 |
| tests/model/test\_baseline\_ann\_one\_layer.py |       51 |        0 |       18 |        9 |     87% |23->25, 24->23, 25->24, 44->46, 45->44, 46->45, 66->68, 67->66, 68->67 |
| tests/model/test\_baseline\_cnn.py             |       65 |        0 |       18 |        9 |     89% |25->27, 26->25, 27->26, 46->48, 47->46, 48->47, 68->70, 69->68, 70->69 |
| tests/model/test\_baseline\_lstm.py            |       61 |        0 |       18 |        9 |     89% |26->28, 27->26, 28->27, 50->52, 51->50, 52->51, 75->77, 76->75, 77->76 |
| tests/model/test\_baseline\_transformer.py     |       61 |        0 |       18 |        9 |     89% |27->29, 28->27, 29->28, 50->52, 51->50, 52->51, 74->76, 75->74, 76->75 |
| tests/test\_auto\_seed\_selector.py            |       30 |        0 |       12 |        6 |     86% |6->8, 7->6, 8->7, 64->66, 65->64, 66->65 |
| tests/test\_custom\_initializers.py            |       88 |        1 |        6 |        3 |     96% |24->23, 87->74, 153 |
| tests/test\_distributions\_qr.py               |      177 |        1 |       28 |        1 |     99% |       424 |
| tests/test\_global\_config.py                  |       44 |        1 |       12 |        1 |     96% |        86 |
| tests/test\_global\_config\_2.py               |       57 |        0 |        6 |        0 |    100% |           |
| tests/test\_manual\_increment\_schemes.py      |       29 |        0 |        4 |        0 |    100% |           |
| tests/test\_models.py                          |       35 |        4 |       18 |        8 |     77% |33->36, 34->33, 35->34, 36->35, 60-61, 67->70, 68->67, 69->68, 70->69, 93-94 |
| tests/test\_utils.py                           |       33 |        1 |        2 |        1 |     94% |        53 |
| tests/test\_weight\_order.py                   |       34 |        0 |        0 |        0 |    100% |           |
|                                      **TOTAL** | **1489** |  **243** |  **368** |   **80** | **79%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://github.com/miranska/qrng-init/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/miranska/qrng-init/tree/python-coverage-comment-action-data)

This is the one to use if your repository is private or if you don't want to customize anything.



## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.