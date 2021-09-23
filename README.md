# Kaggle_SETI_Signal_Search

This is a Kaggle competition: https://www.kaggle.com/c/seti-breakthrough-listen/overview

**Description**:

In this competition, use your data science skills to help identify anomalous signals in scans of Breakthrough Listen targets.
Because there are no confirmed examples of alien signals to use to train machine learning algorithms, the team included some simulated signals (that they call “needles”) in the haystack of data from the telescope.
They have identified some of the hidden needles so that you can train your model to find more.
The data consist of two-dimensional arrays, so there may be approaches from computer vision that are promising, as well as digital signal processing, anomaly detection, and more.
```
Libraries used:

import os
import gc
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split

import tensorflow as tf
import efficientnet.tfkeras as efn
import kerastuner as kt
import tensorflow_addons as tfa

import autokeras as ak

from autogluon.tabular import TabularDataset, TabularPredictor
```
**Spectrogram example**:

![SETI Signal](https://github.com/GaetanPelletier/Kaggle_SETI_Signal_Search/blob/main/SETI_signal.png)

# NB :
The competition was reset. Indeed, data leakage problems have interfered with the participants' results.

These files are no longer relevant from a performance point of view.

As part of my training as a machine learning engineer, I presented my work before the competition was rebooted. In order for the jury to evaluate my work, this repository will not be updated.
