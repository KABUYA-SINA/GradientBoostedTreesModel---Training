import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf

# Gettin' Data
path = "https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins.csv"
dataset = pd.read_csv(path)
label = "species"

# Display the first 3 examples.
dataset.head(3)
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label=label)

tuner = tfdf.tuner.RandomSearch(num_trials=20)

# Hyper-parameters to optimize.
tuner.choice("max_depth", [4, 5, 6, 7])


model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
model.fit(tf_dataset)

print(model.summary())