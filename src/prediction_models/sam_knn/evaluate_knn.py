import json
import numpy as np
np.float = float # hack for multiflow
import pandas as pd 
from utils import load_knn, pickle_knn
from preprocessor import load_dataframe
from data import (max_transform, 
                    standard_transform, 
                    max_inverse_transform,
                    standard_inverse_transform,
                    onehot_transform,
                    groupdf)
from skmultiflow.data import DataStream
from skmultiflow.evaluation import EvaluatePrequential
import knn
from knn import SAMKNNRegressor
import matplotlib.pyplot as plt


# if __name__ == "__main__":

with open('./tmp/all_the_weather.json') as f:
    d = json.load(f)
    weather_data = d['data']

# sam = load_knn('./pickle/sam_knn.pkl')
sam_no_scale = load_knn('./pickle/sam_no_scal.pkl')

df = load_dataframe(weather_data)

df = groupdf(df.copy())

X = df.drop(['precipitation', 'station'], axis=1).values
y = df['precipitation'].values
assert len(X) == len(y)
print(X.shape, y.shape)

# Z_s = standard_transform(X)
# R_s = onehot_transform(X)
# X_s = np.concatenate((Z_s, R_s), axis=-1)
# y_s = max_transform(y)

assert len(X) == len(y)

# Init Data Stream from np.array
ds = DataStream(X, y=y)
ds.prepare_for_use()

evaluator = EvaluatePrequential(show_plot=False, # in a CLI
                            output_file='./plots/eval.pdf',
                            n_wait=730,
                            batch_size=6,
                            metrics=[
                                'mean_square_error',
                                'true_vs_predicted'])

evaluator.evaluate(
    stream=ds,
    model=[sam_no_scale],
    model_names=["SAM"])

pickle_knn(sam_no_scale, path='./pickle/sam_no_scale.pkl')
# in `./pickle/` folder

    
