import sys
from os import path
import json

from predict import _predict
from preprocessor import load_dataframe, load_data_csv
from mutils import plot_predictions                

if __name__ == "__main__":

    df = load_data_csv('./tmp/combined.csv')
    df = df[df['station'] == '72']
    df = df.iloc[-134:,:]
    predictions, dm, data = _predict(data=df)
    df = load_dataframe(data)
    preds = dm.process_preds(predictions)
    
    plot_predictions(preds, df)
    
    # pdf saved to the results folder root