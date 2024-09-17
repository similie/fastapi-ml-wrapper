import json
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from .utils import (
    load_dataframe, 
    gen_datasets, 
    max_transform, 
    all_transforms,
    compute_stochastic_dropout,
    build_ae,
    train_ae,
    build_fc,
    train_fc,
    concatenate_latent_representation,
    MODULE_COLS
)


def runTrainingLoop(trainingStep: str, dims: int = 128):
    #device_context = tf.device('/metal')
    pathPrefix = './tmp' # path to the training data
    with open(f'{pathPrefix}/all_the_weather.json') as f:
        d = json.load(f)
        weather_data = d['data']

    data = load_dataframe(weather_data)
    data.head()

    X_train, X_test, y_train, y_test = gen_datasets(data, 12, shift=False)
    X_train_shift, X_test_shift, y_train_shift, y_test_shift = gen_datasets(data, 12, shift=True)
    print(X_train.shape, y_train.shape, X_train_shift.shape, y_train_shift.shape,\
          X_test.shape, X_test_shift.shape, y_test.shape, y_test_shift.shape)


    y_tst = max_transform(y_test)
    y_tr = max_transform(y_train)

    y_tr_shift = max_transform(y_train_shift)
    y_tst_shift = max_transform(y_test_shift)

    X_tr = all_transforms(X_train)
    X_tr_shift = all_transforms(X_train_shift)
    X_tst = all_transforms(X_test)
    X_tst_shift = all_transforms(X_test_shift)

    if trainingStep == 'ae':
        model = build_ae((12, 9), dims, 1)
        model.summary()

        model, history = train_ae(model, X_tr, X_tr_shift, X_tst, X_tst_shift)
        model.save('./autoencoder.keras')

        # Extract hidden layer from Autoencoder
        lstm = model.layers[1].output
        encoder = Model(model.input, lstm)

        encoder.save('./encoder.keras')

        # end of encoder training.
    elif trainingStep == 'fc':
        encoder = load_model('./encoder.keras')

        X_tr_, X_tst_ = concatenate_latent_representation(encoder, X_tr, X_tst)
        print(X_tr_.shape, X_tst_.shape)

        inputDim = (12, len(MODULE_COLS)+dims+1)
        fc_model = build_fc(inputDim, dims)

        fc_model.summary()

        fc_model, history = train_fc(fc_model, X_tr_, y_train_shift, X_tst_, y_test_shift)

        fc_model.save('./forecaster.keras')

        scores, pred_mean_error, pred_error_std = compute_stochastic_dropout(fc_model, X_tst_, y_tst)
        print(pred_mean_error, pred_error_std)

    else:
        print(f'Param trainingStep must be one of `ae` or `fc`, got: {trainingStep}')


    
if __name__ == "__main__":
    inputDims = 512
    print(f'len cols {len(MODULE_COLS)}, dims: {inputDims}, fc: {len(MODULE_COLS)+inputDims+1}')

    for step in ['ae', 'fc']:
        runTrainingLoop(trainingStep=step, dims=inputDims)
