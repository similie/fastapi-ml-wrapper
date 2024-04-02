from predict import _predict
import os

data_path = os.path.join(os.getcwd(), "../tabula_rasa/data/combined.csv")
# predict_dict = {}
# for latent_dim in [64, 128, 256]: #, 256, 512, 1024]:
#     predictions = _predict(latent_dim, dm)
#     predict_dict[latent_dim] = predictions


prediction, new_feats, model = _predict(data_path, 64)
