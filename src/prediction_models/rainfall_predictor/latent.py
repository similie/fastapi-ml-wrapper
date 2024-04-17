from mutils import plot_latent
from mutils import autoencoder_from_pretrained
from dataset import data_module
from preprocessor import load_data_csv
import pytorch_lightning as pl
import pandas as pd
import numpy as np

df = load_data_csv('./tmp/combined.csv')
df = df[df['station'] == '72']
df = df.iloc[-134:,:].copy()
print('\n',len(df),'\n')

model = autoencoder_from_pretrained(latent_dim=64)

dm = data_module(data=df)
dm.setup(stage='predict')
loader = dm.predict_dataloader
trainer = pl.Trainer(enable_checkpointing=False,
    accelerator="cpu",
    devices=1)

predictions = trainer.predict(model, loader)
predictions = predictions[-180:-8]
preds = np.stack([l[0][:,0].squeeze(0).numpy() for l in predictions[::12]], axis=0)
reals = np.stack([l[1][:,0].squeeze(0).numpy() for l in predictions[::12]], axis=0)
p_df = pd.DataFrame(preds, columns=dm.features)
r_df = pd.DataFrame(reals, columns=dm.features)

plot_latent(r_df, p_df)

