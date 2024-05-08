Env vars inuse by application configs

ADMIN_EMAIL='noreply@similie.org'
API_ENDPOINT='api/v1'

#[Cube JS data aggregation system]#
# Rest-API
CUBE_NAME='all_weather'
CUBE_PORT=4000
CUBE_REST_API='http://localhost:4000/cubejs-api/v1/load'
CUBE_AUTH_KEY='insert cli generated uuid4 key here'
# PG access
CUBE_PG_HOST=''
CUBE_PG_PORT=54321
CUBE_PG_USER=''
CUBE_PG_PASS=''


# [ML Models config section]
# experiment_config:
experiment_name='experiment-1'
experiment_train_path='./data/combined.csv'
experiment_val_path='./data/combined.csv'
experiment_pred_path='./data/combined.csv'
experiment_check_path=''
experiment_weight_decay=1e-4
experiment_lr=0.02
experiment_batch_size=1
experiment_sequence_length=12
experiment_prediction_window=72
experiment_target_col='sum_precipitation'
experiment_groupby_col='station'

# LSTM Config:
lstm_name='tabularasa'
lstm_n_features=6
lstm_hidden_size=128
lstm_sequence_length=12
lstm_batch_size=1
lstm_num_layers=2
lstm_dropout=0.5

# trainer_config:
trainer_precision=64
trainer_accelerator='gpu'
trainer_default_root_dir='results'
trainer_max_epochs=40
trainer_log_every_n_steps=1  
