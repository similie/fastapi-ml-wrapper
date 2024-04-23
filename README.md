# Project Beginn

![Logo for Project Beginn](./static/img/logo2.webp, width=200)

At Project Beginn, our mission is to pioneer advanced machine learning solutions that empower communities and organizations to integrate actionable environmental data seamlessly into their systems. We are committed to developing open-source technologies that are accessible and scalable, fostering innovation in the assessment and management of natural disaster risks. Through collaboration and the continuous refinement of our models, we strive to provide robust tools that contribute to the resilience and safety of societies in the face of climate challenges.


### Env variables in use:
```
ADMIN_EMAIL='noreply@similie.org'
API_ENDPOINT='api/v1'

# [Cube JS data aggregation system]#
# Rest-API
CUBE_NAME='all_weather'
CUBE_PORT=4000
CUBE_REST_API='http://localhost:4000/cubejs-api/v1/load'
CUBE_AUTH_KEY='1234-ABCDEF01-0000-0000'
# PG access
CUBE_PG_HOST=''
CUBE_PG_PORT=54321
CUBE_PG_USER=''
CUBE_PG_PASS=''


# [ML Models config section]
# experiment_config:
experiment_prediction_window=12
experiment_target_col=['precipitation']
experiment_features=['precipitation', 
                      'temperature', 
                      'humidity',
                      'pressure',
                      'wind_speed',
                      'wind_direction',
                      'solar']

# trainer_config:
trainer_accelerator='cpu'
trainer_dtype='np.float32'
trainer_pretrained_path='pretrained_checkpoints'
trainer_num_workers=1  
```
