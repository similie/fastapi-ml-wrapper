JSON loader from cubeJS

load data types from JSON Schema template

/post variables (or env) for setting load URI

POST /micro-service/retrain/
  start_date
  end_date
  station? (or all)

In Data loaders
base loader class
  - csv loader (from form post)
    POST a file to the upload end-point, 
    store in [tmp], 
    validate contents,
    update database with a new access token/user token pair pointing at the file
    post again with the returned token to trigger inference/training

  - json loader (from url request and date range)
    Needs a list of validated/trusted storage points that we are allowed to download from
    No date range = 'last 2 weeks' from the API provided by the api

  - PG loader (from post'ed date range and ENV config for connection)

### Sample Queries to load All weather from Cube aggregation server
  qq = {
    "query":{
      "measures":[
        "all_weather.avg_wind_direction",
        "all_weather.avg_wind_speed",
        "all_weather.avg_soil_moisture",
        "all_weather.avg_dew_point",
        "all_weather.avg_solar",
        "all_weather.avg_temperature",
        "all_weather.avg_humidity",
        "all_weather.avg_pressure",
        "all_weather.sum_precipitation"
        ],
      "timeDimensions":
        [{
          "dimension":"all_weather.date",
          "granularity":"hour",
          "dateRange":["2020-03-05","2020-03-12"]
        }],
        "order":{
          "all_weather.date":"asc"
        },
        "filters":[],
        "dimensions":[]
      }
    }

  q = {
    "query":{
      "measures":[
        "all_weather.avg_wind_direction",
        "all_weather.avg_wind_speed",
        "all_weather.avg_soil_moisture",
        "all_weather.avg_dew_point",
        "all_weather.avg_solar",
        "all_weather.avg_temperature",
        "all_weather.avg_humidity",
        "all_weather.avg_pressure",
        "all_weather.sum_precipitation"
      ],
      "timeDimensions":
      [{
        "dimension":"all_weather.date",
        "granularity":"hour",
        "dateRange":["2020-03-05","2020-03-12"]
      }],
      "order":{
        "all_weather.date":"asc"
      },
      "filters": [
        {
          "member": "all_weather.station",
          "operator": "equals",
          "values": ["27"]
        }
      ],
      "dimensions": [
        "all_weather.station"
      ]
    }
  }
