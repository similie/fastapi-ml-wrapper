# Similie Multi-hazard early warning risk reduction system (MHEWS)  
## Web-services wrapper for AI/ML models
<img width="190" alt="similie-logo-navy" src="https://github.com/similie/fastapi-ml-wrapper/assets/77981429/51ad950c-6170-40ee-8dc0-d8e252d1cb34">

At [Similie](https://www.similie.com), our mission is to pioneer advanced machine learning solutions that empower communities and organizations to integrate actionable environmental data seamlessly into their systems.  

We are committed to developing open-source technologies that are accessible and scalable, fostering innovation in the assessment and management of natural disaster risks. Through collaboration and the continuous refinement of our models, we strive to provide robust tools that contribute to the resilience and safety of societies in the face of climate challenges.  

## Project documentation
The docs for the repo are [in the wiki](https://github.com/similie/fastapi-ml-wrapper/wiki) where you can find out more about the aims of the project, how the code is organised & how to contribute.

## Test coverage  
Go to [actions](https://github.com/similie/fastapi-ml-wrapper/actions), choose the latest run, then in the `jobs` menu on the left hand side, pick `build`, find the 'Test with pytest' action and expand the caret to the left. This will reveal the code coverage report for the build.  

## Project Overview
[Similie](https://www.similie.com) builds localized, multi-hazard early warning systems, focussed on data-poor and/or underserved communities. This repo and it's associated ML models seek to provide a solution for leveraging machine learning algorithms geared toward predicting various indicators relevant in the detection of flash floods and other natural disaster occurrences. In the event of natural disaster, we seek to build affordable solutions where individuals living in underserved economies can have advanced warnings required to move their resources and their families to safety. The model is a machine-learning model that offers predictive outputs of local environmental data aimed at improving our **time-to-event** metrics. 

We use data from a range of low-cost IoT sensors gathering precipitation, river water levels, soil moisture, and other atmospheric parameters to train forecasted precipitation outputs. Our next phases will attempt to use these prediction models to adjust thresholds for what constitutes a potential early warning event. For example, when precipitation at station A records X and the water level at station B records Y, what is the probability of an event for area C? X and Y are optimizations that these models will attempt to solve.

This initial attempt was trialed in Timor-Leste where the lack of adequate weather, water catchment, and other environmental data is scarce. Compounded by increasingly unpredictable climate change-based weather patterns and human-induced deforestation, flash flooding events have become a devastating force in an already vulnerable and resource poor economy. For example, on March 13th of 2020, Timor-Leste experienced an extreme weather event which cost millions of dollars for the Dili urban area.

![image](https://user-images.githubusercontent.com/29231033/113268610-867d8700-9312-11eb-999c-3f0d41a38868.png)

The event  data was recorded by weather stations owned  by GoTL and Similie.

![image](https://user-images.githubusercontent.com/29231033/113271869-f2adba00-9315-11eb-881b-6307b4ba3d9a.png)
  
## Super Quick Start
To see the project running locally in a Docker container  
1. Clone the repo
2. In a terminal window at the project root, type: `docker compose up`
  
When the container has finished building, open a browser and navigate to `http://localhost:5002/api/v1/` or check the api docs at 
`http://localhost:5002/docs`  
  
## Developer Quick Start
The development environment uses Python 3.12, it is strongly recommended that you use a virtual environment for development.  
1. Clone the repo
2. Set up your VENV using Python3.12 and activate it
3. PIP install requirements.txt
4. In a terminal window at the project root, type: `uvicorn app.main:app --reload --port=5002` We use the reload directive to auto-reload code changes during the development cycle
5. Test the webservice is running by opening a browser and navigating to `http://localhost:5002/api/v1/`
6. Checkout a new branch and start coding :) 
  
API Endpoint documentation is automatically published via the Starlette/FastAPI OpenAPI documentation process to `http://localhost:5002/docs` in a running developer environment.  
  
Check [the issues](https://github.com/similie/fastapi-ml-wrapper/issues) and [the wiki](https://github.com/similie/fastapi-ml-wrapper/wiki) if you want to contribute. 🫶  
