# ae_simple
AE for time-series prediction
.  
├── dataset.py  
├── experiment  
│   └── task1  
├── layers  
│   ├── __init__.py  
│   ├── model.py  
│   └── __pycache__  *  
├── mutils.py  
├── preprocessor.py  
├── __pycache__ *  
├── README.md  
├── results  *  
├── task.py  
└── train.py  
7 directories, 15 files  
  *  ignored by git, see the .gitignore  
  
`Autoencoder` in the `task.py` file in root, **must**  
link to the csv in the `dataset.py` file in the `get_dm`  
class initializer... That's the only config  
except for the latent dimension, which is passed to the   
`_train` method, also in `task.py`. 
