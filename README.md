# SAUCIE
An implementation of SAUCIE (Sparse Autoencoder for Clustering, Imputing, and Embedding) in Tensorflow.
***
This repository is a fork of the [original SAUCIE implementation](https://github.com/KrishnaswamyLab/SAUCIE). The authors let me use it for my Bachelor thesis and bump the used tensorflow version to 2, as well as build a web application using the model architecture.  
The code used for measuring the goodness of the dimensionality reduction can be found [here](https://github.com/pachterlab/CBP_2021).


## Usage
### Docker
The project is dockerized. You can use `base_tf_test.Dockerfile` to build an image. If you want to use the interactive streamlit application, run the container with `-p 8501:8501`.
### From the source repository
TBA
### Package
TBA
### On remote server
TBA

## The modules
The models (batch correction and dimensionality reduction/clustering versions) are prepared following the scikit-learn estimator standards. This means you can use the models as follow:
#### Batch correction
``` 
from saucie.saucie import SAUCIE_batches

saucie = SAUCIE_batches()
saucie.fit(data, batches)
cleaned = saucie.transform(data, batches)
```
#### Clustering and dimensionality reduction
``` 
from saucie.saucie import SAUCIE_labels

saucie = SAUCIE_labels()
saucie.fit(data)
encoded = saucie.transform(data)
labels = saucie.predict(data)
```
***  
More info about the parameters of the estimators can be found in the 'saucie/saucie.py' file.