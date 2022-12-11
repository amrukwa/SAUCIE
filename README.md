![Run unit tests](https://github.com/amrukwa/SAUCIE/actions/workflows/pytest.yml/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/amrukwa/saucie/badge)](https://www.codefactor.io/repository/github/amrukwa/saucie)
# SAUCIE
An implementation of SAUCIE (Sparse Autoencoder for Clustering, Imputing, and Embedding) in Tensorflow.
***
This repository is a fork of the [original SAUCIE implementation](https://github.com/KrishnaswamyLab/SAUCIE). The authors let me use it for my Bachelor thesis and bump the used tensorflow version to 2, as well as build a web application using the model architecture.  
The code used for measuring the goodness of the dimensionality reduction can be found [here](https://github.com/pachterlab/CBP_2021).


## Usage
### On remote server
If you don't want to download and install anything, the deployed application is available [here](http://dssoftware.aei.polsl.pl/saucie/SAUCIE/).
### Docker
The other recommended way to use this software is through Docker. This is the second most convenient way, if you want to use saucie application.  
To install latest stable version use:  
```
docker pull amrukwa/saucie
``` 

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
More info about the parameters of the estimators can be found in the `saucie/saucie.py` file.