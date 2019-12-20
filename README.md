# Wine Point Prediction

Docker application that trains and evaluates a model that predicts wine quality from general descriptions. 

Original dataset: https://www.kaggle.com/zynicide/wine-reviews.

Intended as a template for training applications.

To build images execute with version set:

`./build-task-images.sh VERSION`

To run execute: 

`docker-compose up orchestrator` 

To shut down:

`./docker-clean.sh`

Output of task is found in ./data_root/evaluation/
