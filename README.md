# Kaggle CareerCon 2019: Ensemble Prediction for Robot Driving Surfaces
This project serves as my entry into Kaggle's yearly CareerCon machine learning competition for 2019 (https://www.kaggle.com/c/career-con-2019/overview), where orientation, velocity, and acceleration data is provided for robots maneuvering along a set of 9 different types of surfaces. 

To predict the surface that a robot is located on, I use an ensemble method, with SMOTE resampling to balance the surface classes within the training data, within the included Jupyter notebook, with helper functions called from the supporting Python file.

The raw data is simply copied from the publicly available competition datasets and included for convenience. The CSV files preceded with 'final_submission' are my own, automatically generated from various iterations of the models used in development.

