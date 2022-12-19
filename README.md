# Neural_Network_Charity_Analysis

## Overview of the analysis

Beks is a data scientist for non profit foundation Alphabet Soup, which is dedicated to help organizations that protect the environment and improve overall people well being. Her job is to analyze the impact of each donation and vet potential recipients. This helps ensure that the foundation’s money is being used effectively. Because every donation the company makes is not impactful, the project aims to help Beks **predict which organizations are worth donating to and which are high risk**. Since this problem is too complex for the statistical and machine learning models used, deep learning neural network will be designed and trained. The datasource is a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. The below tasks will be achieved:
- Explore and implement neural networks using the TensorFlow platform in Python.
- Preprocess Data for a Neural Network Model.
- Create a binary classification model that can predict if a funded organization will be successful based on the features in the dataset.
- Compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.
- Optimize the Model in order to achieve a target predictive accuracy higher than 75%.
- Store and retrieve trained models for more robust uses.

## Results

### Data Preprocessing

- The dataset has 34,299 organizations that have received funding from Alphabet Soup over the years. The columns that capture metadata about each organization in the dataset are as follows:
  - EIN and NAME—Identification columns
  - APPLICATION_TYPE—Alphabet Soup application type
  - AFFILIATION—Affiliated sector of industry
  - CLASSIFICATION—Government organization classification
  - USE_CASE—Use case for funding
  - ORGANIZATION—Organization type
  - STATUS—Active status
  - INCOME_AMT—Income classification
  - SPECIAL_CONSIDERATIONS—Special consideration for application
  - ASK_AMT—Funding amount requested
  - IS_SUCCESSFUL—Was the money used effectively  
- Among the columns of the dataset, **IS_SUCCESSFUL is the target variable** for the model. It contains the binary data that tells if the organizaton used the charity donation effectively or not.
- The **pre-processing started with dropping EIN and NAME columns because they were considered not beneficial (as targets or features)** in predicting the outcomes of the model. 
- APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS and ASK_AMT are considered as features before optimizing the model. 
- After the number of unique values for each column are determined, those columns that have more than 10 unique values are considered. The number of data points for each unique value in the column are determined and density plots are drawn to find distribution of values.
  - Any APPLICATION_TYPE that appears fewer than 500 times in the dataset is binned as "other".
  - Any CLASSIFICATION that appears fewer than 1800 times in the dataset is binned as "other."
  <img src="images/original_bins.png" width="500"/>

- The categorical variables are encoded using one-hot encoding and merged into the original dataframe.
<img src="images/original_preprocessed.png" width="500"/>

- Finally, the dataset is separated in the appropriate "y" target and "X" features as well as split into training and testing sets accordingly.
- The feature data is scaled so that this normalization prevents variations in the magnitudinal scaling between columns.

### Compiling, Training, and Evaluating the Model

- How many neurons, layers, and activation functions did you select for your neural network model, and why?
- Were you able to achieve the target model performance?
- What steps did you take to try and increase model performance?


## Summary

- Summarize the overall results of the deep learning model. 
- Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
