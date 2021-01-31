# Background
Machine Learning - Predicting how participants perform exercises

There are many devices that record key fitness performance metrics making it possible to collect a large amount of data about personal activity. Usually the goal is to measure regularly so to improve fitness, health and find new patterns in behaviour.
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. This project will use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of this project is to predict, using Trees, Boosting and Random Forest, the manner in which they did the exercises.

# Summary
This project constructs and tests various prediction models to predict the manner or how well participants perform barbell lifts (correctly or incorrectly). The ‘classe’ variable in the training set predicts the manner in which they did the exercise and I use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants to make predictions. The project identifies the most relevant features and applies a model-based approach to detect mistakes in exercise techniques. Data cleaning processing and analysis is carried out on the datasets and cross validation techniques applied. In this exercise I build and test various prediction models including Trees, Boosting and Random Forest. The Out of Sample Error is then calculated using the most accurate model, Random Forest (accuracy of 0.994), and this model is applied to a set of twenty different independent test cases.

More information is available from the Pontifical Catholic University of Rio de Janeiro website: puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

I would like to thank PUC Rio for their generosity in providing access to their Human Activity Recognition datasets.

Exercise Techniques:

A Execution of exercise according to specification.

B Throwing elbows to the front.

C Lifting dumbbell only half way.

D Lowering the dumbbell only halfway.

E Throwing the hips to the front.
