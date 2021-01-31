---
title: "Machine Learning - Predicting how participants perform exercises"
author: "Eamon C"
date: '2021'
output:
  html_document: 
    keep_md: yes
  
---

## Summary
This project constructs and tests various prediction models to predict the manner or how well participants perform barbell lifts (correctly or incorrectly). The 'classe' variable in the training set predicts the manner in which they did the exercise and I use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants to make predictions. The project identifies the most relevant features and applies a model-based approach to detect mistakes in exercise techniques. Data cleaning processing and analysis is carried out on the datasets and cross validation techniques applied. In this exercise I build and test various prediction models including Trees, Boosting and Random Forest. The Out of Sample Error is then calculated using the most accurate model, Random Forest (accuracy of 0.994), and this model is applied to a set of twenty different independent test cases.

More information is available from the Pontifical Catholic University of Rio de Janeiro website:
[puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

I would like to thank PUC Rio for their generosity in providing access to their Human Activity Recognition datasets.

Exercise Techniques:

A Execution of exercise according to specification.

B Throwing elbows to the front.

C Lifting dumbbell only half way.

D Lowering the dumbbell only halfway.

E Throwing the hips to the front.

## Loading & Data Processing


```r
library(plyr);library(dplyr);library(caret)
library(ggplot2); library(rattle)
```


```r
#if (!file.exists('data')) {
#      dir.create('data')
#}
#fileUrl <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
#download.file(fileUrl, destfile = './data/pml_training.csv', method='curl')
#fileUrl <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
#download.file(fileUrl, destfile = './data/pml_testing.csv', method='curl')
list.files('./data')
```

```
## [1] "pml_testing.csv"  "pml_training.csv"
```

```r
#datadownloaded <- date()
```



```r
pml_training <- read.csv('./data/pml_training.csv',na.strings=c('#DIV/0!', 'NA'))
pml_testing <- read.csv('./data/pml_testing.csv', na.strings=c('#DIV/0!', 'NA'))
```

## Exploratory Data Analysis
Check the dataset's dimensions, variables and characteristics.

```r
dim(pml_training);dim(pml_testing);
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

```r
table(pml_training$user_name)
```

```
## 
##   adelmo carlitos  charles   eurico   jeremy    pedro 
##     3892     3112     3536     3070     3402     2610
```

```r
tbl_df(pml_training[,8:50])
```

```
## # A tibble: 19,622 x 43
##    roll_belt pitch_belt yaw_belt total_accel_belt kurtosis_roll_belt
##        <dbl>      <dbl>    <dbl>            <int>              <dbl>
##  1      1.41       8.07    -94.4                3                 NA
##  2      1.41       8.07    -94.4                3                 NA
##  3      1.42       8.07    -94.4                3                 NA
##  4      1.48       8.05    -94.4                3                 NA
##  5      1.48       8.07    -94.4                3                 NA
##  6      1.45       8.06    -94.4                3                 NA
##  7      1.42       8.09    -94.4                3                 NA
##  8      1.42       8.13    -94.4                3                 NA
##  9      1.43       8.16    -94.4                3                 NA
## 10      1.45       8.17    -94.4                3                 NA
## # ... with 19,612 more rows, and 38 more variables: kurtosis_picth_belt <dbl>,
## #   kurtosis_yaw_belt <lgl>, skewness_roll_belt <dbl>,
## #   skewness_roll_belt.1 <dbl>, skewness_yaw_belt <lgl>, max_roll_belt <dbl>,
## #   max_picth_belt <int>, max_yaw_belt <dbl>, min_roll_belt <dbl>,
## #   min_pitch_belt <int>, min_yaw_belt <dbl>, amplitude_roll_belt <dbl>,
## #   amplitude_pitch_belt <int>, amplitude_yaw_belt <dbl>,
## #   var_total_accel_belt <dbl>, avg_roll_belt <dbl>, stddev_roll_belt <dbl>,
## #   var_roll_belt <dbl>, avg_pitch_belt <dbl>, stddev_pitch_belt <dbl>,
## #   var_pitch_belt <dbl>, avg_yaw_belt <dbl>, stddev_yaw_belt <dbl>,
## #   var_yaw_belt <dbl>, gyros_belt_x <dbl>, gyros_belt_y <dbl>,
## #   gyros_belt_z <dbl>, accel_belt_x <int>, accel_belt_y <int>,
## #   accel_belt_z <int>, magnet_belt_x <int>, magnet_belt_y <int>,
## #   magnet_belt_z <int>, roll_arm <dbl>, pitch_arm <dbl>, yaw_arm <dbl>,
## #   total_accel_arm <int>, var_accel_arm <dbl>
```

```r
summary(pml_training[,8:20])
```

```
##    roll_belt        pitch_belt          yaw_belt       total_accel_belt
##  Min.   :-28.90   Min.   :-55.8000   Min.   :-180.00   Min.   : 0.00   
##  1st Qu.:  1.10   1st Qu.:  1.7600   1st Qu.: -88.30   1st Qu.: 3.00   
##  Median :113.00   Median :  5.2800   Median : -13.00   Median :17.00   
##  Mean   : 64.41   Mean   :  0.3053   Mean   : -11.21   Mean   :11.31   
##  3rd Qu.:123.00   3rd Qu.: 14.9000   3rd Qu.:  12.90   3rd Qu.:18.00   
##  Max.   :162.00   Max.   : 60.3000   Max.   : 179.00   Max.   :29.00   
##                                                                        
##  kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
##  Min.   :-2.121     Min.   :-2.190      Mode:logical      Min.   :-5.745    
##  1st Qu.:-1.329     1st Qu.:-1.107      NA's:19622        1st Qu.:-0.444    
##  Median :-0.899     Median :-0.151                        Median : 0.000    
##  Mean   :-0.220     Mean   : 4.334                        Mean   :-0.026    
##  3rd Qu.:-0.219     3rd Qu.: 3.178                        3rd Qu.: 0.417    
##  Max.   :33.000     Max.   :58.000                        Max.   : 3.595    
##  NA's   :19226      NA's   :19248                         NA's   :19225     
##  skewness_roll_belt.1 skewness_yaw_belt max_roll_belt     max_picth_belt 
##  Min.   :-7.616       Mode:logical      Min.   :-94.300   Min.   : 3.00  
##  1st Qu.:-1.114       NA's:19622        1st Qu.:-88.000   1st Qu.: 5.00  
##  Median :-0.068                         Median : -5.100   Median :18.00  
##  Mean   :-0.296                         Mean   : -6.667   Mean   :12.92  
##  3rd Qu.: 0.661                         3rd Qu.: 18.500   3rd Qu.:19.00  
##  Max.   : 7.348                         Max.   :180.000   Max.   :30.00  
##  NA's   :19248                          NA's   :19216     NA's   :19216  
##   max_yaw_belt  
##  Min.   :-2.10  
##  1st Qu.:-1.30  
##  Median :-0.90  
##  Mean   :-0.22  
##  3rd Qu.:-0.20  
##  Max.   :33.00  
##  NA's   :19226
```

```r
sum(is.na(pml_training$max_roll_belt)) #example of NAs
```

```
## [1] 19216
```
The data is multi-dimensional / multi-class and it is difficult to get a picture of what's 
going on. Reading the data into R initially we see some measurements with mislabelled entries (#DIV/0!). 
Some 53 variables have a large numbers of NAs (see 'max_roll_belt' example above), usually 19216+ out of a total of 19622 observations.Although the values in these columns appear genuine they do appear to be summary values rather than measurements.
We will remove the mislabelled entries when reading the data into R and also the variables with a large number of NAs.
We will further reduce the columns by removing time related and non-accelerometer measurement labels.
These changes will also be applied to the pml_testing dataset. 


```r
pml_training <- pml_training[, -(1:7)]
withNA <- colSums(is.na(pml_training)) ==0
namesWithNA <- names(pml_training[,withNA])
pml_training <- pml_training[, namesWithNA]

pml_testing <- pml_testing[, -(1:7)]
withNA <- colSums(is.na(pml_testing)) ==0
namesWithNA <- names(pml_testing[,withNA])
pml_testing <- pml_testing[, namesWithNA]

dim(pml_training);dim(pml_testing);
```

```
## [1] 19622    53
```

```
## [1] 20 53
```

We can also check and ensure there are no Zero Covariates (variables with no variability). In this case there are none.

```r
nsv <- nearZeroVar(pml_training, saveMetrics=TRUE)
head(nsv)
```

```
##                  freqRatio percentUnique zeroVar   nzv
## roll_belt         1.101904     6.7781062   FALSE FALSE
## pitch_belt        1.036082     9.3772296   FALSE FALSE
## yaw_belt          1.058480     9.9734991   FALSE FALSE
## total_accel_belt  1.063160     0.1477933   FALSE FALSE
## gyros_belt_x      1.058651     0.7134849   FALSE FALSE
## gyros_belt_y      1.144000     0.3516461   FALSE FALSE
```

```r
#cont ...
```

We identify Variables that are highly correlated using the 'highlycorrelated' function.

```r
highlycorrelated <- function(dataframe,numToReport)
{
      # find the correlations in dataset
      cormatrix <- cor(dataframe[sapply(dataframe, is.numeric)])
      # set correlations on the diagonal or lower triangle to 0,
      diag(cormatrix) <- 0
      cormatrix[lower.tri(cormatrix)] <- 0
      # convert matrix to a dataframe
      df <- as.data.frame(as.table(cormatrix))
      names(df) <- c("1st Variable", "2nd Variable","Correlation")
      # sort correlations from highest to lowest
      head(df[order(abs(df$Correlation),decreasing=T),],n=numToReport)
}
highlycorrelated(pml_training, 12)
```

```
##          1st Variable     2nd Variable Correlation
## 469         roll_belt     accel_belt_z  -0.9920085
## 157         roll_belt total_accel_belt   0.9809241
## 1695 gyros_dumbbell_x gyros_dumbbell_z  -0.9789507
## 472  total_accel_belt     accel_belt_z  -0.9749317
## 366        pitch_belt     accel_belt_x  -0.9657334
## 477      accel_belt_y     accel_belt_z  -0.9333854
## 2373 gyros_dumbbell_z  gyros_forearm_z   0.9330422
## 420  total_accel_belt     accel_belt_y   0.9278069
## 417         roll_belt     accel_belt_y   0.9248983
## 954       gyros_arm_x      gyros_arm_y  -0.9181821
## 2371 gyros_dumbbell_x  gyros_forearm_z  -0.9144764
## 528      accel_belt_x    magnet_belt_x   0.8920913
```


We remove correlated variables from the datasets.

```r
cormatrix <- cor(pml_training[sapply(pml_training, is.numeric)])
corrVars <- findCorrelation(cormatrix, cutoff =0.90)
pml_training <- pml_training[,-corrVars]
pml_testing <- pml_testing[,-corrVars] #pml_testing dataset

dim(pml_training); dim(pml_testing)
```

```
## [1] 19622    46
```

```
## [1] 20 46
```

## Cross Validation
The pml_training dataset will be split into training and test sets. The training and test datasets
will be used to create and then test the prediction model respectively. The chosen model will then be 
employed to predict 20 different test cases (pml_testing dataset).


```r
set.seed(123)
inTrain <- createDataPartition(y=pml_training$classe, p=0.75, list=FALSE)
training <- pml_training[inTrain,]
testing <- pml_training[-inTrain,]
dim(training); dim(testing)
```

```
## [1] 14718    46
```

```
## [1] 4904   46
```

## Predict with Trees
#### Build the Trees Prediction Model

```r
modFitTree <- train(classe ~., method ='rpart', data=training)
print(modFitTree$finalModel)
```

```
## n= 14718 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 14718 10533 A (0.28 0.19 0.17 0.16 0.18)  
##    2) pitch_forearm< -26.65 1325    60 A (0.95 0.045 0 0 0) *
##    3) pitch_forearm>=-26.65 13393 10473 A (0.22 0.21 0.19 0.18 0.2)  
##      6) magnet_belt_y>=555.5 12292  9375 A (0.24 0.23 0.21 0.18 0.15)  
##       12) magnet_dumbbell_y< 426.5 10075  7243 A (0.28 0.18 0.25 0.17 0.12)  
##         24) roll_forearm< 121.5 6294  3749 A (0.4 0.18 0.18 0.15 0.088) *
##         25) roll_forearm>=121.5 3781  2465 C (0.076 0.17 0.35 0.22 0.18) *
##       13) magnet_dumbbell_y>=426.5 2217  1212 B (0.038 0.45 0.044 0.21 0.26)  
##         26) total_accel_dumbbell>=5.5 1559   646 B (0.055 0.59 0.06 0.028 0.27) *
##         27) total_accel_dumbbell< 5.5 658   238 D (0 0.14 0.0046 0.64 0.22) *
##      7) magnet_belt_y< 555.5 1101   212 E (0.0027 0.0018 0.00091 0.19 0.81) *
```

```r
fancyRpartPlot(modFitTree$finalModel)
```

![](index_files/figure-html/trees-1.png)<!-- -->

#### Predict the test dataset using the Trees Model


```r
predTree <- predict(modFitTree, newdata=testing)
cmTree <- confusionMatrix(predTree, testing$classe)
print(cmTree)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1263  389  411  307  201
##          B   31  293   32   17  123
##          C  101  241  411  285  231
##          D    0   24    0  135   45
##          E    0    2    1   60  301
## 
## Overall Statistics
##                                           
##                Accuracy : 0.49            
##                  95% CI : (0.4759, 0.5041)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.334           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9054  0.30875  0.48070  0.16791  0.33407
## Specificity            0.6272  0.94867  0.78810  0.98317  0.98426
## Pos Pred Value         0.4912  0.59073  0.32388  0.66176  0.82692
## Neg Pred Value         0.9434  0.85118  0.87785  0.85766  0.86784
## Prevalence             0.2845  0.19352  0.17435  0.16395  0.18373
## Detection Rate         0.2575  0.05975  0.08381  0.02753  0.06138
## Detection Prevalence   0.5243  0.10114  0.25877  0.04160  0.07423
## Balanced Accuracy      0.7663  0.62871  0.63440  0.57554  0.65917
```
As we can see that this model has a pretty poor accuracy at just 0.49.

## Predict with Random Forest
#### Build the RF Prediction Model

Note: trainControl method parameter is set to 'cv' rather than bootstrapping, which is the default.


```r
trControl <- trainControl(method='cv', number=5, allowParallel=TRUE, verbose=F)
modFitRF <- train(classe~., data=training, method='rf', trControl=trControl)
```

```r
print(modFitRF)
```

```
## Random Forest 
## 
## 14718 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11775, 11774, 11773, 11776, 11774 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9902156  0.9876223
##   23    0.9902159  0.9876236
##   45    0.9842379  0.9800596
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 23.
```

```r
print(modFitRF$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 23
## 
##         OOB estimate of  error rate: 0.69%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4179    4    1    0    1 0.001433692
## B   19 2816   12    0    1 0.011235955
## C    0   17 2541    9    0 0.010128555
## D    0    0   25 2386    1 0.010779436
## E    0    0    4    7 2695 0.004065041
```

#### Predict the test dataset using the RF Model

```r
predRF <- predict(modFitRF, newdata=testing)
cmRF <- confusionMatrix(predRF, testing$classe)
cmRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    4    0    0    0
##          B    1  944    6    0    0
##          C    0    1  844    9    0
##          D    0    0    5  793    1
##          E    0    0    0    2  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9941         
##                  95% CI : (0.9915, 0.996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9925         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9947   0.9871   0.9863   0.9989
## Specificity            0.9989   0.9982   0.9975   0.9985   0.9995
## Pos Pred Value         0.9971   0.9926   0.9883   0.9925   0.9978
## Neg Pred Value         0.9997   0.9987   0.9973   0.9973   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1925   0.1721   0.1617   0.1835
## Detection Prevalence   0.2851   0.1939   0.1741   0.1629   0.1839
## Balanced Accuracy      0.9991   0.9965   0.9923   0.9924   0.9992
```
This particular model shows a very high accuarcy.

## Predict with Boosting
#### Build the Prediction Model


```r
trControl <- trainControl(method='cv', number=10, allowParallel=TRUE, verbose=F)
modFitGBM <- train(classe~., data=training, method='gbm', trControl=trControl)
```


```r
print(modFitGBM)
```

```
## Stochastic Gradient Boosting 
## 
## 14718 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 13246, 13246, 13246, 13246, 13246, 13246, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7404537  0.6709867
##   1                  100      0.8145795  0.7652919
##   1                  150      0.8469887  0.8063268
##   2                   50      0.8548033  0.8160711
##   2                  100      0.9065761  0.8817718
##   2                  150      0.9283860  0.9093859
##   3                   50      0.8959079  0.8682397
##   3                  100      0.9410927  0.9254592
##   3                  150      0.9582138  0.9471358
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
print(modFitGBM$finalModel)
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 45 predictors of which 40 had non-zero influence.
```

#### Predict the test dataset using the GBM Model

```r
predGBM <- predict(modFitGBM, newdata=testing)
cmGBM <- confusionMatrix(predGBM, testing$classe)
cmGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1372   29    0    0    5
##          B   11  891   33    5    8
##          C    9   26  809   28    8
##          D    3    1   11  759   11
##          E    0    2    2   12  869
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9584          
##                  95% CI : (0.9524, 0.9638)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9474          
##  Mcnemar's Test P-Value : 4.485e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9835   0.9389   0.9462   0.9440   0.9645
## Specificity            0.9903   0.9856   0.9825   0.9937   0.9960
## Pos Pred Value         0.9758   0.9399   0.9193   0.9669   0.9819
## Neg Pred Value         0.9934   0.9853   0.9886   0.9891   0.9920
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2798   0.1817   0.1650   0.1548   0.1772
## Detection Prevalence   0.2867   0.1933   0.1794   0.1601   0.1805
## Balanced Accuracy      0.9869   0.9622   0.9643   0.9688   0.9802
```
This model also has a high accuracy but not as high as the Random Forest Model.

## Accuracy Results for all Models 

```r
results <- c(cmTree$overall[1], cmRF$overall[1], cmGBM$overall[1])
names(results) <- c('Trees','RF', 'GBM')
print(results)
```

```
##     Trees        RF       GBM 
## 0.4900082 0.9940865 0.9584013
```
Considering all three models we can say that the Random Forest model has the highest accuracy.
Therefore the Random forest model will be used to predict the 20 different test cases.

## Predicting the Out of Sample Error
#### Predict the pml_testing dataset

```r
prediction <- predict(modFitRF, newdata=pml_testing)
#print(prediction)

predResults <- data.frame(test_case = pml_testing$problem_id, Prediction = prediction)
print(predResults)
```

```
##    test_case Prediction
## 1          1          B
## 2          2          A
## 3          3          B
## 4          4          A
## 5          5          A
## 6          6          E
## 7          7          D
## 8          8          B
## 9          9          A
## 10        10          A
## 11        11          B
## 12        12          C
## 13        13          B
## 14        14          A
## 15        15          E
## 16        16          E
## 17        17          A
## 18        18          B
## 19        19          B
## 20        20          B
```
