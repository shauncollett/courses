---
title: "Not how much, but how well do people exercise?"
author: "Shaun Collett"
date: "April 16, 2015"
output: html_document
---

### Executive Summary

Using an accelerometers dataset collected while 6 people where asked to perform
correct and incorrect dumbell lifts, we'll perform an analysis to see how WELL people 
exercise.  Using the `classe` variable as the outcome, we'll discover what prediction
model best predicts the outcome, then perform cross-validation to prove our model is a good fit.

To start, we'll load all packages needed for our analysis, then download the training 
and test data sets.


```r
library(RCurl)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
# Load Data
training <- read.csv(
    text = getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),
    stringsAsFactors = FALSE)

testing <- read.csv(
    text = getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"),
    stringsAsFactors = FALSE)
```



### Data Cleaning

Before attempting to model, we first need to clean the data.  Based on exploratory analysis we've
decided to drop seven variables from the data set, since they're focused on metadata and should
not be used for prediction.  Next, we remove columns with NAs and others with blank values that
would create problems during modeling.


```r
# Create training and preliminary test sets.
training$classe <- as.factor(training$classe)

# Drop 1st variable, ID, so it doesn't impact the prediction algorithm.
drop <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2",
          "cvtd_timestamp","new_window","num_window")
trainingcut <- training[, !(names(training) %in% drop)]

# Many columns have large NA values, so drop them.  They do not make good predictors.
l <- sapply(trainingcut,function(x)sum(is.na(x)))
trainingcut <- trainingcut[ ,names(l[l == 0])]

# Drop columns that import with many blanks values - as it happens, these are
#   all character columns.
m <- sapply(trainingcut,function(x)class(x))
trainingcut <- trainingcut[, names(m[m != "character"])]
```

Once we've cleaned our data, we'll create a master training set and preliminary test set.  Finally
we'll make sure our final test set conforms to the training sets in terms of variables and data types.


```r
# Split training set into a master training set and preliminary test set.
inTrain <- createDataPartition(y=trainingcut$classe, p=0.7, list=FALSE)
trainmaster <- trainingcut[inTrain, ]
traintest <- trainingcut[-inTrain, ]

# We need to make sure our test dataset is exactly the same, in terms of columns
#   and data types, as our training.
testtrim <- testing[colnames(trainmaster[, -53])]
for (i in 1:length(testtrim) ) {
        for(j in 1:length(trainmaster)) {
        if( length( grep(names(trainmaster[i]), names(testtrim)[j]) ) ==1)  {
            class(testtrim[j]) <- class(trainmaster[i])
        }      
    }      
}
# Add random row to testtrim to be sure coercion worked, then remove.
testtrim <- rbind(trainmaster[2, -53] , testtrim)
testtrim <- testtrim[-1,]
```



### RPART Training Model

First, let's start by creating a RPART model to see how accurate it is.


```r
set.seed(12345)
fitRP <- rpart(classe ~ ., data=trainmaster, method="class")
predRP <- predict(fitRP, traintest, type="class")
cfRP <- confusionMatrix(predRP, traintest$classe)
cfRP
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1517  176   54  115   25
##          B   56  724   96   89   98
##          C   39   95  790  149  129
##          D   33   94   62  530   56
##          E   29   50   24   81  774
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7366          
##                  95% CI : (0.7252, 0.7478)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6652          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9062   0.6356   0.7700  0.54979   0.7153
## Specificity            0.9121   0.9286   0.9152  0.95021   0.9617
## Pos Pred Value         0.8039   0.6811   0.6572  0.68387   0.8079
## Neg Pred Value         0.9607   0.9139   0.9496  0.91507   0.9375
## Prevalence             0.2845   0.1935   0.1743  0.16381   0.1839
## Detection Rate         0.2578   0.1230   0.1342  0.09006   0.1315
## Detection Prevalence   0.3206   0.1806   0.2042  0.13169   0.1628
## Balanced Accuracy      0.9092   0.7821   0.8426  0.75000   0.8385
```

We can see this model is not accurate at all, with only 73.7% accuracy.



### Random Forest Training Model

We know from our lectures that Random Forest models are significantly more accurate than other models.  Let's
see if that's the case.


```r
fitRF <- randomForest(classe ~ ., data=trainmaster)
predRF <- predict(fitRF, traintest, type="class")
cfRF <- confusionMatrix(predRF, traintest$classe)
cfRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    4    0    0    0
##          B    1 1134    5    0    0
##          C    0    1 1020    6    1
##          D    0    0    1  958    2
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9964          
##                  95% CI : (0.9946, 0.9978)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9955          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9956   0.9942   0.9938   0.9972
## Specificity            0.9991   0.9987   0.9984   0.9994   1.0000
## Pos Pred Value         0.9976   0.9947   0.9922   0.9969   1.0000
## Neg Pred Value         0.9998   0.9989   0.9988   0.9988   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1927   0.1733   0.1628   0.1833
## Detection Prevalence   0.2850   0.1937   0.1747   0.1633   0.1833
## Balanced Accuracy      0.9992   0.9972   0.9963   0.9966   0.9986
```

Indeed that's the case, with an amazing 99.6% accuracy.



### Out of Sample Error

Before our prediction, let's check the out of sample error.


```r
correctpredictions <- 0
for(i in 1:5) {
    correctpredictions <- correctpredictions + sum(fitRF$confusion[i,i])
}
outOfSampleError <- 1 - (correctpredictions / sum(fitRF$confusion[,-6]))
outOfSampleError
```

```
## [1] 0.005314115
```

Just a 0.5% out of sample error.



### Prediction

Finally, we'll predict our final values for submission based on the test dataset that
we've trimmed down to mirror our training sets for accuracy in prediction.


```r
finalAnswers <- predict(fitRF, testtrim, type="class");
```

This is code remaining from prediction submission.


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

setwd("~/Box Sync/Play/datasciencecoursera/Practical Machine Learning/Course Project/Final Submission")
pml_write_files(finalAnswers)
```
