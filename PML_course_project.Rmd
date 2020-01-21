---
title: "Practical Machine Learning Course Project"
author: "Braden Glover"
date: "January 17, 2020"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Goal of this Assignment
The goal of this project is to make predictions on the class of an exercise based on data gathered by personal devices such as Fitbit, Nike FuelBand, and Jawbone up. This data includes measurements from accelerometers on the belt, forearm, arm, and dumbell of six subjects. More information on the data used in this exercise can be found [here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har). 

## Libraries Used and Setting the Seed
Below are the libraries and seed that are used in this study. 
```{r}
suppressMessages(suppressWarnings(library(data.table)))
suppressMessages(suppressWarnings(library(caret)))
suppressMessages(suppressWarnings(library(xgboost)))

set.seed(666)
```
We should also go ahead and set our working directory for this project:
```{r}
setwd("~/ml_proj")
```

## Retrieve the Data
The training and test data sets are found at the following urls: 
```{r}
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```
Next, we will download these files to our working directory. 
```{r}
train_csv <- "training.csv"
test_csv <- "test.csv"

if (!file.exists(train_csv)){
  download.file(train_url, destfile = train_csv, method = "curl")
}
if (!file.exists(test_csv)){
  download.file(test_url, destfile = test_csv, method = "curl")
}
```
Now that the data is downloaded, we can load it into R as a data.table. 
```{r}
training <- fread(train_csv)
test <- fread(test_csv)
dim(training)
dim(test)
```

## Create a Validation Set
You will likely have noticed when looking at the dimensions of the training and test sets, that the test set is very small at only 20 data points. This is because the training set provided is associated with the quiz for the John Hopkins University Data Science Coursera Course. Because of this limitation, we will also create a validation set to get an idea of the out of sample error for the models we build. 

```{r}
validation_points <- createDataPartition(y = training$classe, 
                                  p = 0.2, 
                                  list = FALSE)
validation <- training[validation_points]
training <- training[-validation_points]
```

## Data Cleaning
Upon a brief observation of the training data, it is clear that there are many fields that either have near zero variance or contain almost entirely `NA` values. Here we will attempt to account for both of these problems.
  
First will will remove fields with near zero variance:
```{r}
nzv <- nearZeroVar(training)
nzv <- names(training)[nzv]

training <- training[, (nzv) := NULL]
test <- test[, (nzv) := NULL]
validation <- validation[, (nzv) := NULL]
```
  
Next, we will remove columns whose values are at least 90% `NA` values: 
```{r}
high_nas <- apply(training, 2, function(x) mean(is.na(x))) > 0.9
high_nas <- names(training)[high_nas]

training <- training[, (high_nas) := NULL]
test <- test[, (high_nas) := NULL]
validation <- validation[, (high_nas) := NULL]
```
The first 5 fields in this data is all indentifying information for the 6 subjects of the test. This information is unlikely to add much to the predictive power of our models, so we will remove them as well. 
```{r}
id_cols <- 1:5
id_cols <- names(training)[id_cols]

training <- training[, (id_cols) := NULL]
test <- test[, (id_cols) := NULL]
validation <- validation[, (id_cols) := NULL]
```
We can now look at the dimensions for our data sets and see that we have whittled the number of fields to predict on from **160** to **54** fields. 
```{r}
dim(training)
```

## Data Modelling
For this prediction assignment, we will model the data using the XGBoost algorithm. 

### XGBoost
The XGBoost alogorithm has a few requirements for how the data needs to be structured. The first of these is that the label (in this case, the `classe` variable) needs to be of the integer class. 
```{r}
xg_trainSet <- copy(training)
xg_validSet <- copy(validation)
xg_testSet <- copy(test)


label_train = as.integer(factor(xg_trainSet$classe)) - 1 
xg_trainSet[, classe := NULL]

label_validation = as.integer(factor(xg_validSet$classe)) - 1
xg_validSet[, classe := NULL]
```
Next, we need to create training, validation, and test matrices. 
```{r}
train_data = as.matrix(xg_trainSet)
valid_data = as.matrix(xg_validSet)
xg_testSet[, problem_id := NULL]
test_data = as.matrix(xg_testSet)
```
The next step is to create the xgb.DMatrix objects and to define the main parameters for the XGBoost algorithm. 
```{r}
xgb_train = xgb.DMatrix(data = train_data, label = label_train)
xgb_validation = xgb.DMatrix(data = valid_data, label = label_validation)
xgb_test = xgb.DMatrix(data = test_data)

parameters = list(
  booster = "gbtree", 
  objective = "multi:softmax",
  num_class = 5
)
```

```{r}
model_one <- xgb.train(
  params = parameters, 
  data = xgb_train, 
  nrounds = 1000,
  early_stopping_rounds = 10, 
  watchlist = list(val = xgb_train, watch = xgb_validation), 
  verbose = 1
)
model_one
```
Now that we have our model, we can estimate its performace on our validation data. Since the XGBoost algorithm requires the labels to be integers, it is important to note that:  
 - classe A = 0  
 - classe B = 1  
 - classe C = 2  
 - classe D = 3  
 - classe E = 4  
```{r}
xgb_preds = predict(model_one, valid_data, reshape = T)
fact_xgb_preds <- factor(xgb_preds)
fact_validation_cases <- factor(as.integer(factor(validation$classe)) - 1)
confusionMatrix(fact_validation_cases, fact_xgb_preds)
```
The results from this confusion matrix, it appears that this model is accurate and will do a good job of predicting the exercise classe outside of its training set. 

## Predicting "Classe" in the Test Set
We can now apply our XGBoost model to the test set to get our answers for this assignment.  
```{r}
a <- predict(model_one, xgb_test)
a <- as.character(a)
a <- chartr("01234", "ABCDE", a)

ans <- data.table(Q = 1:20, ans = a)
ans
```
