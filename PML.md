Executive Summary
-----------------

With the advent devices such as Jawbone Up, Nike FuelBand, and Fitbit,
an increasing number of fitness enthusiasts are beginning to track and
monitor their activity levels, with very little bother regarding the
quality of the same. This assignment seeks to better understand,
quantify and predict the performance of a particular physical activity
an individual engages in. The assignment begins with data extraction,
storing, cleaning and separation. Soon after, various predictive models
were formulated using the training data set. Further on, the accuracies
of each of these methods were projected using the test set. From the
models created, the one with the highest accuracy was chosen to predict
how well were the activities performed for 20 different cases within the
testing dataset.

Introduction
------------

### Background

With devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively.These type of devices are part of the
quantified self-movement – a group of enthusiasts who take measurements
of themselves regularly to improve their health, to find patterns in
their behaviour, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it.n this project, your goal will
be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har>

### Data

The data has been made available through the following links: <br> 1.
Training Data Set -
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>
<br> 2. Testing Data Set -
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The Weight Lifting Exercises dataset for this project was obtained from
this source: <http://groupware.les.inf.puc-rio.br/har>. <br> The
reference for the research paper from where the data was extracted is as
follows, Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
Qualitative Activity Recognition of Weight Lifting Exercises.
Proceedings of 4th International Conference in Cooperation with SIGCHI
(Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

### Aim

The aim of the assignment is to design prediction models based on the
available dataset to project the quality of barbell lifts performed by
the participants. Once various prediction models are realised, the most
accurate one is chosen to predict the quality of lifts by participants
for 20 different cases. The results from the prediction are then
utilised to answer a quiz in order to validate the prediction model

### Libraries

The list of libraries that we will be taping into are as follows:

    library(ggplot2)
    library(caret)
    library(kernlab)
    library(MASS)
    library(rpart)
    library(randomForest)

Data Preparation
----------------

### Data Reading

Using the aforementioned links the data is read into the training and
testing variables.

    training<- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                        header = TRUE, 
                        sep = ",", 
                        na.strings = c("NA","#DIV/0!",""))

    testing<- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                       header = TRUE, 
                       sep = ",", 
                       na.strings = c("NA","#DIV/0!",""))

### Data Separation

The training data set is now split into a cross validation set and a
training set, also the testing set is further stored in the variable
test.

    inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
    my_train<- training[inTrain, ]
    my_test<- training[-inTrain, ]
    test<- testing

### Data Cleaning and Tidying

The first part of data cleaning and tidying is the removal of near zero
variance predictors from the data set. These are variables with a rather
constant value and low influence on the outcome of the dataset

    near_zero_train<- nearZeroVar(training)
    near_zero_test<- nearZeroVar(testing)
    my_train<- my_train[,-near_zero_train]
    my_test<- my_test[,-near_zero_train]
    test<- test[,-near_zero_test]

Further on, variables with more than 80% of its data points assigned to
NA is removed

    my_train <- my_train[ lapply( my_train,function(x) sum(is.na(x)) / length(x) ) < 0.2 ]
    my_test <- my_test[ lapply( my_test,function(x) sum(is.na(x)) / length(x) ) < 0.2 ]
    test <- test[ lapply( test, function(x) sum(is.na(x)) / length(x) ) < 0.2 ]

Finally the identification variables are also removed

    my_train<- my_train[,-(1:5)]
    my_test<- my_test[,-(1:5)]
    test<- test[,-c(1:5)]

The final datasets now have the following dimensions

    dim(my_train)

    ## [1] 13737    54

    dim(my_test)

    ## [1] 5885   54

    dim(test)

    ## [1] 20 54

Data cleaning and tidying have resulted in smaller datasets with the
same number of predictors.

### Principal Component Analysis (PCA)

Since we have more than 50 predictors, it would be wise to shrink the
dataset a little further before we proceed with the prediction models.
But, before we go ahead with PCA, we must verify if doing so would be
actually beneficial.

    corMatrix <- abs(cor(my_train[, -54]))
    values <- length(corMatrix)/2
    sum<- sum(corMatrix > 0.8 )
    percent<- round(sum/values *100, 1)

Only 6.5% of the variables seem to have a high correlation. This would
not result in any significant decrease in variance for increased bias.
Hence PCA will not be conducted.

Predictive Models
-----------------

### Decision Trees

The Decision Tree machine learning algorithm is known to perform best
within a non-linear setting. Additionally, the model is known to perform
well within large data sets.

    set.seed(333)
    fit_rpart<- train(classe~., method = "rpart", data = my_train)
    pred_rpart<- predict(fit_rpart, my_test)
    conf_rpart<- confusionMatrix(pred_rpart, my_test$classe)
    conf_rpart

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1478  310  169  161   36
    ##          B   32  372   32  151   77
    ##          C  161  457  825  609  297
    ##          D    0    0    0    0    0
    ##          E    3    0    0   43  672
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.5687         
    ##                  95% CI : (0.556, 0.5814)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.4485         
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8829  0.32660   0.8041   0.0000   0.6211
    ## Specificity            0.8395  0.93847   0.6864   1.0000   0.9904
    ## Pos Pred Value         0.6862  0.56024   0.3512      NaN   0.9359
    ## Neg Pred Value         0.9475  0.85309   0.9432   0.8362   0.9207
    ## Prevalence             0.2845  0.19354   0.1743   0.1638   0.1839
    ## Detection Rate         0.2511  0.06321   0.1402   0.0000   0.1142
    ## Detection Prevalence   0.3660  0.11283   0.3992   0.0000   0.1220
    ## Balanced Accuracy      0.8612  0.63254   0.7452   0.5000   0.8057

### Linear Discriminant Analysis (LDA)

The LDA method is a part of the Dimensionality Reduction machine
learning algorithm. This method is often employed when the outcome of a
dataset is dependent on a large number of predictors, as LDA helps
maximise the separability between the different classes so as to make
better predictions.

    set.seed(333)
    fit_lda<- train(classe~., method = "lda", data = my_train)
    pred_lda<- predict(fit_lda, my_test)
    conf_lda<- confusionMatrix(pred_lda, my_test$classe)
    conf_lda

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1391  189  114   54   43
    ##          B   51  715   88   35  172
    ##          C  108  150  651  113   88
    ##          D  121   45  136  725   86
    ##          E    3   40   37   37  693
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.7094         
    ##                  95% CI : (0.6976, 0.721)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.6319         
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8309   0.6277   0.6345   0.7521   0.6405
    ## Specificity            0.9050   0.9271   0.9055   0.9212   0.9756
    ## Pos Pred Value         0.7767   0.6739   0.5865   0.6514   0.8556
    ## Neg Pred Value         0.9309   0.9121   0.9215   0.9499   0.9233
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2364   0.1215   0.1106   0.1232   0.1178
    ## Detection Prevalence   0.3043   0.1803   0.1886   0.1891   0.1376
    ## Balanced Accuracy      0.8680   0.7774   0.7700   0.8366   0.8081

### Random Forest

The Random Forest method is a subset of the Ensemble machine learning
algorithm. The method constructs numerous decision trees and uses them
to create a classification and predictions consecutively.

    set.seed(333)
    fit_rf<- randomForest(classe~., data = my_train)
    pred_rf<- predict(fit_rf, my_test)
    conf_rf<- confusionMatrix(pred_rf, my_test$classe)
    conf_rf

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    4    0    0    0
    ##          B    0 1134    3    0    0
    ##          C    0    1 1023    5    0
    ##          D    0    0    0  959    1
    ##          E    0    0    0    0 1081
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9976         
    ##                  95% CI : (0.996, 0.9987)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.997          
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9956   0.9971   0.9948   0.9991
    ## Specificity            0.9991   0.9994   0.9988   0.9998   1.0000
    ## Pos Pred Value         0.9976   0.9974   0.9942   0.9990   1.0000
    ## Neg Pred Value         1.0000   0.9989   0.9994   0.9990   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1927   0.1738   0.1630   0.1837
    ## Detection Prevalence   0.2851   0.1932   0.1749   0.1631   0.1837
    ## Balanced Accuracy      0.9995   0.9975   0.9979   0.9973   0.9995

Model Selection and Prediction
------------------------------

### Model Comparison

The table below draws together a comparison between the accuracies
between the different prediction methods used.

    a<- data.frame(Prediction_Method = c("Decision Trees","Linear Discriminant Analysis","Random Forest" ), Accuracy = c("49.0%","71.1%","99.8%"))
    knitr::kable(a)

<table>
<thead>
<tr class="header">
<th align="left">Prediction_Method</th>
<th align="left">Accuracy</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left">Decision Trees</td>
<td align="left">49.0%</td>
</tr>
<tr class="even">
<td align="left">Linear Discriminant Analysis</td>
<td align="left">71.1%</td>
</tr>
<tr class="odd">
<td align="left">Random Forest</td>
<td align="left">99.8%</td>
</tr>
</tbody>
</table>

From the table above it becomes evident that with an accuracy of about
99.8% the random forest method is the most effective predictive model.

### Prediction

To predict the classes for the 20 cases in the testing data set the
random forest method will be employed.

    predict(fit_rf,test)

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
