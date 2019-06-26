DATA 612 Project 3
================

Matrix Factorization Methods
============================

The goal of this assignment is give you practice working with Matrix Factorization techniques.

Your task is implement a matrix factorization method—such as singular value decomposition (SVD) or Alternating Least Squares (ALS)—in the context of a recommender system.

You may approach this assignment in a number of ways. You are welcome to start with an existing recommender system written by yourself or someone else. Remember as always to cite your sources, so that you can be graded on what you added, not what you found. SVD can be thought of as a pre-processing step for feature engineering. You might easily start with thousands or millions of items, and use SVD to create a much smaller set of “k” items (e.g. 20 or 70).

##### Notes/Limitations:

\*SVD builds features that may or may not map neatly to items (such as movie genres or news topics). As in many areas of machine learning, the lack of explainability can be an issue).

\*SVD requires that there are no missing values. There are various ways to handle this, including (1) imputation of missing values, (2) mean-centering values around 0, or (3) <advanced> using a more advance technique, such as stochastic gradient descent to simulate SVD in populating the factored matrices.

\*Calculating the SVD matrices can be computationally expensive, although calculating ratings once the factorization is completed is very fast. You may need to create a subset of your data for SVD calculations to be successfully performed, especially on a machine with a small RAM footprint.

##### Data

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>
First 5 Rows and 5 Columns of Ratings Matrix
</caption>
<thead>
<tr>
<th style="text-align:right;">
Toy Story (1995)
</th>
<th style="text-align:right;">
GoldenEye (1995)
</th>
<th style="text-align:right;">
Four Rooms (1995)
</th>
<th style="text-align:right;">
Get Shorty (1995)
</th>
<th style="text-align:right;">
Copycat (1995)
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
5
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
3
</td>
</tr>
<tr>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
</tbody>
</table>
##### Model

The SVD model(s) will be trained using a 70% split of training to 30% test data. The data will be split similar to that of Project 2 by choosing users that have rated over 50 movies and movies that have been rated over 100 times.

``` r
set.seed(101)
working_data <- MovieLense[rowCounts(MovieLense) > 50, colCounts(MovieLense) > 100]
train_index <- sample(x = c(T,F), size = nrow(working_data), replace = T, prob = c(0.7,0.3))

#set train and test sets  
train <- MovieLense[train_index,]
test <- MovieLense[!train_index,]

#build model
svd_k10 <- Recommender(data = train, method = 'SVD', parameter = list(k = 10))
svd_k20 <- Recommender(data = train, method = 'SVD', parameter = list(k = 20))
svd_k30 <- Recommender(data = train, method = 'SVD', parameter = list(k = 30))

#predictions for model
pred_svd_k10 <- predict(svd_k10, newdata = test, type = "ratings")
pred_svd_k20 <- predict(svd_k20, newdata = test, type = "ratings")
pred_svd_k30 <- predict(svd_k30, newdata = test, type = "ratings")

#evaluation schemes
eval_schem_pred_svd_k10 <- evaluationScheme(pred_svd_k10, 
                                            given = -1 , 
                                            method = "cross-validation")
eval_schem_pred_svd_k20 <- evaluationScheme(pred_svd_k20, 
                                            given = -1,
                                            method = "cross-validation")
eval_schem_pred_svd_k30 <- evaluationScheme(pred_svd_k30, 
                                            given = -1,
                                            method = "cross-validation")
```

    ## SVD run fold/sample [model time/prediction time]
    ##   1  [0.101sec/0.026sec] 
    ##   2  [0.233sec/0.038sec] 
    ##   3  [0.095sec/0.055sec] 
    ##   4  [0.088sec/0.04sec] 
    ##   5  [0.098sec/0.03sec] 
    ##   6  [0.079sec/0.026sec] 
    ##   7  [0.105sec/0.041sec] 
    ##   8  [0.108sec/0.026sec] 
    ##   9  [0.13sec/0.039sec] 
    ##   10  [0.086sec/0.039sec]

    ## SVD run fold/sample [model time/prediction time]
    ##   1  [0.091sec/0.026sec] 
    ##   2  [0.114sec/0.027sec] 
    ##   3  [0.089sec/0.027sec] 
    ##   4  [0.108sec/0.028sec] 
    ##   5  [0.246sec/0.024sec] 
    ##   6  [0.082sec/0.035sec] 
    ##   7  [0.086sec/0.036sec] 
    ##   8  [0.099sec/0.029sec] 
    ##   9  [0.13sec/0.03sec] 
    ##   10  [0.125sec/0.039sec]

    ## SVD run fold/sample [model time/prediction time]
    ##   1  [0.127sec/0.03sec] 
    ##   2  [0.126sec/0.03sec] 
    ##   3  [0.092sec/0.026sec] 
    ##   4  [0.103sec/0.031sec] 
    ##   5  [0.105sec/0.04sec] 
    ##   6  [0.1sec/0.025sec] 
    ##   7  [0.096sec/0.044sec] 
    ##   8  [0.107sec/0.033sec] 
    ##   9  [0.243sec/0.024sec] 
    ##   10  [0.104sec/0.026sec]

##### Evaluating the Models

<table class="table table-striped" style="margin-left: auto; margin-right: auto;">
<caption>
K = 10
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
RMSE
</th>
<th style="text-align:right;">
MSE
</th>
<th style="text-align:right;">
MAE
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
res
</td>
<td style="text-align:right;">
0.0156682
</td>
<td style="text-align:right;">
0.0002775
</td>
<td style="text-align:right;">
0.00776
</td>
</tr>
</tbody>
</table>
<table class="table table-striped" style="margin-left: auto; margin-right: auto;">
<caption>
K = 20
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
RMSE
</th>
<th style="text-align:right;">
MSE
</th>
<th style="text-align:right;">
MAE
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
res
</td>
<td style="text-align:right;">
0.0340359
</td>
<td style="text-align:right;">
0.0013109
</td>
<td style="text-align:right;">
0.0192146
</td>
</tr>
</tbody>
</table>
<table class="table table-striped" style="margin-left: auto; margin-right: auto;">
<caption>
K = 30
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
RMSE
</th>
<th style="text-align:right;">
MSE
</th>
<th style="text-align:right;">
MAE
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
res
</td>
<td style="text-align:right;">
0.0377705
</td>
<td style="text-align:right;">
0.0014845
</td>
<td style="text-align:right;">
0.0243979
</td>
</tr>
</tbody>
</table>
##### Summary

<table class="table table-striped" style="margin-left: auto; margin-right: auto;">
<caption>
K = 10 Predictions
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
Toy Story (1995)
</th>
<th style="text-align:right;">
GoldenEye (1995)
</th>
<th style="text-align:right;">
Four Rooms (1995)
</th>
<th style="text-align:right;">
Get Shorty (1995)
</th>
<th style="text-align:right;">
Copycat (1995)
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
3
</td>
<td style="text-align:right;">
2.778561
</td>
<td style="text-align:right;">
2.755891
</td>
<td style="text-align:right;">
2.733944
</td>
<td style="text-align:right;">
2.808045
</td>
<td style="text-align:right;">
2.758025
</td>
</tr>
<tr>
<td style="text-align:left;">
11
</td>
<td style="text-align:right;">
3.442148
</td>
<td style="text-align:right;">
3.371450
</td>
<td style="text-align:right;">
3.453938
</td>
<td style="text-align:right;">
3.599903
</td>
<td style="text-align:right;">
3.408728
</td>
</tr>
<tr>
<td style="text-align:left;">
12
</td>
<td style="text-align:right;">
4.399158
</td>
<td style="text-align:right;">
4.385571
</td>
<td style="text-align:right;">
4.403879
</td>
<td style="text-align:right;">
5.000000
</td>
<td style="text-align:right;">
4.424181
</td>
</tr>
<tr>
<td style="text-align:left;">
13
</td>
<td style="text-align:right;">
3.000000
</td>
<td style="text-align:right;">
3.000000
</td>
<td style="text-align:right;">
2.937381
</td>
<td style="text-align:right;">
5.000000
</td>
<td style="text-align:right;">
1.000000
</td>
</tr>
<tr>
<td style="text-align:left;">
14
</td>
<td style="text-align:right;">
4.082355
</td>
<td style="text-align:right;">
4.163609
</td>
<td style="text-align:right;">
4.010348
</td>
<td style="text-align:right;">
4.218969
</td>
<td style="text-align:right;">
4.066951
</td>
</tr>
</tbody>
</table>
<table class="table table-striped" style="margin-left: auto; margin-right: auto;">
<caption>
K = 20 Predictions
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
Toy Story (1995)
</th>
<th style="text-align:right;">
GoldenEye (1995)
</th>
<th style="text-align:right;">
Four Rooms (1995)
</th>
<th style="text-align:right;">
Get Shorty (1995)
</th>
<th style="text-align:right;">
Copycat (1995)
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
3
</td>
<td style="text-align:right;">
2.732015
</td>
<td style="text-align:right;">
2.784856
</td>
<td style="text-align:right;">
2.699394
</td>
<td style="text-align:right;">
2.915180
</td>
<td style="text-align:right;">
2.752756
</td>
</tr>
<tr>
<td style="text-align:left;">
11
</td>
<td style="text-align:right;">
3.788347
</td>
<td style="text-align:right;">
3.250558
</td>
<td style="text-align:right;">
3.384717
</td>
<td style="text-align:right;">
3.348055
</td>
<td style="text-align:right;">
3.455905
</td>
</tr>
<tr>
<td style="text-align:left;">
12
</td>
<td style="text-align:right;">
4.534314
</td>
<td style="text-align:right;">
4.338478
</td>
<td style="text-align:right;">
4.439787
</td>
<td style="text-align:right;">
5.000000
</td>
<td style="text-align:right;">
4.420718
</td>
</tr>
<tr>
<td style="text-align:left;">
13
</td>
<td style="text-align:right;">
3.000000
</td>
<td style="text-align:right;">
3.000000
</td>
<td style="text-align:right;">
3.257813
</td>
<td style="text-align:right;">
5.000000
</td>
<td style="text-align:right;">
1.000000
</td>
</tr>
<tr>
<td style="text-align:left;">
14
</td>
<td style="text-align:right;">
4.130642
</td>
<td style="text-align:right;">
4.122807
</td>
<td style="text-align:right;">
4.092871
</td>
<td style="text-align:right;">
4.203746
</td>
<td style="text-align:right;">
4.076789
</td>
</tr>
</tbody>
</table>
<table class="table table-striped" style="margin-left: auto; margin-right: auto;">
<caption>
K = 30 Predictions
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
Toy Story (1995)
</th>
<th style="text-align:right;">
GoldenEye (1995)
</th>
<th style="text-align:right;">
Four Rooms (1995)
</th>
<th style="text-align:right;">
Get Shorty (1995)
</th>
<th style="text-align:right;">
Copycat (1995)
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
3
</td>
<td style="text-align:right;">
2.662091
</td>
<td style="text-align:right;">
2.693945
</td>
<td style="text-align:right;">
2.711290
</td>
<td style="text-align:right;">
2.863166
</td>
<td style="text-align:right;">
2.696945
</td>
</tr>
<tr>
<td style="text-align:left;">
11
</td>
<td style="text-align:right;">
3.774994
</td>
<td style="text-align:right;">
3.237380
</td>
<td style="text-align:right;">
3.370444
</td>
<td style="text-align:right;">
3.437506
</td>
<td style="text-align:right;">
3.447535
</td>
</tr>
<tr>
<td style="text-align:left;">
12
</td>
<td style="text-align:right;">
4.599736
</td>
<td style="text-align:right;">
4.400397
</td>
<td style="text-align:right;">
4.404905
</td>
<td style="text-align:right;">
5.000000
</td>
<td style="text-align:right;">
4.402046
</td>
</tr>
<tr>
<td style="text-align:left;">
13
</td>
<td style="text-align:right;">
3.000000
</td>
<td style="text-align:right;">
3.000000
</td>
<td style="text-align:right;">
3.020693
</td>
<td style="text-align:right;">
5.000000
</td>
<td style="text-align:right;">
1.000000
</td>
</tr>
<tr>
<td style="text-align:left;">
14
</td>
<td style="text-align:right;">
4.417959
</td>
<td style="text-align:right;">
4.138703
</td>
<td style="text-align:right;">
4.016198
</td>
<td style="text-align:right;">
4.205475
</td>
<td style="text-align:right;">
4.046890
</td>
</tr>
</tbody>
</table>
The time it takes to process the recommendations increases with K, however the RMSE, MSE and MAE also appear to increase with K as well. Looking at the first 5 users' ratings for the first 5 months, the ratings seem to be about the same with a little variance. I'd say in this case going with k = 10 would work out better working with a much larger dataset just to be time conservative and also since it had the best RMSE than k = 20 and k = 30.
