Data 612 Project 4
================

Accuracy and Beyond
===================

The goal of this assignment is give you practice working with accuracy and other recommender system metrics.

In this assignment you’re asked to do at least one or (if you like) both of the following:

-   Work in a small group, and/or
-   Choose a different dataset to work with from your previous projects.

Deliverables

1.  As in your previous assignments, compare the accuracy of at least two recommender system algorithms against your offline data.

2.  Implement support for at least one business or user experience goal such as increased serendipity, novelty, or diversity.

3.  Compare and report on any change in accuracy before and after you’ve made the change in \#2.

4.  As part of your textual conclusion, discuss one or more additional experiments that could be performed and/or metrics that could be evaluated only if online evaluation was possible. Also, briefly propose how you would design a reasonable online evaluation environment.

You’ll find some of the material discussed in this week’s reading to be helpful in completing this project. You may also want to look at papers online, or search for “recsys” on youtube or slideshare.net.

### Data

``` r
#split training at 70%, keep min-8, good ratings >= 5, k =1 (for now)
set.seed(123)
ratings_jokes = Jester5k[rowCounts(Jester5k) > 50, colCounts(Jester5k) > 2500]
es = evaluationScheme(data = ratings_jokes,
                      method = "split",
                      train = 0.70,
                      given = (min(rowCounts(ratings_jokes))-8),
                      goodRating = 5.00,
                      k = 1
                      )
#train, known, unknown
train = getData(es, "train")
known = getData(es, "known")
unknown = getData(es, "unknown")

# take a look at train
train
```

    ## 2712 x 70 rating matrix of class 'realRatingMatrix' with 182206 ratings.

``` r
#unknown items by user
qplot(rowCounts(unknown)) + geom_histogram(binwidth = 10) + ggtitle("Unknown Items by Users") +xlab("Number of Unknown Jokes")
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](DATA_612_Project_4_files/figure-markdown_github/unnamed-chunk-1-1.png)

For the Jester ratings I will be defining the joke ratings as following:

-   bad jokes: between -10 and -5
-   kinda bad jokes: between -5 and 0
-   kinda good jokes: between 0 and 5
-   good jokes: between 5 and 10

For this project I will be looking at jokes that have been rated over 2500 times, and users having rated over 50 jokes. The training set will be split as 70 percent of the data using the evaluateScheme function, and good jokes will be at a 5 and above rating.

### Models

##### IBCF

``` r
#model
IBCF_model = Recommender(data = train,
                         method = "IBCF",
                         parameter = NULL)

#predictions
IBCF_pred = predict(object = IBCF_model,
                    newdata = known,
                    n = 10,
                    type = "ratings")

#accuracy
calcPredictionAccuracy(x = IBCF_pred,
                                       data = unknown,
                                       byUser = FALSE
                                       )
```

    ##      RMSE       MSE       MAE 
    ##  5.104112 26.051963  4.039152

##### UBCF

``` r
#model
UBCF_model = Recommender(data = train,
                         method = "UBCF",
                         parameter = NULL)

#predictions
UBCF_pred = predict(object = UBCF_model,
                    newdata = known,
                    n = 10,
                    type = "ratings")

#accuracy
calcPredictionAccuracy(x = UBCF_pred,
                                       data = unknown,
                                       byUser = FALSE
                                       )
```

    ##      RMSE       MSE       MAE 
    ##  4.458046 19.874179  3.499569

### Evaluating the Models

``` r
#evaluate model
models_to_evaluate = list("IBCF" = list(name = "IBCF", param = NULL),
                          "UBCF" = list(name = "UBCF", param = NULL)
                          )

n_recom = c(1,5,seq(10,50,10))

eval_list_results = evaluate(x = es,
                        method = models_to_evaluate,
                        n = n_recom)
```

    ## IBCF run fold/sample [model time/prediction time]
    ##   1  [0.089sec/0.183sec] 
    ## UBCF run fold/sample [model time/prediction time]
    ##   1  [0.025sec/2.503sec]

``` r
#ROC curve
plot(eval_list_results,
     annotate = TRUE)
```

![](DATA_612_Project_4_files/figure-markdown_github/unnamed-chunk-4-1.png)

``` r
#Precision-Recall
plot(eval_list_results,
     "prec/rec",
     annotate = TRUE)
```

![](DATA_612_Project_4_files/figure-markdown_github/unnamed-chunk-4-2.png)

In both the ROC curveand the Precision-Recall curve,the UBCF model appears to perform the best. It's possible that we could optimize the numeric k parameter in the IBCF and possibly return a better performing model.

### Summary

Compairing the models, it is expected that UBCF model would perform the best, and based off of the ROC and precision-recall charts we see that is the case. When working with larger datasets, a UBCF model might be more time consuming, since it required to calculate similarities between users as new users are entered as opposed to the IBCF model which calculates the similarities between items, and then does not need access to the initial dataset, unless new items are being added. Since we'd expect users to be added more frequently than items, an IBCF model might be more appropriate for the sake of time. However, when working with a smaller dataset and looking for better results, and UBCF model would be of better use. In the case of choosing the IBCF model, we could then go ahead and optimize the k-parameter to tune the model to perform better.

In an online enviroment, we start off by having the user rate joke one, and then running the recommendation algorithm to have the user rate the next joke depening on the rate of the first joke and continue this process to see how the recommendation algorithm deals with live data as opposed to offline data.
