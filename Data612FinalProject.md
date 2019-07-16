Data 612 Final Project
================

### Introduction

##### Objective

The overall goal is to produce quality recommendations by extracting insights from a large dataset. You may do so using Spark, or another distributed computing method, OR by effectively applying one of the more advanced mathematical techniques we have covered.

You should use this project to showcase some of the concepts that you have learned in this course, while delivering on the (probably) less familiar Spark platform. You are welcome to submit a compelling alternative proposal (subject to approval), such as implementing a recommender system using in Microsoft Azure ML Studio or with Google TensorFlow, or building out an application of a certain complexity using another tool. You may work in a small group (2-3) on this assignment.

##### Intro

For this project, I will be building a recommender system for books. The method I plan on using is ALS as well as utilizing a Spark Cluster on my local drive. The dataset I am using is Book-Corssing datasetm which contains user ratings on thousands of books. I didn't end up using the ALS method in my previous assignments, in one of them I actually used the SVD model instead. I also wasn't able to use Spark for the previous assignment so I found the final to be a good opportunity to use both. I will be building the ALS model and tuning the iterations to see how it effects the ALS performance.

### Exploring the Data

First we load the dataset from github, explore the data, and then create training and test sets to build the model. The dataset is the Book-Corssing Dataset. The data contains the book rating information. Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

##### Ratings

``` r
#load data
ratings = as.data.frame(read.csv("https://raw.githubusercontent.com/hvasquez81/DATA612/master/BX-Book-Ratings.csv"))

#Checking data dimensions
dim(ratings)
```

    ## [1] 1048575       3

``` r
# Check rating hist
hist((ratings %>% filter(Book.Rating != 0))[,3], main = "Distribution of Book Ratings")
```

![](Data612FinalProject_files/figure-markdown_github/unnamed-chunk-2-1.png)

The histogram shows the distribution of ratings, for books that have been rated. Lookin at the code, books with ratings of 0 have been excluded since those were not rated by the user. There is an apparent left skewness in this histogram.

##### Users

``` r
books_per_user = ratings %>% 
  filter(Book.Rating !=0) %>% 
  group_by(User.ID) %>% 
  tally()
```

    ## Warning: The `printer` argument is deprecated as of rlang 0.3.0.
    ## This warning is displayed once per session.

``` r
summary(books_per_user$n)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ##    1.000    1.000    1.000    5.623    3.000 8524.000

Half of the people in this set have only read one book. 25% have read between 1 and 3 books, and 25% have read between 3 and 8524 books.

##### Books

``` r
number_of_times_books_rated = ratings %>% 
  filter(Book.Rating !=0) %>%
  group_by(ISBN) %>%
  tally()
summary(number_of_times_books_rated$n)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   1.000   1.000   2.271   2.000 635.000

``` r
#mean book rating
mean(data.matrix((ratings %>% filter(Book.Rating !=0) %>% filter(!is.na(Book.Rating)) %>% select(Book.Rating))))
```

    ## [1] 7.601851

There's roughly 174k books, of which 75% of them have been rated between 1 and 2 times.

##### Minizing data - Relevant Data

``` r
#creating list of books and users
number_of_times_books_rated_morethan1 = number_of_times_books_rated %>% 
  filter(n >1 )
relv_books = number_of_times_books_rated_morethan1$ISBN
books_per_user_morethan1 = books_per_user %>% filter(n > 1)
relv_users = books_per_user_morethan1$User.ID

#filtering by books, users, and ratings greater than 0
relv_ratings = ratings %>% filter(ISBN %in% relv_books) %>% filter(User.ID %in% relv_users) %>% filter(Book.Rating >0)

dim(relv_ratings)
```

    ## [1] 244308      3

### Testing and Training sets

For the model building process I will be splitting the training and testing by 70%.

``` r
#connect to local spark cluster
sc = spark_connect(master = "local")
spark_data = relv_ratings
spark_data$User.ID = as.integer(spark_data$User.ID)
spark_data$ISBN = as.integer(spark_data$ISBN)

set.seed(1234)
#split training and test
training = sample(x = c(TRUE, FALSE), size = nrow(spark_data),
                      replace = TRUE, prob = c(0.70, 0.30))

train = spark_data[training, ]
test = spark_data[!training, ]

#load to sc
spark_training = sdf_copy_to(sc, 
                             train,
                             "training_set",
                             overwrite = TRUE)

spark_testing = sdf_copy_to(sc, 
                             test,
                             "testing_set",
                             overwrite = TRUE)
```

### Model Building and Testing

``` r
#model performance table
model_performance = as.data.frame(matrix(nrow = 25, ncol = 6))
colnames(model_performance) = c("iterations", 
                                "training time", 
                                "predicting time",
                                "MSE",
                                "RMSE",
                                "MAE")

#iterate for different max_iter values for ALS model
for (i in 1:25) { 

#model training
tic()
ALS_model = ml_als(spark_training,
                   max_iter = i,
                   reg_param = 0.1,
                   rating_col = "Book_Rating",
                   user_col = "User_ID",
                   item_col = "ISBN")
als_train_time = toc(quiet = TRUE)

#Model Predicting
tic()
ALS_model_predict = ALS_model$.jobj %>%
  invoke("transform", spark_dataframe(spark_testing)) %>%
  collect()
als_predict_time = toc(quiet = TRUE)

#performance - mse, rmse, mae
ALS_model_predict = ALS_model_predict[!is.na(ALS_model_predict$prediction), ]

#MSE
ALS_mse = mean((ALS_model_predict$Book_Rating - ALS_model_predict$prediction )^2)

#RMSE
ALS_rmse = sqrt(ALS_mse)

#MAE
ALS_mae = mean(abs(ALS_model_predict$Book_Rating - ALS_model_predict$prediction))

model_performance[i,"iterations"] = i
model_performance[i,"training time"] = round(als_train_time$toc - als_train_time$tic, 2)
model_performance[i,"predicting time"] = round(als_predict_time$toc - als_predict_time$tic, 2)
model_performance[i, "MSE"] = ALS_mse
model_performance[i, "RMSE"] = ALS_rmse
model_performance[i, "MAE"] = ALS_mae
}
```

### How did each Model perform?

``` r
model_performance %>% 
  kable() %>% 
  kable_styling("striped", full_width = TRUE)
```

<table class="table table-striped" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
iterations
</th>
<th style="text-align:right;">
training time
</th>
<th style="text-align:right;">
predicting time
</th>
<th style="text-align:right;">
MSE
</th>
<th style="text-align:right;">
RMSE
</th>
<th style="text-align:right;">
MAE
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2.36
</td>
<td style="text-align:right;">
1.04
</td>
<td style="text-align:right;">
94.461871
</td>
<td style="text-align:right;">
9.719150
</td>
<td style="text-align:right;">
8.197925
</td>
</tr>
<tr>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.38
</td>
<td style="text-align:right;">
0.52
</td>
<td style="text-align:right;">
69.490479
</td>
<td style="text-align:right;">
8.336095
</td>
<td style="text-align:right;">
7.080068
</td>
</tr>
<tr>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.35
</td>
<td style="text-align:right;">
0.52
</td>
<td style="text-align:right;">
46.472961
</td>
<td style="text-align:right;">
6.817108
</td>
<td style="text-align:right;">
5.701261
</td>
</tr>
<tr>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
1.42
</td>
<td style="text-align:right;">
0.44
</td>
<td style="text-align:right;">
32.425877
</td>
<td style="text-align:right;">
5.694372
</td>
<td style="text-align:right;">
4.658891
</td>
</tr>
<tr>
<td style="text-align:right;">
5
</td>
<td style="text-align:right;">
1.59
</td>
<td style="text-align:right;">
0.49
</td>
<td style="text-align:right;">
24.884585
</td>
<td style="text-align:right;">
4.988445
</td>
<td style="text-align:right;">
4.001963
</td>
</tr>
<tr>
<td style="text-align:right;">
6
</td>
<td style="text-align:right;">
1.81
</td>
<td style="text-align:right;">
0.44
</td>
<td style="text-align:right;">
20.407849
</td>
<td style="text-align:right;">
4.517505
</td>
<td style="text-align:right;">
3.569266
</td>
</tr>
<tr>
<td style="text-align:right;">
7
</td>
<td style="text-align:right;">
2.23
</td>
<td style="text-align:right;">
0.67
</td>
<td style="text-align:right;">
17.472460
</td>
<td style="text-align:right;">
4.180007
</td>
<td style="text-align:right;">
3.266391
</td>
</tr>
<tr>
<td style="text-align:right;">
8
</td>
<td style="text-align:right;">
2.04
</td>
<td style="text-align:right;">
0.43
</td>
<td style="text-align:right;">
15.417258
</td>
<td style="text-align:right;">
3.926482
</td>
<td style="text-align:right;">
3.043832
</td>
</tr>
<tr>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
2.10
</td>
<td style="text-align:right;">
0.40
</td>
<td style="text-align:right;">
13.902390
</td>
<td style="text-align:right;">
3.728591
</td>
<td style="text-align:right;">
2.874116
</td>
</tr>
<tr>
<td style="text-align:right;">
10
</td>
<td style="text-align:right;">
2.13
</td>
<td style="text-align:right;">
0.42
</td>
<td style="text-align:right;">
12.749219
</td>
<td style="text-align:right;">
3.570605
</td>
<td style="text-align:right;">
2.741013
</td>
</tr>
<tr>
<td style="text-align:right;">
11
</td>
<td style="text-align:right;">
2.33
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
11.843289
</td>
<td style="text-align:right;">
3.441408
</td>
<td style="text-align:right;">
2.633696
</td>
</tr>
<tr>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
2.33
</td>
<td style="text-align:right;">
0.41
</td>
<td style="text-align:right;">
11.120251
</td>
<td style="text-align:right;">
3.334704
</td>
<td style="text-align:right;">
2.546024
</td>
</tr>
<tr>
<td style="text-align:right;">
13
</td>
<td style="text-align:right;">
2.47
</td>
<td style="text-align:right;">
0.43
</td>
<td style="text-align:right;">
10.530513
</td>
<td style="text-align:right;">
3.245075
</td>
<td style="text-align:right;">
2.472968
</td>
</tr>
<tr>
<td style="text-align:right;">
14
</td>
<td style="text-align:right;">
2.63
</td>
<td style="text-align:right;">
0.40
</td>
<td style="text-align:right;">
10.043225
</td>
<td style="text-align:right;">
3.169105
</td>
<td style="text-align:right;">
2.411482
</td>
</tr>
<tr>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
2.82
</td>
<td style="text-align:right;">
0.40
</td>
<td style="text-align:right;">
9.635060
</td>
<td style="text-align:right;">
3.104039
</td>
<td style="text-align:right;">
2.358929
</td>
</tr>
<tr>
<td style="text-align:right;">
16
</td>
<td style="text-align:right;">
3.05
</td>
<td style="text-align:right;">
0.46
</td>
<td style="text-align:right;">
9.289590
</td>
<td style="text-align:right;">
3.047883
</td>
<td style="text-align:right;">
2.313636
</td>
</tr>
<tr>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
3.07
</td>
<td style="text-align:right;">
0.42
</td>
<td style="text-align:right;">
8.994524
</td>
<td style="text-align:right;">
2.999087
</td>
<td style="text-align:right;">
2.274530
</td>
</tr>
<tr>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
3.06
</td>
<td style="text-align:right;">
0.70
</td>
<td style="text-align:right;">
8.740584
</td>
<td style="text-align:right;">
2.956448
</td>
<td style="text-align:right;">
2.240548
</td>
</tr>
<tr>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
3.36
</td>
<td style="text-align:right;">
0.41
</td>
<td style="text-align:right;">
8.520577
</td>
<td style="text-align:right;">
2.919003
</td>
<td style="text-align:right;">
2.210739
</td>
</tr>
<tr>
<td style="text-align:right;">
20
</td>
<td style="text-align:right;">
3.31
</td>
<td style="text-align:right;">
0.47
</td>
<td style="text-align:right;">
8.328759
</td>
<td style="text-align:right;">
2.885959
</td>
<td style="text-align:right;">
2.184324
</td>
</tr>
<tr>
<td style="text-align:right;">
21
</td>
<td style="text-align:right;">
3.30
</td>
<td style="text-align:right;">
0.47
</td>
<td style="text-align:right;">
8.160520
</td>
<td style="text-align:right;">
2.856662
</td>
<td style="text-align:right;">
2.160919
</td>
</tr>
<tr>
<td style="text-align:right;">
22
</td>
<td style="text-align:right;">
3.51
</td>
<td style="text-align:right;">
0.44
</td>
<td style="text-align:right;">
8.012155
</td>
<td style="text-align:right;">
2.830575
</td>
<td style="text-align:right;">
2.139968
</td>
</tr>
<tr>
<td style="text-align:right;">
23
</td>
<td style="text-align:right;">
3.92
</td>
<td style="text-align:right;">
0.45
</td>
<td style="text-align:right;">
7.880655
</td>
<td style="text-align:right;">
2.807250
</td>
<td style="text-align:right;">
2.121289
</td>
</tr>
<tr>
<td style="text-align:right;">
24
</td>
<td style="text-align:right;">
3.72
</td>
<td style="text-align:right;">
0.43
</td>
<td style="text-align:right;">
7.763557
</td>
<td style="text-align:right;">
2.786316
</td>
<td style="text-align:right;">
2.104580
</td>
</tr>
<tr>
<td style="text-align:right;">
25
</td>
<td style="text-align:right;">
3.95
</td>
<td style="text-align:right;">
0.40
</td>
<td style="text-align:right;">
7.658816
</td>
<td style="text-align:right;">
2.767457
</td>
<td style="text-align:right;">
2.089499
</td>
</tr>
</tbody>
</table>
As the iterations increase, from 1 to 20 training times also increase. However predicting times went down. The performance from 1 iteration to 20 is significant in MSE, RMSE and MAE. Although training time is about a third of the time as 20 iterations, the performance of 20 iterations is much better. Being my first time using Spark, I was pretty impressed with the speed. I noticed that the prediction times and training times were significantly faster than using the recommenderlab methods, which I expected they would be.
