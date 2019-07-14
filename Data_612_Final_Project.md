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

![](Data_612_Final_Project_files/figure-markdown_github/unnamed-chunk-2-1.png) The histogram shows the distribution of ratings, for books that have been rated. Lookin at the code, books with ratings of 0 have been excluded since those were not rated by the user. There is an apparent left skewness in this histogram.

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

``` r
books_per_user_morethan1 = books_per_user %>% filter(n > 1)
summary(books_per_user_morethan1$n)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##    2.00    2.00    4.00   12.12    9.00 8524.00

If we take a look at this set, about 75% of people have read between 2 and 9 books, which is better than previously where a majority of the set had only read between 1 and 3 books.

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

There's roughly 174k books, of which 75% of them have been rated between 1 and 2 times.

``` r
number_of_times_books_rated_morethan1 = number_of_times_books_rated %>% 
  filter(n >1 )

summary(number_of_times_books_rated_morethan1$n)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   2.000   2.000   3.000   5.268   5.000 635.000

If we limit the set to only include books rated more than once, 75% of the books have been rated between 2 and 5 times.

##### Selecting the Relevant Data

``` r
#creating list of books and users
relv_books = number_of_times_books_rated_morethan1$ISBN
relv_users = books_per_user_morethan1$User.ID

#filtering by books, users, and ratings greater than 0
relv_ratings = ratings %>% filter(ISBN %in% relv_books) %>% filter(User.ID %in% relv_users) %>% filter(Book.Rating >0)

dim(relv_ratings)
```

    ## [1] 244308      3

After filtering out for the relevant data, we are left with 244k ratings to work with and build the model.

### Testing and Training sets

For the model building process I will be splitting the training and testing by 75%.

``` r
#connect to local spark cluster
sc = spark_connect(master = "local")
spark_data = relv_ratings
spark_data$User.ID = as.integer(spark_data$User.ID)
spark_data$ISBN = as.integer(spark_data$ISBN)

set.seed(1234)
#split training and test
training = sample(x = c(TRUE, FALSE), size = nrow(spark_data),
                      replace = TRUE, prob = c(0.75, 0.25))

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
model_performance = as.data.frame(matrix(nrow = 20, ncol = 6))
colnames(model_performance) = c("iterations", 
                                "training time", 
                                "predicting time",
                                "MSE",
                                "RMSE",
                                "MAE")

#iterate for different max_iter values for ALS model
for (i in 1:20) { 

#model training
tic()
ALS_model = ml_als(spark_training,
                   max_iter = i,
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
2.52
</td>
<td style="text-align:right;">
1.22
</td>
<td style="text-align:right;">
93.611090
</td>
<td style="text-align:right;">
9.675282
</td>
<td style="text-align:right;">
8.171920
</td>
</tr>
<tr>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.51
</td>
<td style="text-align:right;">
0.55
</td>
<td style="text-align:right;">
64.502021
</td>
<td style="text-align:right;">
8.031315
</td>
<td style="text-align:right;">
6.815844
</td>
</tr>
<tr>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.33
</td>
<td style="text-align:right;">
0.50
</td>
<td style="text-align:right;">
40.314764
</td>
<td style="text-align:right;">
6.349391
</td>
<td style="text-align:right;">
5.263617
</td>
</tr>
<tr>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
1.39
</td>
<td style="text-align:right;">
0.44
</td>
<td style="text-align:right;">
28.091130
</td>
<td style="text-align:right;">
5.300107
</td>
<td style="text-align:right;">
4.277764
</td>
</tr>
<tr>
<td style="text-align:right;">
5
</td>
<td style="text-align:right;">
1.62
</td>
<td style="text-align:right;">
0.42
</td>
<td style="text-align:right;">
21.948283
</td>
<td style="text-align:right;">
4.684899
</td>
<td style="text-align:right;">
3.704397
</td>
</tr>
<tr>
<td style="text-align:right;">
6
</td>
<td style="text-align:right;">
1.77
</td>
<td style="text-align:right;">
0.42
</td>
<td style="text-align:right;">
18.327450
</td>
<td style="text-align:right;">
4.281057
</td>
<td style="text-align:right;">
3.337946
</td>
</tr>
<tr>
<td style="text-align:right;">
7
</td>
<td style="text-align:right;">
1.80
</td>
<td style="text-align:right;">
0.40
</td>
<td style="text-align:right;">
15.927592
</td>
<td style="text-align:right;">
3.990939
</td>
<td style="text-align:right;">
3.083514
</td>
</tr>
<tr>
<td style="text-align:right;">
8
</td>
<td style="text-align:right;">
2.10
</td>
<td style="text-align:right;">
0.44
</td>
<td style="text-align:right;">
14.218463
</td>
<td style="text-align:right;">
3.770738
</td>
<td style="text-align:right;">
2.896063
</td>
</tr>
<tr>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
2.34
</td>
<td style="text-align:right;">
0.42
</td>
<td style="text-align:right;">
12.936145
</td>
<td style="text-align:right;">
3.596685
</td>
<td style="text-align:right;">
2.751797
</td>
</tr>
<tr>
<td style="text-align:right;">
10
</td>
<td style="text-align:right;">
2.25
</td>
<td style="text-align:right;">
0.42
</td>
<td style="text-align:right;">
11.937152
</td>
<td style="text-align:right;">
3.455018
</td>
<td style="text-align:right;">
2.636599
</td>
</tr>
<tr>
<td style="text-align:right;">
11
</td>
<td style="text-align:right;">
2.31
</td>
<td style="text-align:right;">
0.59
</td>
<td style="text-align:right;">
11.139189
</td>
<td style="text-align:right;">
3.337542
</td>
<td style="text-align:right;">
2.543054
</td>
</tr>
<tr>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
2.41
</td>
<td style="text-align:right;">
0.41
</td>
<td style="text-align:right;">
10.489966
</td>
<td style="text-align:right;">
3.238822
</td>
<td style="text-align:right;">
2.465523
</td>
</tr>
<tr>
<td style="text-align:right;">
13
</td>
<td style="text-align:right;">
2.58
</td>
<td style="text-align:right;">
0.42
</td>
<td style="text-align:right;">
9.954809
</td>
<td style="text-align:right;">
3.155124
</td>
<td style="text-align:right;">
2.400318
</td>
</tr>
<tr>
<td style="text-align:right;">
14
</td>
<td style="text-align:right;">
2.65
</td>
<td style="text-align:right;">
0.39
</td>
<td style="text-align:right;">
9.508646
</td>
<td style="text-align:right;">
3.083609
</td>
<td style="text-align:right;">
2.344860
</td>
</tr>
<tr>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
2.74
</td>
<td style="text-align:right;">
0.40
</td>
<td style="text-align:right;">
9.132785
</td>
<td style="text-align:right;">
3.022050
</td>
<td style="text-align:right;">
2.297111
</td>
</tr>
<tr>
<td style="text-align:right;">
16
</td>
<td style="text-align:right;">
2.92
</td>
<td style="text-align:right;">
0.39
</td>
<td style="text-align:right;">
8.813351
</td>
<td style="text-align:right;">
2.968729
</td>
<td style="text-align:right;">
2.255676
</td>
</tr>
<tr>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
2.89
</td>
<td style="text-align:right;">
0.42
</td>
<td style="text-align:right;">
8.540060
</td>
<td style="text-align:right;">
2.922338
</td>
<td style="text-align:right;">
2.219408
</td>
</tr>
<tr>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
3.19
</td>
<td style="text-align:right;">
0.37
</td>
<td style="text-align:right;">
8.304865
</td>
<td style="text-align:right;">
2.881816
</td>
<td style="text-align:right;">
2.187482
</td>
</tr>
<tr>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
3.23
</td>
<td style="text-align:right;">
0.47
</td>
<td style="text-align:right;">
8.101354
</td>
<td style="text-align:right;">
2.846288
</td>
<td style="text-align:right;">
2.159577
</td>
</tr>
<tr>
<td style="text-align:right;">
20
</td>
<td style="text-align:right;">
3.49
</td>
<td style="text-align:right;">
0.40
</td>
<td style="text-align:right;">
7.924356
</td>
<td style="text-align:right;">
2.815023
</td>
<td style="text-align:right;">
2.134997
</td>
</tr>
</tbody>
</table>
As the iterations increase, from 1 to 20 training times also increase. However predicting times went down. The performance from 1 iteration to 20 is significant in MSE, RMSE and MAE. Although training time is about a third of the time as 20 iterations, the performance of 20 iterations is much better. Being my first time using Spark, I was pretty impressed with the speed. I noticed that the prediction times and training times were significantly faster than using the recommenderlab methods, which I expected they would be.
