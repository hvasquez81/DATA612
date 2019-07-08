Data 612 Final Project Planning Document
================

### Instructions

Find an interesting dataset and describe the system you plan to build out. If you would like to use one of the datasets you have already worked with, you should add a unique element or incorporate additional data. (i.e. explicit features you scrape from another source, like image analysis on movie posters). The overall goal, however, will be to produce quality recommendations by extracting insights from a large dataset. You may do so using Spark, or another distributed computing method, OR by effectively applying one of the more advanced mathematical techniques we have covered. There is no preference for one over the other, as long as your recommender works! The planning document should be written up and published as a notebook on GitHub or in RPubs.Please submit the link in the Unit 4 folder, due Thursday, July 5.

### The Dataset

The data set I will be working with for this project is the Book-Corssing Dataset. The book crossing dataset comrpsies of 3 tables. The dataset can be found and downloaded here: <http://www2.informatik.uni-freiburg.de/~cziegler/BX/>

##### BX-Users

Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Demographic data is provided (Location, Age) if available. Otherwise, these fields contain NULL-values.

##### BX-Books

Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (Book-Title, Book-Author, Year-Of-Publication, Publisher), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the Amazon web site.

##### BX-Book-Ratings

Contains the book rating information. Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

### The Recommender System

The recommender system I am looking to build will recommend books to users. Originally, I wanted to try implementing the attributes in the BX-Users table, however since userIDs map to integers for annonymity I am unable to do so. Also, age is extremely sparse in that dataset which is another issue with my original plan.
