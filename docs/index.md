--- 
title: "Hands-on Machine Learning with R"
date: "2018-08-07"
site: bookdown::bookdown_site
output: bookdown::gitbook
documentclass: book
bibliography: [book.bib, packages.bib]
biblio-style: apalike
link-citations: yes
github-repo: bradleyboehmke/hands-on-machine-learning-with-r
description: "A Machine Learning Algorithmic Deep Dive Using R."
---


# Preface {-}

<img src="images/mlr.png"  style="float:right; margin: 0px 0px 0px 0px; width: 40%; height: 40%;" />
Welcome to Hands-on Machine Learning with R.  This book provides hands-on modules for many of the most common machine learning methods to include:

- Generalized low rank models
- Clustering algorithms
- Autoencoders
- Regularized models
- Random forests 
- Gradient boosting machines 
- Deep neural networks
- Stacking / super learner
- and more!

You will learn how to build and tune these various models with R packages that have been tested and approved due to their ability to scale well. However, my motivation in almost every case is to describe the techniques in a way that helps develop intuition for its strengths and weaknesses.  For the most part, I minimize mathematical complexity when possible but also provide resources to get deeper into the details if desired.


## Who should read this {-}

I intend this work to be a practitioner's guide to the machine learning process and a place where one can come to learn about the approach and to gain intuition about the many commonly used, modern, and powerful methods accepted in the machine learning community. If you are familiar with the analytic methodologies, this book may still serve as a reference for how to work with the various R packages for implementation.  While an abundance of videos, blog posts, and tutorials exist online, I've long been frustrated by the lack of consistency, completeness, and bias towards singular packages for implementation. This is what inspired this book. 

This book is not meant to be an introduction to R or to programming in general; as I assume the reader has familiarity with the R language to include defining functions, managing R objects, controlling the flow of a program, and other basic tasks.  If not, I would refer you to [R for Data Science](http://r4ds.had.co.nz/index.html) [@wickham2016r] to learn the fundamentals of data science with R such as importing, cleaning, transforming, visualizing, and exploring your data. For those looking to advance their R programming skills and knowledge of the languge, I would refer you to [Advanced R](http://adv-r.had.co.nz/) [@wickham2014advanced]. Nor is this book designed to be a deep dive into the theory and math underpinning machine learning algorithms. Several books already exist that do great justice in this arena (i.e. [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) [@esl], [Computer Age Statistical Inference](https://web.stanford.edu/~hastie/CASI/) [@apm], [Deep Learning](http://www.deeplearningbook.org/) [@goodfellow2016deep]). 

Instead, this book is meant to help R users learn to use the machine learning stack within R, which includes using various R packages such as `glmnet`, `h20`, `ranger`, `xgboost`, `lime`, and others to effectively model and gain insight from your data. The book favors a hands-on approach, growing an intuitive understanding of machine learning through concrete examples and just a little bit of theory.  While you can read this book without opening R, I highly recommend you experiment with the code examples provided throughout.

## Why R {-}

R has emerged over the last couple decades as a first-class tool for scientific computing tasks, and has been a consistent leader in implementing statistical methodologies for analyzing data. The usefulness of R for data science stems from the large, active, and growing ecosystem of third-party packages: `tidyverse` for common data analysis activities; `h2o`, `ranger`, `xgboost`, and others for fast and scalable machine learning; `lime`, `pdp`, `DALEX`, and others for machine learning interpretability; and many more tools will be mentioned throughout the pages that follow.  

## Structure of the book {-} 

Each chapter of this book focuses on a particular part of the machine learning process along with various packages to perform that process.  

TBD...


## Conventions used in this book {-}

The following typographical conventions are used in this book:

* ___strong italic___: indicates new terms,
* __bold__: indicates package & file names,
* `inline code`: monospaced highlighted text indicates functions or other commands that could be typed literally by the user,
* code chunk: indicates commands or other text that could be typed literally by the user


```r
1 + 2
## [1] 3
```

In addition to the general text used throughout, you will notice the following code chunks with images, which signify:

<div class="rmdtip">
<p>Signifies a tip or suggestion</p>
</div>

<div class="rmdnote">
<p>Signifies a general note</p>
</div>

<div class="rmdwarning">
<p>Signifies a warning or caution</p>
</div>


## Additional resources {-}

There are many great resources available to learn about machine learning.  At the end of each chapter I provide a *Learn More* section that lists resources that I have found extremely useful for digging deeper into the methodology and applying with code.  


## Feedback {-}

Reader comments are greatly appreciated.  To report errors or bugs please post an issue at https://github.com/bradleyboehmke/hands-on-machine-learning-with-r/issues.


## Acknowledgments {-} 

TBD


## Software information {-} 

An online version of this book is available at https://bradleyboehmke.github.io/hands-on-machine-learning-with-r/.  The source of the book is available at https://github.com/bradleyboehmke/hands-on-machine-learning-with-r. The book is powered by https://bookdown.org which makes it easy to turn R markdown files into HTML, PDF, and EPUB.

This book was built with the following packages and R version.  All code was executed on 2013 MacBook Pro with a 2.4 GHz Intel Core i5 processor, 8 GB of memory, 1600MHz speed, and double data rate synchronous dynamic random access memory (DDR3). 


```r
# packages used
pkgs <- c(
  "AmesHousing",
  "caret",
  "data.table",
  "dplyr",
  "ggplot2",
  "glmnet",
  "h2o",
  "pROC",
  "purrr",
  "ROCR",
  "rsample"
)

# package & session info
devtools::session_info(pkgs)
#> Session info -------------------------------------------------------------
#>  setting  value                       
#>  version  R version 3.5.1 (2018-07-02)
#>  system   x86_64, darwin15.6.0        
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  tz       America/New_York            
#>  date     2018-08-07
#> Packages -----------------------------------------------------------------
#>  package      * version    date       source                         
#>  abind          1.4-5      2016-07-21 CRAN (R 3.5.0)                 
#>  AmesHousing    0.0.3      2017-12-17 CRAN (R 3.5.0)                 
#>  assertthat     0.2.0      2017-04-11 CRAN (R 3.5.0)                 
#>  backports      1.1.2      2017-12-13 CRAN (R 3.5.0)                 
#>  BH             1.66.0-1   2018-02-13 CRAN (R 3.5.0)                 
#>  bindr          0.1.1      2018-03-13 CRAN (R 3.5.0)                 
#>  bindrcpp       0.2.2      2018-03-29 CRAN (R 3.5.0)                 
#>  bitops         1.0-6      2013-08-17 CRAN (R 3.5.0)                 
#>  broom          0.5.0      2018-07-17 CRAN (R 3.5.0)                 
#>  caret          6.0-80     2018-05-26 CRAN (R 3.5.0)                 
#>  caTools        1.17.1     2014-09-10 CRAN (R 3.5.0)                 
#>  class          7.3-14     2015-08-30 CRAN (R 3.5.1)                 
#>  cli            1.0.0      2017-11-05 CRAN (R 3.5.0)                 
#>  codetools      0.2-15     2016-10-05 CRAN (R 3.5.1)                 
#>  colorspace     1.3-2      2016-12-14 CRAN (R 3.5.0)                 
#>  crayon         1.3.4      2017-09-16 CRAN (R 3.5.0)                 
#>  CVST           0.2-2      2018-05-26 CRAN (R 3.5.0)                 
#>  data.table     1.11.4     2018-05-27 CRAN (R 3.5.0)                 
#>  ddalpha        1.3.3      2018-04-30 CRAN (R 3.5.0)                 
#>  DEoptimR       1.0-8      2016-11-19 CRAN (R 3.5.0)                 
#>  dichromat      2.0-0      2013-01-24 CRAN (R 3.5.0)                 
#>  digest         0.6.15     2018-01-28 CRAN (R 3.5.0)                 
#>  dimRed         0.1.0      2017-05-04 CRAN (R 3.5.0)                 
#>  dplyr          0.7.6      2018-06-29 cran (@0.7.6)                  
#>  DRR            0.0.3      2018-01-06 CRAN (R 3.5.0)                 
#>  fansi          0.2.3      2018-05-06 cran (@0.2.3)                  
#>  foreach        1.4.4      2017-12-12 CRAN (R 3.5.0)                 
#>  gdata          2.18.0     2017-06-06 CRAN (R 3.5.0)                 
#>  geometry       0.3-6      2015-09-09 CRAN (R 3.5.0)                 
#>  ggplot2        3.0.0      2018-07-03 CRAN (R 3.5.0)                 
#>  glmnet         2.0-16     2018-04-02 CRAN (R 3.5.0)                 
#>  glue           1.3.0      2018-07-23 Github (tidyverse/glue@66de125)
#>  gower          0.1.2      2017-02-23 CRAN (R 3.5.0)                 
#>  gplots         3.0.1      2016-03-30 CRAN (R 3.5.0)                 
#>  graphics     * 3.5.1      2018-07-05 local                          
#>  grDevices    * 3.5.1      2018-07-05 local                          
#>  grid           3.5.1      2018-07-05 local                          
#>  gtable         0.2.0      2016-02-26 CRAN (R 3.5.0)                 
#>  gtools         3.5.0      2015-05-29 CRAN (R 3.5.0)                 
#>  h2o            3.18.0.11  2018-05-24 CRAN (R 3.5.0)                 
#>  ipred          0.9-6      2017-03-01 CRAN (R 3.5.0)                 
#>  iterators      1.0.9      2017-12-12 CRAN (R 3.5.0)                 
#>  jsonlite       1.5        2017-06-01 CRAN (R 3.5.0)                 
#>  kernlab        0.9-26     2018-04-30 CRAN (R 3.5.0)                 
#>  KernSmooth     2.23-15    2015-06-29 CRAN (R 3.5.1)                 
#>  labeling       0.3        2014-08-23 CRAN (R 3.5.0)                 
#>  lattice        0.20-35    2017-03-25 CRAN (R 3.5.1)                 
#>  lava           1.6.1      2018-03-28 CRAN (R 3.5.0)                 
#>  lazyeval       0.2.1      2017-10-29 CRAN (R 3.5.0)                 
#>  lubridate      1.7.4      2018-04-11 CRAN (R 3.5.0)                 
#>  magic          1.5-8      2018-01-26 CRAN (R 3.5.0)                 
#>  magrittr       1.5        2014-11-22 CRAN (R 3.5.0)                 
#>  MASS           7.3-50     2018-04-30 CRAN (R 3.5.1)                 
#>  Matrix         1.2-14     2018-04-13 CRAN (R 3.5.1)                 
#>  methods      * 3.5.1      2018-07-05 local                          
#>  mgcv           1.8-24     2018-06-23 CRAN (R 3.5.1)                 
#>  ModelMetrics   1.1.0      2016-08-26 CRAN (R 3.5.0)                 
#>  munsell        0.4.3      2016-02-13 CRAN (R 3.5.0)                 
#>  nlme           3.1-137    2018-04-07 CRAN (R 3.5.1)                 
#>  nnet           7.3-12     2016-02-02 CRAN (R 3.5.1)                 
#>  numDeriv       2016.8-1   2016-08-27 CRAN (R 3.5.0)                 
#>  pillar         1.3.0      2018-07-14 cran (@1.3.0)                  
#>  pkgconfig      2.0.1      2017-03-21 CRAN (R 3.5.0)                 
#>  plogr          0.2.0      2018-03-25 CRAN (R 3.5.0)                 
#>  plyr           1.8.4      2016-06-08 CRAN (R 3.5.0)                 
#>  pROC           1.12.1     2018-05-06 CRAN (R 3.5.0)                 
#>  prodlim        2018.04.18 2018-04-18 CRAN (R 3.5.0)                 
#>  purrr          0.2.5      2018-05-29 CRAN (R 3.5.0)                 
#>  R6             2.2.2      2017-06-17 CRAN (R 3.5.0)                 
#>  RColorBrewer   1.1-2      2014-12-07 CRAN (R 3.5.0)                 
#>  Rcpp           0.12.17    2018-05-18 CRAN (R 3.5.0)                 
#>  RcppRoll       0.2.2      2015-04-05 CRAN (R 3.5.0)                 
#>  RCurl          1.95-4.10  2018-01-04 CRAN (R 3.5.0)                 
#>  recipes        0.1.2      2018-01-11 CRAN (R 3.5.0)                 
#>  reshape2       1.4.3      2017-12-11 CRAN (R 3.5.0)                 
#>  rlang          0.2.1      2018-05-30 CRAN (R 3.5.0)                 
#>  robustbase     0.93-0     2018-04-24 CRAN (R 3.5.0)                 
#>  ROCR           1.0-7      2015-03-26 CRAN (R 3.5.0)                 
#>  rpart          4.1-13     2018-02-23 CRAN (R 3.5.1)                 
#>  rsample        0.0.2      2017-11-12 CRAN (R 3.5.0)                 
#>  scales         0.5.0      2017-08-24 CRAN (R 3.5.0)                 
#>  sfsmisc        1.1-2      2018-03-05 CRAN (R 3.5.0)                 
#>  splines        3.5.1      2018-07-05 local                          
#>  SQUAREM        2017.10-1  2017-10-07 CRAN (R 3.5.0)                 
#>  stats        * 3.5.1      2018-07-05 local                          
#>  stats4         3.5.1      2018-07-05 local                          
#>  stringi        1.2.4      2018-07-20 cran (@1.2.4)                  
#>  stringr        1.3.1      2018-05-10 CRAN (R 3.5.0)                 
#>  survival       2.42-3     2018-04-16 CRAN (R 3.5.1)                 
#>  tibble         1.4.2      2018-01-22 CRAN (R 3.5.0)                 
#>  tidyr          0.8.1      2018-05-18 CRAN (R 3.5.0)                 
#>  tidyselect     0.2.4      2018-02-26 CRAN (R 3.5.0)                 
#>  timeDate       3043.102   2018-02-21 CRAN (R 3.5.0)                 
#>  tools          3.5.1      2018-07-05 local                          
#>  utf8           1.1.4      2018-05-24 CRAN (R 3.5.0)                 
#>  utils        * 3.5.1      2018-07-05 local                          
#>  viridisLite    0.3.0      2018-02-01 CRAN (R 3.5.0)                 
#>  withr          2.1.2      2018-03-15 CRAN (R 3.5.0)
```




