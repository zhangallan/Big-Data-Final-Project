library(data.table) # Importing
library(gamlr) # Lasso regression
library(ggplot2) # Plotting
library(maptpx) # Topic modeling
library(tm) # Text cleaning
library(wordcloud) # Topic model visualization

## Allan's working directory
setwd("C:\\HOME\\Big Data")

########### INITIALIZATION #############
###### Read the data ######

## Number of rows to read in (also the number of documents). -1 imports everything
n_documents = 1000

data <- fread("Train_rev1.csv", nrows=n_documents)

###### Clean the data #####

## Subset out a certain number of descriptions
test = data[1:n_documents, FullDescription]

corp <- Corpus(VectorSource(test)) # make it into a corpus
corp <- tm_map(corp, content_transformer(tolower)) # have to do this -- otherwise it will no longer by PlainDocument

## This needs to be done first because the SMART word list includes punctuation
corp <- tm_map(corp,removeWords,stopwords("Smart"))

corp <- tm_map(corp,removeNumbers)
corp <- tm_map(corp,removePunctuation)

corp <- tm_map(corp,stripWhitespace)

## In case it becomes faster to simply save the corpus. Doubt we'll need this, but just in case
## writeCorpus(corp)

dtm <- DocumentTermMatrix(corp) # make it into matrix

#################### Exploring the dtm ####################

## See terms that appear in most documents. Gives us hints as to what else to remove
findFreqTerms(dtm, 1000)
findFreqTerms(dtm, 800)

## See highly correlated terms with computer
findAssocs(dtm, "computer", .5)

## See how removing different amounts of sparse terms affects our matrix
dtm
dtm_test = removeSparseTerms(dtm, .9)

#################### Principal Components ####################

X <- as.matrix(dtm_test) # use this to run regression
F = X/rowSums(X)

## Running PCA. Careful because this can take a while
test_pca = prcomp(F, scale=TRUE)
plot(test_pca)

## Look at top components of first two PCA components
test_pca$rotation[order(abs(test_pca$rotation[, 1]), decreasing=TRUE), 1][1:20]
test_pca$rotation[order(abs(test_pca$rotation[, 2]), decreasing=TRUE), 2][1:20]

## Plot out the PCA components
qplot(x = test_pca$rotation[, 1], y = test_pca$rotation[, 2], geom="text", label = names(test_pca$rotation[, 1]))

qplot(x = test_pca$rotation[, 3], y = test_pca$rotation[, 4], geom="text", label = names(test_pca$rotation[, 3]))

#################### Topic Modeling ####################
## I'm not entirely sure what this will get us, but it's something to try
## Assuming that each job posting is a document

## Warning this will take a while
tpc = topics(X, K = seq(5, 50, by=5))

## Wordcloud of the results
for (i in 1:tpc$K) {
    rand_color_group = row.names(brewer.pal.info)[runif(1, min=18, max=35)]
    rand_color = brewer.pal(9, rand_color_group)[runif(1, min=6, max=length(brewer.pal(9, rand_color_group)))]
    wordcloud(row.names(tpc$theta), freq=tpc$theta[, i], min.freq=.004, colors=rand_color)
    readline(prompt="Press enter to see the next wordcloud")
}

## Need to do some more research into Latent Dirichlet Allocation packages in R.
## Looks like topicmodels is one?

#################### Running gamlr on all terms ####################
y = data[1:n_documents, SalaryNormalized]

reg_allterms = gamlr(dtm_test, y)

coef(reg_allterms)[order(abs(coef(reg_allterms)), decreasing=TRUE), 1][1:20]

#################### Running PCR ####################
reg_pc = gamlr(predict(test_pca), y)

coef(reg_pc)[order(abs(coef(reg_pc)), decreasing=TRUE), 1][1:20]
