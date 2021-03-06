
library(tm)
library(RTextTools)
library(e1071)

library(dplyr)
library(caret)
# Library for parallel processing
library(doMC)
registerDoMC(cores=detectCores())  # Use all available cores


## Read in the data
df<- read.csv("C:/Users/sroberts/Downloads/movie-pang02.csv", stringsAsFactors = FALSE)
glimpse(df)

## Randomize the dataset
set.seed(1)
df <- df[sample(nrow(df)), ]
df <- df[sample(nrow(df)), ]
glimpse(df)

## Convert the 'class' variable from character to factor.
df$class <- as.factor(df$class)

## Tokenisation
corpus <- Corpus(VectorSource(df$text))
corpus
inspect(corpus[1:3])

## Data Cleanup
# Use dplyr's  %>% (pipe) utility to do this neatly.
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

## Create Document Term Matrix
dtm <- DocumentTermMatrix(corpus.clean)
inspect(dtm[40:50, 10:15])

## Partitioning the Data
df.train <- df[1:1500,]
df.test <- df[1501:2000,]

dtm.train <- dtm[1:1500,]
dtm.test <- dtm[1501:2000,]

corpus.clean.train <- corpus.clean[1:1500]
corpus.clean.test <- corpus.clean[1501:2000]

## Feature Selection
dim(dtm.train)
fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))
# Use only 5 most frequent words (fivefreq) to build the DTM
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))
dim(dtm.train.nb)

## Boolean feature Multinomial Naive Bayes
# We'll use a variation of the multinomial Naive Bayes algorithm known as binarized (boolean feature) Naive Bayes
# In this method, the term frequencies are replaced by Boolean presence/absence features.
# The logic behind this being that for sentiment classification, word occurrence matters more than word frequency.
# Function to convert the word frequencies to yes (presence) and no (absence) labels
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}
# Apply the convert_count function to get final training and testing DTMs
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

## Training the Naive Bayes Model
# Train the classifier
system.time( classifier <- naiveBayes(trainNB, df.train$class, laplace = 1) )
# Testing the Predictions
system.time( pred <- predict(classifier, newdata=testNB) )
# Create a truth table by tabulating the predicted class labels with the actual class labels 
table("Predictions"=pred, "Actual"=df.test$class)

## Confusion matrix
conf.mat <- confusionMatrix(data=pred, reference=df.test$class)
conf.mat
conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy']
