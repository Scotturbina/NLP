require(ggplot2)
require(e1071)
require(caret)
require(quanteda) # text analytics
require(irlba)# Feature extraction
require(randomForest)

# Load the .csv data and explore in R Studio

spam.raw<-read.csv("~/Kaggle/spam.csv", stringsAsFactors = FALSE)

# Clean up the data frame 

spam.raw<-spam.raw[,1:2]
# Change columns names

names(spam.raw)<-c("Label", "Text")

# Check for missing values
length(which(!complete.cases(spam.raw)))

# convert our class label into factor

spam.raw$Label<-as.factor(spam.raw$Label)


# Lets take a look at distribution of the class labels

# the proportion of the labels
prop.table(table(spam.raw$Label))

table(spam.raw$Label)

# Lets see the text length of the SMS
spam.raw$TextLength<-nchar(spam.raw$Text)
summary(spam.raw$TextLength)

# Visualize distribution with ggplot2 adding segmentation for ham/spam

ggplot(spam.raw, aes(x= TextLength, fill= Label))+theme_bw()+geom_histogram(binwidth = 5)

# Use caret to create 70%/30% stratified split. Set random 
#seed

set.seed(32984)

indexes<-createDataPartition(spam.raw$Label, times = 1,p=0.7, list = FALSE)

train<-spam.raw[indexes,]
test<-spam.raw[-indexes,]

train.tokens<-tokens(train$Text, what = "word", remove_numbers = TRUE, remove_punct = TRUE,
                     remove_symbols = TRUE, remove_hyphens = TRUE, ngrams = 1L)


# Convert all to lower

train.tokens<-tokens_tolower(train.tokens)

train.tokens<-tokens_select(train.tokens, stopwords(),selection = "remove")

## Steeming

train.tokens<-tokens_wordstem(train.tokens, language = "english")

# Create our first bag of words model

train.tokens.dfm<-dfm(train.tokens, tolower = FALSE)

# transform to a matrix and inspect

train.tokens.matrix<-as.matrix(train.tokens.dfm)
View(train.tokens.matrix)

# I got the dimension for the metrics

dim(train.tokens.matrix)

# Wi will use CV
# require more processing and therefore more time

# Setup a the feature data frame with labels

train.tokens.df<-cbind(Label= train$Label, as.data.frame(train.tokens.dfm))

# cleanup column names

names(train.tokens.df)<-make.names(names(train.tokens.df))

# Use caret to create stratified 10 fold cv repeated
# 3 times

set.seed(48743)
cv.folds<-createMultiFolds(train$Label, k=10, times = 3)

cv.cntrl<-trainControl(method = "repeatedcv", number = 10, repeats =3, index = cv.folds)# if we do not specified index we are not going to get stratified cv

#to cut down on total execution time, use
#the doSNOW package to allow multi-core training in parallel

# it is important to alter the code to my environment

require(doSNOW)

# Time the code execution

start.time<-Sys.time()

cl<-makeNWScluster(2, type= "SOCK")

registerDoSNOW(cl)

total.time<-Sys.time()-start.time
total.time

rpart.cv.1<-train(Label~.,data=train.tokens.df, method= "rpart", trControl=cv.cntrl, tuneLength=7)



