setwd("~/Desktop/Whats Cooking")
install.packages('jsonlite')
library(jsonlite)

train <- fromJSON("train.json")
test <- fromJSON("test.json")

#add dependent variable
test$cuisine <- NA
#combine data set
combi <- rbind(train, test)

#install package
install.packages("tm")
library(tm)
#create corpus
corpus <- Corpus(VectorSource(combi$ingredients))

# Lower Case the Corpus
corpus <- tm_map(corpus, tolower)
corpus[[1]]

# Remove punctuation in the corpus
corpus <- tm_map(corpus, removePunctuation)
corpus[[1]]

#Remove Stopwords
corpus <- tm_map(corpus, removeWords, c(stopwords('english')))
corpus[[1]]

#Remove Whitespaces
corpus <- tm_map(corpus, stripWhitespace)
corpus[[1]]

#Perform Stemming
corpus <- tm_map(corpus, stemDocument)
corpus[[1]]


corpus <- tm_map(corpus, PlainTextDocument)

#document matrix
frequencies <- DocumentTermMatrix(corpus) 
frequencies

#Data Exploration

#organizing frequency of terms
freq <- colSums(as.matrix(frequencies))
length(freq)

ord <- order(freq)
ord

#if you wish to export the matrix (to see how it looks) to an excel file
m <- as.matrix(frequencies)
dim(m) write.csv(m, file = 'matrix.csv')

#check most and least frequent words
freq[head(ord)]
freq[tail(ord)]

#check our table of 20 frequencies
head(table(freq),20)
tail(table(freq),20)

#remove sparse terms
sparse <- removeSparseTerms(frequencies, 1 - 3/nrow(frequencies))
dim(sparse)

#create a data frame for visualization
wf <- data.frame(word = names(freq), freq = freq)
head(wf)

#plot terms which appear atleast 10,000 times
library(ggplot2)

chart <- ggplot(subset(wf, freq >10000), aes(x = word, y = freq))
chart <- chart + geom_bar(stat = 'identity', color = 'black', fill = 'white')
chart <- chart + theme(axis.text.x=element_text(angle=45, hjust=1))
chart

#find associated terms
findAssocs(frequencies, c('salt','oil'), corlimit=0.30)

#plot 5000 most used words
wordcloud(names(freq), freq, max.words = 5000, scale = c(6, .1), colors = brewer.pal(6, 'Dark2'))

#create sparse as data frame
newsparse <- as.data.frame(as.matrix(sparse))
dim(newsparse)

#check if all words are appropriate
colnames(newsparse) <- make.names(colnames(newsparse))

#check for the dominant dependent variable
table(train$cuisine)

#add cuisine
newsparse$cuisine <- as.factor(c(train$cuisine, rep('italian', nrow(test))))

#split data 
mytrain <- newsparse[1:nrow(train),]
mytest <- newsparse[-(1:nrow(train)),]

library(xgboost)
library(Matrix)

# creating the matrix for training the model
ctrain <- xgb.DMatrix(Matrix(data.matrix(mytrain[,!colnames(mytrain) %in% c('cuisine')])), label = as.numeric(mytrain$cuisine)-1)

#advanced data set preparation
dtest <- xgb.DMatrix(Matrix(data.matrix(mytest[,!colnames(mytest) %in% c('cuisine')]))) 
watchlist <- list(train = ctrain, test = dtest)

#train multiclass model using softmax
#first model
xgbmodel <- xgboost(data = ctrain, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 20, verbose = 1, watchlist = watchlist)

#second model
xgbmodel2 <- xgboost(data = ctrain, max.depth = 20, eta = 0.2, nrounds = 250, objective = "multi:softmax", num_class = 20, watchlist = watchlist)

#third model
xgbmodel3 <- xgboost(data = ctrain, max.depth = 25, gamma = 2, min_child_weight = 2, eta = 0.1, nround = 250, objective = "multi:softmax", num_class = 20, verbose = 2,watchlist = watchlist)



