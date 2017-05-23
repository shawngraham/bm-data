options(java.parameters = "-Xmx6144m")
install.packages("mallet")
install.packages("rJava")
require(mallet)
require(rJava)
library(mallet)



captionstext <- read.csv("bmsource.csv", stringsAsFactors = FALSE)
## kludge for when username comesthrough as numeric rather than character
captionstext$id <-as.character(captionstext$id)

documents <- data.frame(text = captionstext$tweets,
                        id = make.unique(captionstext$id),
                        class = captionstext$keywords,
                        stringsAsFactors=FALSE)
mallet.instances <- mallet.import(documents$id, documents$text, "en.txt", token.regexp = "\\p{L}[\\p{L}\\p{P}]+\\p{L}")



## Create a topic trainer object.
n.topics <- 25
topic.model <- MalletLDA(n.topics)

## Load our documents. We could also pass in the filename of a
##  saved instance list file that we build from the command-line tools.
topic.model$loadDocuments(mallet.instances)

## Get the vocabulary, and some statistics about word frequencies.
##  These may be useful in further curating the stopword list.
vocabulary <- topic.model$getVocabulary()
word.freqs <- mallet.word.freqs(topic.model)

## Optimize hyperparameters every 20 iterations,
##  after 50 burn-in iterations.
topic.model$setAlphaOptimization(20, 50)

## Now train a model. Note that hyperparameter optimization is on, by default.
##  We can specify the number of iterations. Here we'll use a large-ish round number.
topic.model$train(1000)

## NEW: run through a few iterations where we pick the best topic for each token,
##  rather than sampling from the posterior distribution.
topic.model$maximize(10)

## Get the probability of topics in documents and the probability of words in topics.
## By default, these functions return raw word counts. Here we want probabilities,
##  so we normalize, and add "smoothing" so that nothing has exactly 0 probability.
doc.topics <- mallet.doc.topics(topic.model, smoothed=T, normalized=T)
topic.words <- mallet.topic.words(topic.model, smoothed=T, normalized=T)
# from http://www.cs.princeton.edu/~mimno/R/clustertrees.R
## transpose and normalize the doc topics
topic.docs <- t(doc.topics)
topic.docs <- topic.docs / rowSums(topic.docs)

write.csv(doc.topics, file = "25topicsinposts.csv")

## Get a vector containing short names for the topics
topics.labels <- rep("", n.topics)
for (topic in 1:n.topics) topics.labels[topic] <- paste(mallet.top.words(topic.model, topic.words[topic,], num.top.words=6)$words, collapse=" ")
# have a look at keywords for each topic
topics.labels

# create data.frame with columns as authors and rows as topics
topic_docs <- data.frame(topic.docs)
names(topic_docs) <- documents$id

# find top n topics for a certain author
df1 <- t(topic_docs[,grep("2", names(topic_docs))])

#8963295 is a person who has 'for sale' in her post
#255766488 natural_selections - skullshop.ca
#361451583 ryan matthew cohn
#234396855 pandora's box york
colnames(df1) <- topics.labels
require(reshape2)
topic.proportions.df <- melt(cbind(data.frame(df1),
                                   document=factor(1:nrow(df1))),
                             variable.name="topic",
                             id.vars = "document")
# plot for each doc by that author
require(ggplot2)
dpi=600    #pixels per square inch
png("jan5-newsourcedata/fig-pandorasbox.png", width=14*dpi, height=14*dpi, res=dpi)

ggplot(topic.proportions.df, aes(topic, value, fill=document)) +
  geom_bar(stat="identity") +
  ylab("proportion") +
  theme(axis.text.x = element_text(angle=90, hjust=1)) +
  coord_flip() +
  facet_wrap(~ document, ncol=5)
dev.off()


## cluster based on shared words
dpi=600    #pixels per square inch
png("jan5-newsourcedata/fig-topics-w-labels.png", width=14*dpi, height=14*dpi, res=dpi)
plot(hclust(dist(topic.words)), labels=topics.labels)
dev.off()

dpi=600    #pixels per square inch
png("jan5-newsourcedata/fig-topics-with-topic-numbers.png", width=14*dpi, height=14*dpi, res=dpi)
plot(as.dendrogram(hclust(dist(topic.words))),horiz=F,cex=1, main="Dendrogram of topics within Instagram Bone Trade posts")
dev.off()



##### 04

library(cluster)
topic_df_dist <-  as.matrix(daisy(t(topic_docs), metric =  "euclidean", stand = TRUE))
# Change row values to zero if less than row minimum plus row standard deviation
# keep only closely related documents and avoid a dense spagetti diagram
# that's difficult to interpret (hat-tip: http://stackoverflow.com/a/16047196/1036500)
topic_df_dist[ sweep(topic_df_dist, 1, (apply(topic_df_dist,1,min) + apply(topic_df_dist,1,sd) )) > 0 ] <- 0

###05
#' Use kmeans to identify groups of similar authors

km <- kmeans(topic_df_dist, n.topics)
# get names for each cluster
allnames <- vector("list", length = n.topics)
for(i in 1:n.topics){
  allnames[[i]] <- names(km$cluster[km$cluster == i])
}

# Here's the list of authors by group
allnames



###06
#### network diagram using Fruchterman & Reingold algorithm
# static
install.packages("igraph")
library(igraph)
g <- as.undirected(graph.adjacency(topic_df_dist))
layout1 <- layout.fruchterman.reingold(g, niter=500)
plot(g, layout=layout1, edge.curved = TRUE, vertex.size = 1,  vertex.color= "grey", edge.arrow.size = 0, vertex.label.dist=0.5, vertex.label = NA)


# interactive in a web browser
install.packages("devtools")
devtools::install_github("d3Network", "christophergandrud")
require(d3Network)
d3SimpleNetwork(get.data.frame(g),width = 1500, height = 800,
                textColour = "orange", linkColour = "red",
                fontsize = 10,
                nodeClickColour = "#E34A33",
                charge = -100, opacity = 0.9, file = "d3net.html")
# find the html file in working directory and open in a web browser

# for Gephi
# this line will export from R and make the file 'g.graphml'
# in the working directory, ready to open with Gephi
write.graph(g, file="g-postswithamounts.graphml", format="graphml")

## did line 172, 176, then wrote it to graphml file (l213). brought that into
## gephi. Gephi detected parallel edges, consolodated those into weight
## so most were weight 1, then 5,6,7. I removed weight 1
## this left nodes w/o any degree at all, so I removed those too.
