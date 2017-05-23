### Ben Schmidt's WEM
library(devtools)
install_github("bmschmidt/wordVectors")
install.packages("magrittr")

install.packages("rtools")

## get the diaries into WEM
#install.packages('dplyr')
library(rtools)
library(wordVectors)
library(tsne)
library(magrittr)
library(dplyr)
library(ggplot2)



## train a new model


## the csv file is the tweets without ids & keywords & header
model = train_word2vec("bmsource2.csv",output="captions_vectors.bin",threads = 4,vectors = 1500,window=12,force=TRUE)

## or comment that line out and uncomment the line below to load an existing model
#model = read.vectors("captions_vectors.bin")

##Check
model
tail(rownames(model))

# below are example snippets looking for vectors aligned to particular words or phrases.
# holy
nearest_to(model,model[["holy"]])

holy_words = model %>% nearest_to(model[["holy"]],100) %>% names
sample(holy_words,20)

# wall
nearest_to(model,model[["wall"]])

wall_words = model %>% nearest_to(model[[c("wall")]],100) %>% names
sample(wall_words,10)

# hadrian

nearest_to(model,model[["hadrian"]])

hadrian_words = model %>% nearest_to(model[[c("hadrian")]],100) %>% names
sample(hadrian_words,50)


# in the examples below, we visualize some of these vectors
# legality

nearest_to(model,model[["trump"]])

trump_words = model %>% nearest_to(model[[c("trump","president")]],100) %>% names
sample(trump_words,10)

g7 = model[rownames(model) %in% trump_words [1:50],]

group_distances7 = cosineDist(g7,g7) %>% as.dist
plot(as.dendrogram(hclust(group_distances7)),cex=1, main="Cluster dendrogram of the fifty words closest to a 'trump' vector\nin BM Data")



#gender?

## get 50 close words to the 'she' vector
nearest_to(model,model[["herself"]])
she_words = model %>% nearest_to(model[[c("she","her")]],100) %>% names
sample(she_words,50)
g = model[rownames(model) %in% she_words [1:50],]
group_distances = cosineDist(g,g) %>% as.dist
plot(as.dendrogram(hclust(group_distances)),horiz=F,cex=1,main="Cluster dendrogram of the fifty words closest to a 'she' vector\nin BM data")


## get 50 close words to the 'he' vector

nearest_to(model,model[["himself"]])
he_words = model %>% nearest_to(model[[c("he","his","him","himself")]],100) %>% names
sample(he_words,50)

g2 = model[rownames(model) %in% he_words [1:50],]

group_distances = cosineDist(g2,g2) %>% as.dist
plot(as.dendrogram(hclust(group_distances)),horiz=F,cex=1,main="Cluster dendrogram of the fifty words closest to a 'he' vector\nin BM data ")

## get 50 close words to the "i" vecotr

nearest_to(model,model[["mine"]])
i_words = model %>% nearest_to(model[[c("i","me")]],100) %>% names
sample(i_words,50)

g3 = model[rownames(model) %in% i_words [1:50],]

group_distances = cosineDist(g3,g3) %>% as.dist
plot(as.dendrogram(hclust(group_distances)),horiz=F,cex=1,main="Cluster dendrogram of the fifty words closest to an 'I' vector\nin BM Data")


###some tsne plots
install.packages("tsne")
plot(model)

some_groups = nearest_to(model,model[[c("wall")]],75)
plot(filter_to_rownames(model,names(some_groups)))

dev.off()


##wordscores
word_scores = data.frame(word=rownames(model))

##crossplots


##check that words exist in the model
nearest_to(model,model[["US"]])
nearest_to(model,model[["Mexico"]])
#####good/bad, skull/bone

## uk_vector = model[[c("good","better","best")]]-model[[c("bad")]]

uk_vector = model[[c("England")]]-model[[c("Scotland")]]
us_vector = model[[c("US")]] - model[[c("Mexico")]]

word_scores$us_score = model %>% cosineSimilarity(us_vector) %>% as.vector
word_scores$uk_score = cosineSimilarity(model,uk_vector) %>% as.vector

groups = c("us_score","uk_score")

word_scores %>% mutate( usedness=ifelse(us_score>0,"US","Mexico"),uk=ifelse(uk_score>0,"England","Scotland")) %>% group_by(uk,usedness) %>% filter(rank(-(abs(us_score*uk_score)))<=36) %>% mutate(eval=-1+rank(abs(uk_score)/abs(us_score))) %>% ggplot() + geom_text(aes(x=eval %/% 12,y=eval%%12,label=word,fontface=ifelse(usedness=="US",2,3),color=uk),hjust=0) + facet_grid(uk~usedness) + theme_minimal() + scale_x_continuous("",lim=c(0,3)) + scale_y_continuous("") + labs(title=" Scotland (red) and England (blue) words \nused to describe Mexcio (italics) and US (bold)") + theme(legend.position="none")


####binaries

library(ggplot2)

word_scores$us_score = model %>% cosineSimilarity(us_vector) %>% as.vector

ggplot(word_scores %>% filter(abs(us_score)>.40)) + geom_bar(aes(y=us_score,x=reorder(word,us_score),fill=us_score<0),stat="identity") + coord_flip()+scale_fill_discrete("Indicative of",labels=c("us","mexi")) + labs(title="The words showing the strongest skew along \n the continuum from 'us' to 'mexico''")


