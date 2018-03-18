setwd("C:/D/Phyllis/NWU/Course/2018 Winter/MSiA 493 Analytics Value Chain/Project/Project_important/team9_projectC/development/data")
train <- read.csv('train.csv')

train$binary <- apply(train[,c(3:8)],1,sum)
sum(train$binary == 1)
train_new <- train[train$binary==1,]

set.seed(123)
train_non <- train[train$binary==0,]
smp_size <- floor((1/3) * nrow(train_non))
train_ind <- sample(seq_len(nrow(train_non)), size = smp_size)

train_new <- c(train_new, train[train_ind]
test <- train[-train_ind, ]
write.csv(train, file = "train_sub.csv")
