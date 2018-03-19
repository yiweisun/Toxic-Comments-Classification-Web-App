setwd("C:/D/Phyllis/NWU/Course/2018 Winter/MSiA 493 Analytics Value Chain/AWS/msia423_project/development/data")
train <- read.csv('train.csv')

train$binary <- apply(train[,c(3:8)],1,sum)
sum(train$binary == 1)
train_new <- train[train$binary==1,]

set.seed(123)
train_non <- train[train$binary==0,]
train_size <- floor((1/28) * nrow(train_non))
test_size <- floor((1/3)*nrow(train_new))
train_ind <- sample(seq_len(nrow(train_non)), size = train_size)
test_ind <- sample(seq_len(nrow(train_new)), size = test_size)

train_sub <- rbind(test_new[test_ind,], train_non[train_ind,])
write.csv(train_sub, file = "train_sub.csv")
