library(C50)
library(caret)

test <- read.csv('./ml-small-tratado/classificacao_genero_test.csv')
train <- read.csv('./ml-small-tratado/classificacao_genero_train.csv')

train$userId <- as.factor(train$userId)
test$userId <- as.factor(test$userId)

train.X <-train[, c(1, 4:23)]
train.y <- train[, 3]
test.X <- test[, c(1, 4:23)]
test.y <- test[, 3]


model <- C50::C5.0(train.X, train.y)
summary(model)

pred <- predict(model, test.X)
confusionMatrix(data=pred, reference=test.y, positive = 'True')


