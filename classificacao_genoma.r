library(C50)
library(caret)

test <- read.csv('./ml-small-tratado/classificacao_genoma_test.csv')
train <- read.csv('./ml-small-tratado/classificacao_genoma_train.csv')
train <- train[, c(-3)]
test <- test[, c(-3)]

train$userId <- as.factor(train$userId)
test$userId <- as.factor(test$userId)

# train.X <-train[, c(1, 2, 4:1131)]
train.X <-train[, c(1, 4:503)]
train.y <- train[, 3]
# test.X <- test[, c(1, 2, 4:1131)]
test.X <- test[, c(1, 4:503)]
test.y <- test[, 3]


model <- C50::C5.0(train.X, train.y)
summary(model)

pred <- predict(model, test.X)
confusionMatrix(data=pred, reference=test.y, positive = 'True')


