library(C50)

test <- read.csv('./ml-small-tratado/classificacao_genoma_test.csv')
train <- read.csv('./ml-small-tratado/classificacao_genoma_train.csv')

#retira colunas nao utilizadas
train <- train[ -c(2, 4)]
test <- test[ -c(2, 4)]

#limita aos 500 primeiros tags
train<- train[, c(1:502)]
test<- test[, c(1:502)]

train$userId <- as.factor(train$userId)
test$userId <- as.factor(test$userId)

model <- rpart::rpart(rating ~ ., data=train,method='anova')
summary(model)

pred <- predict(model, test[-2])

print("Mean squared error")
mean((test$rating - pred)^2)