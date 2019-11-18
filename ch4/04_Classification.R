library(rpart)
library(ROCR)
library(e1071)
#install.packages("mlbench")
library(mlbench)


## 4.1 이항 분류 문제
#  Accuracy, 정확도 : 얼마나 잘 맞추느냐
#  Precision, 예측도 : 비용문제를 다룰 때, 주로 사용. 
#  Recall, 복원력(== Sensitivity) : 부도날 확률을 구할 때 사용
#  Specificity, 특이도 : 
#  다 높아야함. but, 특이도와 복원력이 서로 상충됨.
#  cut-off value가 높아지면 specificity는 높아지고, recall은 낮아짐.
#  cut-off value가 낮아지면 specificity는 낮아지고, recall은 높아짐.


#  조화평균
harmonic.mean <- function(a, b) 2*a*b/(a+b)

#  
accuracyMeasures <- function(pred, truth, cutoff = 0.5, name = "model") {
  ctable <- table(Actual = truth,
                  Predicted = 1*(pred >= cutoff))
  accuracy <- sum(diag(ctable))/sum(ctable)
  precision <- ctable[2,2]/sum(ctable[,2])
  recall <- ctable[2,2]/sum(ctable[2,])
  f1 <- harmonic.mean(precision, recall)
  specificity <- ctable[1, 1]/sum(ctable[1,])
  list(model = name,
       accuracy = accuracy,
       specificity = specificity,
       sensitivity = recall,
       recall = recall,
       precision = precision,
       f1.score = f1,
       confusionMatrix = ctable)
}



#  Moderate threshold일 때, 최적의 cut-off value
#--> (0,1)이면 신탁임.
#--> ROC curve 밑의 넓이를 AUC라고 한다.(Area Under Curve)







## 4.2 로지스틱 이항분류 logistic classification
#  logit function = log(p/(1-p))
#  불균형 자료(unbalanced data) : 1과 0의 비율이 극히 차이나는 경우
#--> cut-off를 엄청 낮춰서 해야함.
library(data.table)
adult <- read.csv(file = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header = F)
names(adult) <- c("age", "workclass", "fnlwgt", "education", "education_num",
                  "marital_status", "occupation", "relationship", "race", "sex",
                  "capital_gain", "capital_loss", "hours_per_week",
                  "native_country", "wage")

head(adult) # wage가 target


#  data split
set.seed(0)
N <- nrow(adult)
n.tr <- round(0.7*N)
tr <- sort(sample(1:N, n.tr, replace = FALSE))
adult.tr <- adult[tr,]
adult.te <- adult[-tr,]

#  model fitting
glm.fit <- glm(wage ~ ., data = adult.tr, family = binomial)

#  predict
Actual <- 1*(adult.te$wage == " >50K")
Pr.glm <- predict(glm.fit, newdata = adult.te, type = "response") # must type = "response" 안하면 logit으로 나옴.

#  model feature
accuracyMeasures(pred = Pr.glm, truth = Actual, cutoff = 0.5, 
                 name = "Logistic with cutoff = 0.5") 


#  ROC curve
library(ROCR)
Pred.glm <- prediction(Pr.glm, Actual)
Perf.glm <- performance(Pred.glm, measure = "tpr", x.measure = "fpr")
plot(Perf.glm, main = "ROC curve")
abline(0, 1, lty = 2)

# AUC
performance(Pred.glm, "auc")@y.values[[1]]









###################################################################################################
## 04/10
## Best Cut-off value 구하기
install.packages("fmsb")
library(fmsb)

roc.obj <- roc(Pr.glm, Actual, maxdist = FALSE)
best.cutoff <- roc.obj$cutoff[which.min(roc.obj$distance)]  # distance는 (0,1)점에서 커브와의 거리 중 가장 가까운 거리
best.specificity <- 1 - roc.obj$falsepos[which.min(roc.obj$distance)]
best.sensitivity <- roc.obj$sens[which.min(roc.obj$distance)]

plot(roc.obj)
points(1 - best.specificity, best.sensitivity, pch = 19, col = "navy")
text(1 - best.specificity + 0.24, best.sensitivity, 
     paste("Cut-off = ", round(best.cutoff, 4), sep = ""))
text(0.5, 0.5, 
     paste("AUC = ", round(sum(roc.obj$aucpiece), 4), sep = ""))


accuracyMeasures(pred = Pr.glm, truth = Actual, cutoff = best.cutoff, 
                 name = "Logistic with the best cutoff")


boxplot(Pr.glm ~ adult.te$wage, 
        xlab = "Predicted probability",
        ylab = "Actual wage (test dataset)",
        horizontal = TRUE, col = "lightgray")
abline(v = best.cutoff, col = 2, lty = 2)









## 4.3 의사결정나무 decision tree

adult <- read.csv(choose.files())

set.seed(0)
N <- nrow(adult)
n.tr <- round(0.7*N)
tr <- sort(sample(1:N, n.tr, replace = FALSE))
adult.tr <- adult[tr,]
adult.te <- adult[-tr,]


library(rpart)

tree.fit <- rpart(wage ~ ., data = adult.tr)

plot(tree.fit)
text(tree.fit)  

Actual <- 1*(adult.te$wage == " >50K")
Pr.tree <- predict(tree.fit, newdata = adult.te)[,2]
Predicted.tree <-  1*(Pr.tree >= 0.5)
table(Actual, Predicted.tree)




library(ROCR)

Pred.tree <- prediction(Pr.tree, Actual)
Perf.tree <- performance(Pred.tree, measure = "tpr", x.measure = "fpr")
plot(Perf.tree, main = "ROC curve", cex.axis = 0.7)
plot(Perf.glm, col = "red", add = TRUE)
abline(0, 1, lty = 2)
legend("bottomright", inset = .1, cex = 0.7,
       legend = c("Logistic Regression", "Decision Tree"), 
       col = c("red", "black"), lty = c(1, 1))

performance(Pred.tree, "auc")@y.values[[1]]








## 4.4 나이브베이즈 naive Bayes
#  베이지안 룰로 하는 것인데, 데이터를 집어 넣었을 때, 어떤 클래스가 가장 높은 확률을 갖는다면 그 클래스로 분류한다.
library(e1071)

data(HouseVotes84, package = "mlbench")
summary(HouseVotes84)

NB <- naiveBayes(Class ~ ., data = HouseVotes84)
predict(NB, newdata = HouseVotes84[1:5,])
predict(NB, newdata = HouseVotes84[1:5,], type = "raw")


pred <- predict(NB, HouseVotes84)
table(pred, HouseVotes84$Class)




data(Glass, package = "mlbench")
str(Glass)

NB <- naiveBayes(Type ~ ., data = Glass)
predict(NB, newdata = Glass[1:5,])
predict(NB, newdata = Glass[1:5,], type = "raw") # 분류 확률

pred <- predict(NB, Glass)
table(pred, Glass$Type)

sum(diag(table(pred, Glass$Type)))/sum(table(pred, Glass$Type))




## 4.5 신경망 모형 neural network
install.packages("neuralnet")
library(neuralnet)
data(airquality)
dat <- na.omit(airquality[,1:4])
dat <- scale(dat)
pairs(dat[,1:4])


nnfit <- neuralnet(Ozone ~ Solar.R + Wind + Temp, data = dat,
                   hidden = c(5, 5), linear.output = TRUE)
pred <- compute(nnfit, dat[,2:4])$net.result

X <- scale(dat[,2:4])
Y <- dat[,1]
pred.lm <- predict(lm(Y ~ X))
p <- c(pred, pred.lm)
par(mfrow = c(1,2))
plot(dat[,1], pred, xlab = "Actual", ylab = "Predicted",
     xlim = range(p), ylim = range(p), main = "Neural Network",
     sub = paste("RMSE = ", round(sqrt(mean((pred - Y)^2)), 2), sep = ""))
abline(0, 1, col = 2, lty = 2)
plot(dat[,1], pred.lm, xlab = "Actual", ylab = "Predicted",
     xlim = range(p), ylim = range(p), main = "Multiple Linear Regression",
     sub = paste("RMSE = ", round(sqrt(mean((pred.lm - Y)^2)), 2), sep = ""))
abline(0, 1, col = 2, lty = 2)





## 4.5.2 신경망 모형의 기본 개념

## w^(1)_12 : 1번째 층에서 두번째 입력노드에서 1번째 히든 노드로 가는 가중치를 의미.
## 



## 함수 근사
f <- function(x) 0.25*x^2
x <- seq(from = -2, to = 2, by = 0.005)
y <- f(x)
dat <- data.frame(x, y)

library(neuralnet)

set.seed(0)
h <- 3
NN <- neuralnet(y ~ x, data = dat,                 # 이 부분은 잘 알아야함.
                hidden = h, act.fct = "logistic",  # logistic이 sigmoid 함수임. relu 쓰고 싶으면 Relu쓰면 됨.
                linear.output = TRUE)              # 회귀 문제일 때 T, 분류 문제면 F
p <- compute(NN, data.frame(x))

plot(x, y, type = "l", lwd = 2,
     cex.lab = 0.8, cex.axis = 0.7)
for ( k in 2:(h+1) )
  lines(x, p$neurons[[2]][,k], col = (k+1), lty = 2)
grid <- seq(from = 1, to = length(x), length = 9)
points(x[grid], p$net.result[grid], col = 2, pch = 19)





#  예제
library(neuralnet)

dat <- na.omit(airquality[,1:4])
dat <- scale(dat)                # 표준화는 꼭 해줘야 좋음.
pairs(dat[,1:4])


#  모델식
nnfit <- neuralnet(Ozone ~ Solar.R + Wind + Temp, data = dat,  # 중간 은닉층에 대한 활성함수를 지정 안하면 default가 sigmoid함수임.
                   hidden = c(5, 5), linear.output = TRUE)     # (5, 5) : 은닉층을 2개를 쌓겠다. 노드 수는 5개씩
pred <- neuralnet::compute(nnfit, dat[,2:4])$net.result



X <- scale(dat[,2:4])
Y <- dat[,1]
pred.lm <- predict(lm(Y ~ X))
p <- c(pred, pred.lm)
par(mfrow = c(1,2))
plot(dat[,1], pred, xlab = "Actual", ylab = "Predicted",
     xlim = range(p), ylim = range(p), main = "Neural Network",
     sub = paste("RMSE = ", round(sqrt(mean((pred - Y)^2)), 2), sep = ""))
abline(0, 1, col = 2, lty = 2)
plot(dat[,1], pred.lm, xlab = "Actual", ylab = "Predicted",
     xlim = range(p), ylim = range(p), main = "Multiple Linear Regression",
     sub = paste("RMSE = ", round(sqrt(mean((pred.lm - Y)^2)), 2), sep = ""))
abline(0, 1, col = 2, lty = 2)





#  예제 2
data(infert)
summary(infert)

nn.infert <- neuralnet(case ~ parity + induced + spontaneous, 
                       data = infert, err.fct = "ce", hidden = c(7),  # err.fct == error function : cross-entropy / default는 mse
                       linear.output = FALSE)


x <- infert[,c("parity", "induced", "spontaneous")]
pred <- compute(nn.infert, x)$net.result 
boxplot(pred ~ infert$case, col = "lightgray", horizontal = TRUE,
        xlab = "Predicted probability", ylab = "Actual case")
abline(v = 0.5, col = 2, lty = 2)


Predicted <- 1*(pred >= 0.5)      # cut-off value가 0.5
table(Predicted, Actual = infert$case)

















