## 9. Ensemble methods

## 9.1 Ensemble method: Bagging(or bootstrap aggregation)
loglikelihood <- function(y, py) {
  pysmooth <- ifelse(py <= 1e-12, 1e-12, 
                     ifelse(py >= 1 - 1e-12, 1 - 1e-12, py))  # py가 너무 0,1이거나 가까우면 조정해줌. --> 좋은 것은 아님.
                                                              # log를 취할 때, 에러를 막기 위해서
  sum(y*log(pysmooth) + (1 - y)*log(1 - pysmooth)) # pysmooth == theta : Cross-Entropy
}

accuracyMeasures <- function(pred, truth, name = "model") {
  dev.norm <- -2*loglikelihood(as.numeric(truth), pred)/length(pred) # like Cross-Entropy
  ctable <- table(truth = truth,
                  pred = (pred > 0.5))  # cutoff = 0.5 // Confusion Matrix
  # --> assemble은 cut-off에 크게 영향받지 않는다. : 그래도 어느정도 잘 찾아주는게 좋지 않을까?
  print(ctable)
  accuracy <- sum(diag(ctable))/sum(ctable)
  precision <- ctable[2,2]/sum(ctable[,2])
  recall <- ctable[2,2]/sum(ctable[2,])
  f1 <- 2*precision*recall/(precision + recall)
  data.frame(model = name, accuracy = accuracy, f1 = f1, dev.norm)
}


#  Decision tree
spamD <- read.table('https://raw.githubusercontent.com/WinVector/zmPDSwR/master/Spambase/spamD.tsv', header = TRUE, sep = '\t')

spamTrain <- subset(spamD, spamD$rgroup >= 10)
spamTest <- subset(spamD, spamD$rgroup < 10)

spamVars <- setdiff(colnames(spamD), list('rgroup', 'spam'))  # rgroup, spam 컬럼을 제외함.(차집합)
spamFormula <- as.formula(paste('spam == "spam"', 
                                paste(spamVars, collapse=' + '), 
                                sep=' ~ '))

library(rpart)

treemodel <- rpart(spamFormula, spamTrain) 
# Recursive Partitioning and Regression Trees

plot(treemodel)
text(treemodel)


accuracyMeasures(predict(treemodel, newdata = spamTrain), 
                 spamTrain$spam == "spam", 
                 name = "tree, training") 

accuracyMeasures(predict(treemodel, newdata = spamTest),
                 spamTest$spam == "spam", 
                 name = "tree, test")




#  Bagging
ntrain <- dim(spamTrain)[1]
n <- ntrain
ntree <- 100 
samples <- sapply(1:ntree, 
                  FUN = function(iter){
                    sample(1:ntrain, size = n, replace = TRUE)
                  }
)
#--> take about 1 minute
treelist <- lapply(1:ntree, 
                   FUN = function(iter){
                     samp <- samples[,iter]
                     rpart(spamFormula, spamTrain[samp,])
                   }
)

predict.bag <- function(treelist, newdata) {
  preds <- sapply(1:length(treelist), 
                  FUN = function(iter) {
                    predict(treelist[[iter]], newdata = newdata)
                  }
  )
  predsums <- rowSums(preds)
  predsums/length(treelist)
}

accuracyMeasures(predict.bag(treelist, newdata = spamTrain), 
                 spamTrain$spam == "spam", 
                 name = "bagging, training")



accuracyMeasures(predict.bag(treelist, newdata = spamTest), 
                 spamTest$spam == "spam",
                 name = "bagging, test")





## 9.2 Ensemble method: Random forest
#--> Classification) sqrt(변수의 수) 만큼의 변수를 선택한다.
#--> Regression) 변수의 수/3 만큼의 변수를 선택한다.

library(randomForest)

set.seed(5123512)
fmodel <- randomForest(x = spamTrain[,spamVars],
                       y = spamTrain$spam, 
                       ntree = 100,
                       nodesize = 7,   # 트리의 깊이.
                       # mtry = ?      # 변수의 갯수를 선택할 것을 정함.
                       importance = TRUE)

accuracyMeasures(predict(fmodel,    
                         newdata = spamTrain[,spamVars],
                         type = 'prob')[,'spam'],   
                 spamTrain$spam == "spam",
                 name = "random forest, train")



accuracyMeasures(predict(fmodel,
                         newdata = spamTest[,spamVars],
                         type = 'prob')[,'spam'],
                 spamTest$spam == "spam",
                 name = "random forest, test")

varImpPlot(fmodel, type = 1)  # type = 1 : Accuracy 기준
varImpPlot(fmodel, type = 2)  # type = 2 : Gini Impurity 기준



varImp <- importance(fmodel, type = 2)
selVars <- names(sort(varImp[,1], decreasing = TRUE))[1:25]  # 상위 변수 선택
fsel <- randomForest(x = spamTrain[,selVars],
                     y = spamTrain$spam,
                     ntree = 100,
                     nodesize = 7, 
                     importance = TRUE)
accuracyMeasures(predict(fsel,
                         newdata = spamTrain[,selVars], 
                         type = 'prob')[,'spam'],
                 spamTrain$spam == "spam",
                 name = "RF small, train")


accuracyMeasures(predict(fsel,
                         newdata = spamTest[,selVars],
                         type = 'prob')[,'spam'],
                 spamTest$spam == "spam",
                 name = "RF small, test")







## 9.3 Ensemble method: Boosting
## 9.3.1 Adaboost
n.tr <- nrow(spamTrain)
n.test <- nrow(spamTest)
w <- rep(1, n.tr)/n.tr
M <- 20
G.tr <- matrix(NA, M, n.tr)
G.test <- matrix(NA, M, n.test)
alpha <- rep(NA, M)
for ( m in 1:M ) {
  treemodel <- rpart(spamFormula, data = spamTrain, weights = w)
  pred.tr <- predict(treemodel, newdata = spamTrain) >= 0.5
  pred.test <- predict(treemodel, newdata = spamTest) >= 0.5
  p.tr <- rep("non-spam", n.tr)
  p.test <- rep("non-spam", n.test)
  p.tr[pred.tr] <- "spam"
  p.test[pred.test] <- "spam"
  G.tr[m, ] <- p.tr
  G.test[m, ] <- p.test
  err <- sum(w[p.tr != spamTrain$spam])/sum(w)
  alpha[m] <- .5*log(min(1 - err, 1 - 1e-6)/max(err, 1e-6))
  w[p.tr == spamTrain$spam] <- w[p.tr == spamTrain$spam]*exp(-alpha[m])
  w[p.tr != spamTrain$spam] <- w[p.tr != spamTrain$spam]*exp(alpha[m])
  w <- w/sum(w)
}
print(alpha)


alpha.tr <- matrix(alpha, M, n.tr, byrow = TRUE)
alpha.test <- matrix(alpha, M, n.test, byrow = TRUE)

accuracyMeasures(colSums(alpha.tr*(2*(G.tr == "spam") - 1)) > 0,   
                 spamTrain$spam == "spam",
                 name = "adaboost, train")

accuracyMeasures(colSums(alpha.test*(2*(G.test == "spam") - 1)) > 0,   
                 spamTest$spam == "spam",
                 name = "adaboost, test")




## 9.3.2 Functional gradient boosting algorithm



library(gbm)
#--> 굉장히 계산량이 많아서 조심해야함.
boostedTree <- gbm(spamFormula, data = spamTrain, 
                   distribution = "bernoulli", 
                   cv.folds = 5, n.trees = 5000)

best.iter <- gbm.perf(boostedTree, method = "cv") 









## 9.3.3 XGBoost(Extreme gradient boosting)
#--> Gradient boosting 의 속도 문제를 병렬 처리를 통해 획기적으로 개선
#--> 결측치 처리, 데이터 imbalance 문제, variable importance 문제 해결
library(xgboost)

#  Training
spamTrainLabel <- 1*(spamTrain$spam == "spam")   # I()로 해줘야함.
XTrain <- data.matrix(spamTrain[, spamVars])
DTrain <- xgb.DMatrix(data = XTrain, label = spamTrainLabel)  

#  Test
spamTestLabel <- 1*(spamTest$spam == "spam")
XTest <- data.matrix(spamTest[, spamVars])
DTest <- xgb.DMatrix(data = XTest, label = spamTestLabel)

#  make list into train/test set
#--> for stopping error
watchlist <- list(train = DTrain, test = DTest)


bst <- xgb.train(data = DTrain,
                 nrounds = 3,     # number of running time
                 watchlist = watchlist,
                 objective = "binary:logistic") # 학습모델
#--> multi class를 classification하려면 클래스가 몇 개 인지 지정해야함.




bst <- xgb.train(data = DTrain,
#                nthread = 3,   # 병렬처리를 위한 core활용 수
                 max_depth = 10, nrounds = 10000, 
                 watchlist = watchlist,
                 early_stopping_rounds = 30,   # 개선이 없으면 30번에서 끝내라
                 maximize = FALSE,
                 print_every_n = 10,  # 10번에 한 번씩만 결과를 출력해라.
                 scale_pos_weight = sum(spamTrainLabel == 0)/sum(spamTrainLabel == 1), # 해줘도 되고 안해줘되 되는데 상황에 따라 다르다. 
                 objective = "binary:logistic")


pred <- predict(bst, XTrain)
accuracyMeasures(pred,   
                 spamTrain$spam == "spam",
                 name = "XGboost, train")

pred <- predict(bst, XTest)
accuracyMeasures(pred,   
                 spamTest$spam == "spam",
                 name = "XGboost, test")




importance <- xgb.importance(feature_names = spamVars, model = bst)
head(importance, n = 10)


xgb.plot.importance(importance_matrix = importance)









#  Random Forest와 결합
bst <- xgb.train(data = DTrain,
                 max_depth = 10, nrounds = 10000, 
                 watchlist = watchlist,
                 early_stopping_rounds = 30, maximize = FALSE,
                 print_every_n = 10,
                 subsample = 0.5,  # row방향으로 샘플링할 때의 비율
                 colsample_bytree = 0.5, # 변수 선택하는 비율
                 num_parallel_tree = 1000,
                 scale_pos_weight = sum(spamTrainLabel == 0)/sum(spamTrainLabel == 1),
                 objective = "binary:logistic")

pred <- predict(bst, XTrain)
accuracyMeasures(pred,   
                 spamTrain$spam == "spam",
                 name = "XGboost_RF, train")


pred <- predict(bst, XTest)
accuracyMeasures(pred,   
                 spamTest$spam == "spam",
                 name = "XGboost_RF, test")

importance <- xgb.importance(feature_names = spamVars, model = bst)
head(importance, n = 10)

xgb.plot.importance(importance_matrix = importance)



