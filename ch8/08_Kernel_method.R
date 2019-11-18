## 8. Kernel methods


#  Linear : <x, x`> --> 벡터의 내적표현
#  



library(e1071)

data(Glass, package = "mlbench")
str(Glass)


train_index <- sample(1:nrow(Glass), round(nrow(Glass)*0.7))
GlassTrain <- Glass[train_index,]
GlassTest <- Glass[-train_index,]

# SVM 학습.
tobj <- tune.svm(Type ~ . , data = GlassTrain,
                 gamma = 10^(-2:2), cost = 10^(-2:2))  # 얘네들을 조절을 잘 해야함.


best.gamma <- tobj$best.parameters[[1]]
best.cost <- tobj$best.parameters[[2]]
best.gamma; best.cost

#--> 사용하는 기본 함수는 radial basis를 사용.
svm.model <- svm(Type ~ ., data = GlassTrain,
                 cost = best.cost, gamma = best.gamma)


pred <- predict(svm.model, GlassTest[,-10])
table(pred, GlassTest$Type)
