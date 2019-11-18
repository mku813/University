## Regularization

#install.packages("glmnet")
library(glmnet)


set.seed(0)
# # of rows < # of variables
n <- 50
p <- 100

sigma <- 0.25
beta <- c(1:5, rep(0, p - 5))

x <- cbind(matrix(runif(n*p), n, p))
y <- 10 + as.vector(x%*%beta) + sigma*rnorm(n)
# y = 10 + 1*x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5 + 0*x_i + e, e ~ N(0, 0.25^2) , 5 < i <= 100
# --> 6 ~ 100번째 변수들은 y와 무관


# modeling
fit <- glmnet(x, y)       # 이때, alpha는 1로 LASSO가 default이다.
plot(fit, xvar = "lambda", label = TRUE)    # 변수 중요도 확인. default : xvar = L1-norm 



# 자동 모형 선택
# --> 자동으로 교차검증을 해줌.
cvfit <- cv.glmnet(x, y)  # family = "binomial"이라는 파라메터를 추가하면 로지스틱 모형으로 적합. 
plot(cvfit)               # 교차검증 결과
# --> 왼쪽으로 갈수록 복잡한 모형 : 모든 변수를 포함
# --> 각 Lambda값에서 교차 검증 이후에 나온 결과를 error-bar로 나타낸 것이다. 빨간색은 평균



print(s <- cvfit$lambda.min)
print(beta.hat <- coef(cvfit, s = "lambda.min"))
length(which(beta.hat > 0)) # 8개의 변수 선택!
# 교차겁증 오차의 평균값을 최소화하는 것이 lambda.min.
# 최적의 예측력을 위해서 lambda.min를 사용한다.


pred.lasso <- predict(fit, newx = x, s = s)
plot(y, pred.lasso, xlab = "Actual", ylab = "Predicted")
abline(0, 1, col = 2, lty = 2)





## Elastic-Net
fit <- glmnet(x, y, alpha = .5)            # alpha = 
plot(fit, xvar = "lambda", label = TRUE)


cvfit <- cv.glmnet(x, y, alpha = .5)
plot(cvfit)



print(cvfit$lambda.min)
print(s <- cvfit$lambda.min)

plot(y, pred.lasso, xlab = "Actual", ylab = "Predicted")
abline(0, 1, col = 2, lty = 2)
points(y, predict(fit, newx = x, s = s), col = 3, pch = "E")




## Non-normal y

data(BinomialExample)

fit <- glm(y ~ x, family = binomial("logit"))
summary(fit)




fit <- glmnet(x, y, family = "binomial")
plot(fit, xvar = "lambda", label = TRUE)




cvfit <- cv.glmnet(x, y, alpha = .5)
plot(cvfit)






print(cvfit$lambda.min)





print(s <- cvfit$lambda.min)





pred <- predict(fit, newx = x, type = "class", s = s)
table(y, pred)












