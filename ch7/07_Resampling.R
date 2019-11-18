##########################################################################################################
#################################Resampling Methods#########################################################################
##########################################################################################################

## 별도) Resampling Methods

## Resampling의 이유는 모형 평가, 모형 선택을 하기 위함.


#  Validation Set Approach
#  --> validation set을 만든다.
#  --> test, training, validation : parameter를 최적으로 훈련을 시키고 제대로 test를 해봐야함.

library(ISLR)
help(Auto)
str(Auto)


plot(mpg ~ horsepower, data = Auto, cex = 0.5, pch = 19)

#  다음의 다항회귀(polynomial regression) 모형을 고려 :
#  mpg ~ horsepower
#  mpg ~ horsepower + horsepower^2
#  mpg ~ horsepower + horsepower^2 + … + horsepower^p

#  Sampling
set.seed(0)
N <- nrow(Auto)
tr <- sample(1:N, round(N/2), replace = FALSE)
Auto.tr <- Auto[tr,]
Auto.val <- Auto[-tr,]

mse <- rep(NA, 10)
for ( p in 1:10 ) {
  fit <- lm(mpg ~ poly(horsepower, p), data = Auto.tr)  # poly가 알아서 다 써줌.
  mse[p] <- mean((predict(fit, Auto.val) - Auto.val$mpg)^2)
}

plot(1:10, mse
     , type = "b"  # 선과 점을 둘 다 사용하겠다는 의미
     , pch = 19    #  교수님이 사랑하는 점 스타일
     , col = 2, ylim = c(12, 32),
     xlab = "Degree of polynomial", ylab = "Mean Squared Error")
#  --> 한 번만 샘플링해서 결정하기는 어렵다.




#  10번 샘플링해서 확인
plot(1:10, mse, type = "l", lwd = 3, ylim = c(12, 32),
     xlab = "Degree of polynomial", ylab = "Mean Squared Error")
for ( j in 2:10 ) {
  tr <- sample(1:N, round(N/2), replace = FALSE)
  Auto.tr <- Auto[tr,]
  Auto.val <- Auto[-tr,]
  mse <- rep(NA, 10)
  for ( p in 1:10 ) {
    fit <- lm(mpg ~ poly(horsepower, p), data = Auto.tr)
    mse[p] <- mean((predict(fit, Auto.val) - Auto.val$mpg)^2)
  }
  lines(1:10, mse, col = j, lwd = 3)
}
#  여기서는 2차가 최선이라고 생각되지만 보통의 데이터는 충분치 않을수 있다.




#  Validation Set Approach의 장단점
#  장점 : 
#    단순하다
#    실행이 쉽다
#  단점 : 
#    평가자료 MSE가 분할결과에 따라 변동이 심하다 ⇒ test MSE의 추정치로 사용이 부적당함
#  사용가능한 자료의 일부분만 훈련자료(training data)로 이용 ⇒ 통계적 방법론은 적은 수의 자료를 이용하는 경우 성능이 저하











## Leave-One-Out Cross Validation (LOOCV)

mse <- rep(0, 10)
for ( p in 1:10 ) {
  for ( i in 1:N ) {
    tr <- Auto[-i,]
    val <- Auto[i,]
    fit <- lm(mpg ~ poly(horsepower, p), data = tr)
    mse[p] <- mse[p] + (predict(fit, val) - tr$mpg)^2
  }
}
mse <- mse/N

plot(1:10, mse, type = "l", lwd = 3, main = "LOOCV", #ylim = c(12, 32)
     xlab = "Degree of polynomial", ylab = "Mean squared error")





## K-fold Cross Validation
#  10-fold를 10번 반복하는 코드
set.seed(0)

K <- 10
M <- round(N/K)
V <- matrix(0, K, 2)
for ( k in 1:K ) {
  V[k,] <- c(M*(k - 1) + 1, min(M*k, N))
}

ind <- sample(1:N)
mse <- matrix(NA, K, 10)
for ( k in 1:K ) {
  val <- ind[V[k, 1]:V[k, 2]]
  tr <- ind[-(V[k, 1]:V[k, 2])]
  for ( p in 1:10 ) {
    fit <- lm(mpg ~ poly(horsepower, p), data = Auto, subset = tr)
    mse[k, p] <- mean((predict(fit, Auto[val,]) - Auto$mpg[val])^2)
  }
}
CV <- colMeans(mse)
print((1:10)[which.min(CV)])

plot(1:10, CV, type = "l", lwd = 3, ylim = c(12, 32), 
     main = paste(k, "-fold CV", sep = ""),
     xlab = "Degree of polynomial", ylab = "Mean squared error")

for ( j in 2:10 ) {
  ind <- sample(1:N)
  mse <- matrix(NA, K, 10)
  for ( k in 1:K ) {
    val <- ind[V[k, 1]:V[k, 2]]
    tr <- ind[-(V[k, 1]:V[k, 2])]
    for ( p in 1:10 ) {
      fit <- lm(mpg ~ poly(horsepower, p), data = Auto, subset = tr)
      mse[k, p] <- mean((predict(fit, Auto[val,]) - Auto$mpg[val])^2)
    }
  }
  CV <- colMeans(mse)
  print((1:10)[which.min(CV)])
  lines(1:10, CV, lwd = 3, col = j)
}

#  package 사용
library(boot)
set.seed(0)
cv <- rep(NA, 10)
for ( p in 1:10 ) {
  fit <- glm(mpg ~ poly(horsepower, p), data = Auto)  # cv.glm으로 하기 위해 glm을 사용. family = "binomial"일때만 로지스틱, 디폴트는 가우시안.
  cv[p] <- cv.glm(Auto, fit, K = 10)$delta[1]  # K를 지정하지 않으면 LOOCV
}
plot(1:10, cv, type = "b", col = 2, pch = 19, 
     xlab = "Degree of polynomial", ylab = "MSE")













## 분류문제에 대한 Cross Validation의 적용
cv.glmnet()  # --?알파를 정하는 elastic net을 의미?









library(ISLR)
plot(Portfolio, col = "darkgreen", pch = 20)


MVP <- function(X, Y) (var(Y) - cov(X, Y))/(var(X) + var(Y) - 2*cov(X, Y))
print(alpha <- MVP(Portfolio$X, Portfolio$Y))


set.seed(0)

B <- 5000
alpha.boot <- rep(NA, B)
N <- nrow(Portfolio)

for ( b in 1:B ) {
  Portfolio.boot <- Portfolio[sample(1:N, N, replace = TRUE),] 
  alpha.boot[b] <- with(Portfolio.boot, MVP(X, Y))
}

hist(alpha.boot, probability = TRUE, xlab = expression(alpha), 
     col = "lightgray", border = "white", nclass = 100,
     main = "Bootstrap distribution")
abline(v = alpha, col = 2)


