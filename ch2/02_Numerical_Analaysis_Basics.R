## 2019.03.08


## 2.1 행렬 다루기
## 2.1.1. 반복실행
set.seed(123457)
N <- 1000  # 행의 갯수
d <- 100   # 열의 갯수
X <- matrix(rnorm(N*d, mean = 3, sd = 10), N, d)

#  평균 벡터 만들기(행 & 열)
head(apply(X, 1, mean)) # 행 방향으로 평균 계산 반복 실행
head(apply(X, 2, mean)) # 열 빙향으로 평균 계산 반복 실행

head(apply(X, 1, sd))   # 실습) 행방향 표준편차 구하기
head(apply(X, 2, sd))   # 실습) 열방향 표준편차 구하기

m1 <- apply(X, 1, mean)
s1 <- apply(X, 1, sd)

plot(m1)
abline(h=3, col=2)
abline(h = c(3 - 1.96, 3 + 1.96), col = 4)  # 신뢰구간



head(rowMeans(X))
head(colMeans(X))
head(rowSums(X))
head(colSums(X))


# scale함수는 각 열에 대해서 적용해준다
Z <- scale(X) # scale(X, center = TRUE, scale = TRUE)
round(Z[1:5, 1:10], 2)
round(cov(Z)[1:5, 1:5], 2)








## 2.1.2. 행렬연산
a <- matrix(rnorm(12), 4, 3)
#  transpose
t(a)
#  행렬 곱
t(a) %*% a   
#  역행렬
b <- t(a) %*% a
b.inv <- solve(b)  # solve는 연립방정식 풀 때 사용됨.

b %*% b.inv

b.inv %*% b

#  연립방정식
Amat <- matrix(c(1,  2,  3,
                 2, -5, -1,
                 0,  1, -1),
               ncol = 3, byrow = TRUE)
bvec <- c(5, 7, 0)
x <- solve(Amat, bvec)
x

as.vector(Amat%*%x) == bvec
identical(as.vector(Amat%*%x), bvec)


#  행렬식
det(b)
#  외적, outer product
x <- c(1:3)
y <- c(4:6)
outer(x, y)









## 2.1.3. 행렬의 분해
#  상삼각, 하삼각 행렬
b
upper.tri(b)
lower.tri(b)

b[lower.tri(b)] <- NA
b

#  QR 분해
#  행렬 X를 X=QR 과 같이 분해하는 것을 가리킨다.
#  단, Q는 직교행렬(역행렬이 전치행렬과 같음)이고 R은 상삼각행렬이다.
#  선형연립방정식의 해를 안정적으로 구하는 데 매우 유용하다.
A.qr <- qr(Amat)
Q <- qr.Q(A.qr)
R <- qr.R(A.qr)
R
t(Q) %*% bvec

#  고유값 분해
N <- 100
d <- 5
D <- matrix(rnorm(N*d), N, d)
A <- t(D) %*% D
eig <- eigen(A)
print(lambda <- eig$values)  # 고유값
print(P <- eig$vectors)      # 고유벡터


identical(A, P %*% diag(lambda) %*% t(P)) # 원래 같음.












#  이를 응용하면 자료에 대한 whitening 또는 coloring에 활용하는 것이 가능하다.
library(mvtnorm)

N <- 100
d <- 3
Sigma <- matrix(c(3, 0.5, 1,
                  0.5, 5, 0.9,
                  1, 0.9, 8), byrow = TRUE, ncol = 3)
X <- rmvnorm(N, sigma = Sigma)
S <- cov(X)
S

eig <- eigen(S)
lambda <- eig$values
P <- eig$vectors
W <- P %*% diag(1/lambda^0.5) %*% t(P) 
Z <- X %*% W
cov(Z)








N <- 7
d <- 7
A <- round(matrix(runif(N*d), N, d))
A.svd <- svd(A)
str(A.svd)

par(mfrow = c(3, 3))
image(A, axes = FALSE, main = "Original image")
plot(A.svd$d, type = "b", 
     xlab = "dimension", ylab = "Singular values", 
     main = "Singular values")
abline(h = 1, col = 2, lty = 2)
image(A.svd$u[,1]%*%t(A.svd$v[,1]), 
      axes = FALSE, main = "Eigen image 1")
image(A.svd$u[,2]%*%t(A.svd$v[,2]), 
      axes = FALSE, main = "Eigen image 2")
image(A.svd$u[,3]%*%t(A.svd$v[,3]), 
      axes = FALSE, main = "Eigen image 3")
image(A.svd$u[,4]%*%t(A.svd$v[,4]), 
      axes = FALSE, main = "Eigen image 4")
image(A.svd$u[,5]%*%t(A.svd$v[,5]), 
      axes = FALSE, main = "Eigen image 5")
image(A.svd$u[,6]%*%t(A.svd$v[,6]), 
      axes = FALSE, main = "Eigen image 6")
image(A.svd$u[,7]%*%t(A.svd$v[,7]), 
      axes = FALSE, main = "Eigen image 7")

?svd

d.tilde <- sum(A.svd$d >= 1)
A.tilde <- round(A.svd$u[,1:d.tilde] %*% diag(A.svd$d[1:d.tilde]) %*% t(A.svd$v[,1:d.tilde]))
par(mfrow = c(1, 2))
image(A, axes = FALSE, main = "Original")
image(A.tilde, axes = FALSE, main = "Compressed")



## 2.2 수치 미분
## 2.2.1. 미분 derivative

#  지수함수를 예로 x=1일 때의 미분 --> 접선의 기울기를 구해보자
round(-exp(-1), 6)  # 실제 미분계수



f <- function(x) exp(-x)
h <- 2^(-2*(0:10))
x <- 1
round(rbind(h, (f(x + h) - f(x))/h), 6) # 10번째쯤 갈수록 수렴하고 있다.

round(rbind(h, (f(x + h) - f(x - h))/(2*h)), 6) # better // 6번째쯤에 수렴




derivative <- function(f, x, tol = 1e-8, tol.h = 5e-10){  # tol은 정확도라고 생각하면 됨.
  h <- 1e-3
  der0 <- (f(x + h) - f(x - h))/(2*h)
  iter <- 0
  while(1) {
    iter <- iter + 1
    h <- 0.25*h
    der1 <- (f(x + h) - f(x - h))/(2*h)
    diff <- max(abs(der1 - der0))
    if (diff < tol | h < tol.h ) {
      ans <- der1
      break
    }
  }
  
  return(list(deriv = ans, diff = diff, h = h, iter = iter))
}


#  sin함수로 한 번 해보자.
x <- seq(from = 0, to = 2*pi, by = 0.0025)
#f <- function(x) x^3 - x^2 
f <- function(x) sin(x)

der <- derivative(f, x)
plot(x, f(x), type = 'l')
























##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################



## 고유값 분해
# A = P %*% large_lambda %*% t(P) = small_lambda1 * eigenVector1 %*% t(eigenVector1) + small_lambda2 * eigenVector2 %*% t(eigenVector2) + ...


## 양정치 행렬(positive definite) / 양반정치 행렬(positive semi definite)


## 특이값 분해
#  U와 V가 직교행렬 == 내적하면 0
#  A = U %*% Σ %*% t(V)일때, A %*% t(A) == V %*% SIGMA^2 %*% t(V) ==> P %*% large_lambda %*% t(P) 꼴로 나옴.
#  // 직교 성질을 갖는 행렬을 서로 전치행렬로 바꾸어 곱하면 I가 나온다.
#  Σ σ


## 그림의 denoise하기! 아니면 파일사이즈 줄이기
N <- 7
d <- 7
A <- round(matrix(runif(N*d), N, d))
A.svd <- svd(A)   # Compute the singular-value decomposition of a rectangular matrix.
str(A.svd)
A.svd$d     # 이 순서대로 정보량의 순위를 알 수 있음 + 1 이하는 버려볼까~?
A.svd$u
A.svd$v

par(mfrow = c(3, 3))
image(A, axes = FALSE, main = "Original image")
plot(A.svd$d, type = "b",    # singular value
     xlab = "dimension", ylab = "Singular values", 
     main = "Singular values")
abline(h = 1, col = 2, lty = 2)
image(A.svd$u[,1]%*%t(A.svd$v[,1]), 
      axes = FALSE, main = "Eigen image 1")
image(A.svd$u[,2]%*%t(A.svd$v[,2]), 
      axes = FALSE, main = "Eigen image 2")
image(A.svd$u[,3]%*%t(A.svd$v[,3]), 
      axes = FALSE, main = "Eigen image 3")
image(A.svd$u[,4]%*%t(A.svd$v[,4]), 
      axes = FALSE, main = "Eigen image 4")
image(A.svd$u[,5]%*%t(A.svd$v[,5]), 
      axes = FALSE, main = "Eigen image 5")
image(A.svd$u[,6]%*%t(A.svd$v[,6]), 
      axes = FALSE, main = "Eigen image 6")
image(A.svd$u[,7]%*%t(A.svd$v[,7]), 
      axes = FALSE, main = "Eigen image 7")

A.svd$d


# size 축소 후 복원
d.tilde <- sum(A.svd$d >= 1)
A.tilde <- round(A.svd$u[,1:d.tilde] %*% diag(A.svd$d[1:d.tilde]) %*% t(A.svd$v[,1:d.tilde]))
par(mfrow = c(1, 2))
image(A, axes = FALSE, main = "Original")
image(A.tilde, axes = FALSE, main = "Compressed")

# 노이즈 제거
A <- matrix(c(rep(1, 2048), rep(0, 2048)), 64, 64)
par(mfrow = c(1, 2))
image(A, axes = FALSE, main = "Original")
A[32, 63] <- 1 - A[32, 63]
A[12, 12] <- 1 - A[12, 12]
image(A, axes = FALSE, main = "Contaminated")

A.svd <- svd(A)
str(A.svd)

d.tilde <- sum(A.svd$d >= 1)
A.tilde <- round(A.svd$d[1] * A.svd$u[,1:d.tilde] %*% t(A.svd$v[,1:d.tilde]))
par(mfrow = c(1, 2))
image(A, axes = FALSE, main = "Contaminated")
image(A.tilde, axes = FALSE, main = "Reconstructed")



f <- function(x) sin(x)
a <- 0; b <- pi/2; inc <- 2e-6
x <- seq(from = a, to = b, by = inc)
y <- f(x)
n <- length(y)
integ.rect <- sum(y[-n]*inc)
integ.trpz <- sum(y*inc) - 0.5*(y[1] + y[n])*inc
round(c(integ.rect, integ.trpz), 6)


GaussianQuadratureIntegral <- function(f, lower, upper, n = 5)
{
  if (n == 2) {
    w <- c(1, 1)
    x <- c(-0.577350269, 0.577350269)
  } else if (n == 3) {
    w <- c(0.5555556, 0.8888889, 0.5555556)
    x <- c(-0.774596669, 0, 0.774596669)
  } else if (n == 4) {
    w <- c(0.3478548, 0.6521452, 0.6521452, 0.3478548)
    x <- c(-0.861136312, -0.339981044, 0.339981044, 0.861136312)
  } else if (n == 5) {
    w <- c(0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269)
    x <- c(-0.906179846, -0.538469310, 0, 0.538469310, 0.906179846)
  } else if (n == 6) {
    w <- c(0.1713245, 0.3607616, 0.4679139, 0.4679139, 0.3607616, 0.1713245)
    x <- c(-0.932469514, -0.661209386, -0.238619186, 0.238619186, 0.661209386, 0.932469514)
  } else {
    cat("Check n: 2, 3, ..., 6\n")
    return(NA)
  }
  ans <- 0.5*(upper - lower)*sum(w*f(0.5*((upper - lower)*x + (upper + lower))))
  return(ans)
}

GaussianQuadratureIntegral(f, 0, pi/2, n = 2)














##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################



## 2.3.4 몬테카를로 방법
#  대수의 법칙을 이용한 것이다.
#  정밀도는 살짝 떨어지지만 다변량에서는 파워풀하게 적용될 수 있다.
n <- 1e6
x <- rexp(n)
mean(x)
#  오차의 한계 구하기(like 신뢰구간)
1.96*sd(x)/sqrt(n)
c(mean(x) - 1.96*sd(x)/sqrt(n), mean(x) + 1.96*sd(x)/sqrt(n))


## 2.3.5 Importance sampling
#  몬테카를로 적분 예시
#  uniform 분포 사용
f <- function(x) exp(-x)/(1 + x*x)*(x >= 0)*(x <= 1)
n <- 1e6
y <- runif(n)
g1 <- function(y) dunif(y)

cat("근사값:", round(mean(f(y)/g1(y)), 6), 
    "  s.e.:", round(sd(f(y)/g1(y))/sqrt(n), 6), "\n")


#  분포를 어떤 분포를 쓰느냐에 따라 성능이 더 좋아질 수도 있다.
#  여기서는 절단한 지수분포를 사용
#  
u <- runif(n); y <- -log(1 - u*(1 - exp(-1)))
g2 <- function(y) exp(-y)/(1 - exp(-1)) # better
cat("근사값:", round(mean(f(y)/g2(y)), 6), 
    "  s.e.:", round(sd(f(y)/g2(y))/sqrt(n), 6), "\n")

#  Truncated exponential방법이 더 변화가 없음.
x <- seq(from = 0, to = 1, by = 0.001)
plot(x, f(x)/g1(x), type = "l", ylab = "f(x)/g(x)")
lines(x, f(x)/g2(x), col = 2)
legend("topright", c("Uniform", "Truncated exponential"), 
       col = 1:2, lty = rep(1, 2))

#  결국! f(x)와 g(x)의 함수 형태가 비슷해야 더 좋은 형태가 된다.



N <- 10000
U <- runif(N)
X <- -log(1-U)
x <- seq(0,10,by = 0.005)
hist(X, probability = T)
lines(x, dexp(x))
plot(density(X))




## 2.4 방정식 해 구하기
## 2.4.1 Bisection search
#  이진트리랑 같은 개념
f <- function(x) x^2 - 1

a <- 0; b <- 3 
iter <- 0; max.iter <- 1000; tol <- 1e-5
r <- seq(from = a, to = b, length = 3)
y <- c(f(r[1]), f(r[2]), f(r[3]))

if ( y[1]*y[3] > 0 ) stop("f does not have opposite sign at endpoints")

while (iter < max.iter & abs(y[2]) > tol ) {
  iter <- iter + 1
  if ( y[1]*y[2] < 0 ) {
    r[3] <- r[2]; y[3] <- y[2]
  } else {
    r[1] <- r[2]; y[1] <- y[2]
  }
  r[2] <- (r[1] + r[3])/2
  y[2] <- f(r[2])
  cat("Iteration ", iter, ":", "x = ", round(r[2], 4), "f(x) = ", round(y[2], 4), "\n")
}



## 2.4.2 Newton-Raphson method
#  미분개념을 이용한 것.(테일러급수)
#  gradient 방법론과 비슷
#  때문에 미분 가능 함수만 적용 가능
f <- function(x) x^2 - 1
f1d <- function(x) 2*x

iter <- 0; max.iter <- 1000
dd <- 1000; tol <- 1e-6
x0 <- 2
x <- rep(NA, max.iter)

while ( abs(dd) > tol & iter <= max.iter) {
  iter <- iter + 1
  dd <-  f(x0)/f1d(x0)
  x[iter] <- x0 <- x0 - dd
}
cat(x0, "(# of iterations:", iter, ")\n")
plot(x[1:iter], type = "b", xlab = "Iteration", ylab = "Solution")



## 2.4.3 uniroot() 함수
#  해를 구해주는 함수가 있음.
#  Brent method를 이용한 해를 구해준다. 
uniroot(f, lower = 0, upper = 3, tol = 1e-6)








## 2.5 최적화
#  : 다변수 상태에서 최소값을 찾으려는 행위, 방정식의 해를 구하는 문제임.
#  2.5.1 Newton-Raphson method
#  이차미분을 해야함. + 역행렬을 구해야함.
f <- function(x) x^4 + 3*x^2
f1d <- function(x) 4*x^3 + 6*x
f2d <- function(x) 12*x^2 + 6

x <- seq(from = -2, to = 2, by = 0.002)
plot(x, f(x), type = 'l')

#  적용
dd <- 1000; tol <- 1e-6
iter <- 0; max.iter <- 1000

x0 <- 1
x <- rep(NA, max.iter)

while ( abs(dd) > tol & iter <= max.iter) {
  iter <- iter + 1
  dd <-  f1d(x0)/f2d(x0)
  x[iter] <- x0 <- x0 - dd
}
cat(x0, "(# of iterations:", iter, ")\n")
plot(x[1:iter], type = "b", xlab = "Iteration", ylab = "Solution")
abline(h = pi, col = 2, lty = 2)




#  2.5.2 Gradient method
#  --> gradient descent
#  η(에타)는 학습률이라고 함. 보폭을 정해주는 역할. 잘 정해야함. 한 번 정하면 업데이트를 안함.
f <- function(x) x^4 + 3*x^2
f1d <- function(x) 4*x^3 + 6*x

eta <- 0.05; dd <- 1000; tol <- 1e-6
iter <- 0; max.iter <- 1000

x0 <- 1
x <- rep(NA, max.iter)

while ( abs(dd) > tol & iter <= max.iter) {
  iter <- iter + 1
  dd <- f1d(x0)
  x[iter] <- x0 <- x0 - eta*dd  # gradient ascent는 +eta*dd 이다.
}
cat(x0, "(# of iterations:", iter, ")\n")
plot(x[1:iter], type = "b", xlab = "Iteration", ylab = "Solution")





#  2.5.3 Golden section search (SKIP!)
GoldenSection <- function(f, a, b, tol = 1e-8, max.iter = 1000, doplot = TRUE)
  #   golden section search
  #   to find the minimum of f on [a,b]
  #   f: a strictly unimodal function on [a,b]
{
  gr <- (sqrt(5) + 1) / 2
  c <- b - (b - a) / gr
  d <- a + (b - a) / gr 
  iter <- 0
  x <- rep(NA, max.iter)
  while (abs(c - d) > tol) {
    iter <- iter + 1
    ifelse(f(c) < f(d), b <- d, a <- c)
    c <- b - (b - a) / gr
    d <- a + (b - a) / gr
    x[iter] <- (a + b)/2
  }
  cat(x[iter], "(# of iterations:", iter, ")\n")
  if (doplot) plot(x[1:iter], type = "b", xlab = "Iteration", ylab = "Solution")
  
  return(list(optimal = x[iter], objective = f(x[iter]), iter = iter))
}

GoldenSection(f, a = -5, b = 3)




#  2.5.4 optimise(), optim() 함수 이용
#  optimise()는 golden section search 와 successive parabolic interpolation 을 결합한 방식으로 최적해를 찾아준다.
optimize(f, interval = c(-1, 1), maximum = FALSE, tol = 1e-6)  # objective는 목적이 무엇인지 































