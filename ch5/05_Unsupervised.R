

## 1. Clustering
#  1.1 유사성 측도
#  Hamming distance (number of mismatches): categorical 데이터에 사용 --> 다른 변수의 갯수

set.seed(0)
x <- rbind(matrix(rnorm(100, sd = 0.3), ncol = 2),
           matrix(rnorm(100, mean = 1, sd = 0.3), ncol = 2))
colnames(x) <- c("x", "y")


cl <- kmeans(x, 2)


par(mfrow = c(1, 2))
plot(x, xlab = "", ylab = "", col = "gray", pch = 19, 
     main = "Original data")
color <- cl$cluster + 1
plot(x, xlab = "", ylab = "", col = color, pch = 19, 
     main = "Clustered data")
points(cl$centers, col = 2:3, pch = "+", cex = 2)



## Data loading
protein <- read.csv(choose.files())
protein <- protein[, -1]
summary(protein)


vars.to.use <- colnames(protein)[-1]
#pmat <- scale(protein[,vars.to.use])
pmat <- sapply(protein[, vars.to.use], scale)


pKmeans <- kmeans(pmat, 5) 


print.clusters <- function(labels, k) {
  for(i in 1:k) {
    print(paste("cluster", i))
    print(protein[labels == i, c("Country","RedMeat","Fish","Fr.Veg")])
  }
}

print.clusters(pKmeans$cluster, 5)




#  1.3 계층적 군집 알고리즘 Hierarchical clustering
d <- dist(pmat, method = "euclidean") # Euclidean distances
pfit <- hclust(d, method = "ward.D2")


plot(pfit, labels = protein$Country)
rect.hclust(pfit, k = 5)









# ? 극댓값을 찾을 수 없는경우는?
# ? overfit이 발생할 수 있을 거 같은데...

## 2. Density estimation
#  2.1 Parametric
#  분포를 알아야 MLE를 사용할 수 있음. MLE는 그냥 최적의 파라메터를 찾기 위함이다.
### MLE 계산하는 것이 기말고사에 나옴. + 실습도 같이 낼 듯(수치미분(Gradient)으로 편미분하게끔)


#  2.2 Nonparametric: Kernel smoothing
#  쪼만한 가우시안 그래프 하나를 커널이라고 한다. == 데이터 포인트마다의 가우시안
#  커널(가우시안)들을 종합해 놓은 것이 커널 스무디라고 한다.
#  식에서 h가 클수록 커널의 분포가 퍼짐을 의미함.
#  좋은 커널을 만들어야 추정이 잘 됨.
#  --> Epanechnikou kernel // K(u) = 0.75(1-u^2), |u| <= 1

data(faithful)   # 간헐천 분출시간, 대기시간

par(mfrow = c(1, 2))

hist(faithful$eruptions, probability = TRUE,   # 최적의 bandwidth를 알아서 찾아줌.
     xlim = c(0, 7), main = "", col = "gray", border = "white", xlab = "Eruptions")
rug(faithful$eruptions)    # 데이터 포인트를 x축에 뿌려주는 것.
lines(density(faithful$eruptions), col = 2, lwd = 3)

hist(faithful$eruptions, probability = TRUE, nclass = 20, 
     xlim = c(0, 7), main = "", col = "gray", border = "white", xlab = "Eruptions")
rug(faithful$eruptions)
lines(density(faithful$eruptions), col = 2, lwd = 3)



#  ex
set.seed(1)
n <- 1000
x <- rnorm(n, mean = 5, sd = 3)
mu.mle <- mean(x)
sd.mle <- sqrt(var(x)*(n-1)/n)


plot(density(x), ylim = c(0, 0.15))
lines(sort(x), dnorm(sort(x), mean = mu.mle, sd = sd.mle), col = "darkred", lwd = 3)
lines(sort(x), dnorm(sort(x), mean = 5, sd = 3), col = "blue", lwd = 3)


## 3. Dimension reduction
#  3.1 Principal Component Analysis, PCA

summary(state.x77)
dat <- scale(state.x77[, 2:6])
PC <- princomp(dat)
summary(PC)
loadings(PC)   # 빈칸은 0이라고 생각하면 됨.
# --> 1st 주성분은 살기 좋은 곳 이라고 해석할 수 있음.
plot(PC, main = "Screeplot", cex.main = 0.9, cex.axis = 0.7, cex = 0.7)

PC$scores[1:7, ]  # pc score를 볼 수 있음.

plot(PC$scores[,1:2], xlab = "PC1", ylab = "PC2", 
     xlim = c(-4, 4), ylim = c(-4, 4))
abline(h = 0, lty = 2)
abline(v = 0, lty = 2)








#  3.2 요인분석 FA, factor analysis
#  X=LF+ϵ   ,p≥m   여기서 F는 z_1~z_m을 나타냄. ϵ은 약간의 에러를 적용시켜서 더함.
#  --> F는 common factors, ϵ은 uniqueness(unique factor)라고 부름.
#  cf) Confirmative FA(CFA), 우리가 흔히 아는 FA는 Exploratal FA(EFA)라고 한다.

data(state)
head(state.x77)
factanal(state.x77[, 2:6], 2) # 2 == number of factor
# --> Uniquenesses == VAR of 프사이
# --> Cumulative Var는 그닥 중요하지 않음.
# --> The p-value is 0.219라는 뜻은 충분하다라는 뜻.





#  3.3 독립성분분석 ICA, independent component analysis
#  Factor analysis의 비정규성을 따를 때 더 좋음.
#  화이트닝(whitening) --> VAR(X) == SIGMA일때, VAR( -sqrt(SIGMA) %*% X ) == I




























