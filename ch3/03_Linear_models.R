install.packages("ISLR")
library(ISLR)

## 3.5 질적 설명변수 Qualitative Predictors
#  one-hot 인코딩 : 더미변수를 spread 시키는 것(마지막 값은 변수로 만들지 않는다.)
data(Credit)
summary(Credit)


## 3.5.1 모형의 해석 Interpretation
#  별도로 가변수(one-hot 인코딩)를 만들 필요가 없다. R이 알아서 해줌.
fit.lm <- lm(Balance ~ Income + Gender, data = Credit)
summary(fit.lm)


## 3.5.2



## 3.6 교호작용 Interaction
adv <- read.csv(choose.files())
adv <- adv[,-1]
summary(adv)


## 3.6.1 양적변수 사이의 교호작용
#  if 교호작용으로 발생한 변수만 유의한다면 주효과 변수들은 꼭 포함시켜야 한다.
fit.lm <- lm(Sales ~ TV * Radio, data = adv)
summary(fit.lm)


## 3.6.2 질적변수와의 교호작용
fit.lm1 <- lm(Balance ~ Income + Student, data = Credit) # 주효과만 고려
fit.lm2 <- lm(Balance ~ Income * Student, data = Credit) # 교호작용까지 고려
summary(fit.lm1) # 어떤 경우에는 교호작용이 필요할 수도 있고 아닐 수도 있다.
summary(fit.lm2)


## 3.7 Potential Fit Problems
## 3.7.1 Non-linearity of the data





## 3.7.2 Non-constant variance of error terms
Auto <- read.csv(choose.files())
summary(Auto)
data(mtcar)
fit.lm <- lm(mpg ~ horsepower + I(horsepower^2), data = Auto)
summary(fit.lm)




## 3.7.3 Collinearity(다중공산성)
#  예를 들어 몸무게를 예측할 때, 몸무게 ~ 키 + 허리둘레 + 앉은키 --> 나쁜짓임. 왜냐하면 회귀계수를 구할때 역행렬을 잘못 구할 확률이 있음.
#  








install.packages("faraway")
library(faraway)

data(orings)
str(orings)








