# upgrade R version over 3.5.1
#install.packages("keras")
library(keras)

# To install the core Keras library + TensorFlow, 최초 한 번만 실행 
# install_keras() # 실행 전에 Anaconda for Python 3.x 설치해야 함 

mnist <- dataset_mnist() # 데이터 업로드
x_train <- mnist$train$x # 이미지
y_train <- mnist$train$y # label : 0 ~ 9
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 28*28))  # 28*28 행렬이라는 뜻
x_test <- array_reshape(x_test, c(nrow(x_test), 28*28))
# rescale
x_train <- x_train / 255   # 변수들 표준화? 개념(0~1 값으로 변환)
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)    # factor화 : 10은 0~9까지임을 의미
y_test <- to_categorical(y_test, 10)      # factor화


# 모델링
model <- keras_model_sequential() # model이라는 객체 선언
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)
# 200960 == 28*28 * 256 + 256




# 학습
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')  # 모니터링을 accuracy로 하겠다.
)

history <- model %>% fit(   # fit이 진짜 학습 시작
  x_train, y_train, 
  epochs = 20,              # 전체 데이터를 20번 돌리면서 학습시키겠다.
  batch_size = 100,         # gradient에 접근할 때 데이터 한 개체를 학습시키기 보다, 100개씩 접근시키는 것이 더 효율적이다.
  validation_split = 0.2    # 
)

plot(history)



# 모델 적합
model %>% evaluate(x_test, y_test)


Predicted <- model %>% predict_classes(x_test)
Actual <- as.integer(apply(y_test, 1, which.max) - 1)  # 데이터의 구조를 보면 유용한 트릭임을 알 수 있다.
# Confusion Table
table(Actual, Predicted)




# 결과 확인
Right_cases <- (1:10000)[Predicted == Actual]
Wrong_cases <- (1:10000)[Predicted != Actual]


# 데이터를 이미지로 바꿔주는 함수 : show_digit
par(mfrow = c(1, 2))
show_digit(x_test[Right_cases[1],])
show_digit(x_test[Wrong_cases[1],])







## 3. Convolutional Neural Network
#  3.1 기본 개념

#  Convolutional 개념
# --> 그림 7.4에서 가운데 곱하는 인자를 filter라고 함.
# --> 수학에서 Convolution이라고 하면 내적과 같은 의미이다.
# --> 계산방식 : (1*2 + 2*0 + 3*1) + (0*0 + 1*1 + 2*2) + (3*1 + 0*0 + 1*2) = 15
# --> 이런 방식을 돌아가면서 계산한다.
# --> 이런 Convolution 방법을 거치면서 이미지 데이터를 축약시키고 추상화시키는 과정을 한다.
# --> 여기에 bias를 더해준다. : 얼마나 좌우상하로 움직여줄 것인가를 정해주는 역할
# --> Convolution 과정을 여러개 거치다보면 너무 축약되어서 더이상 Convolution을 진행할 수 없어진다.
# --> 이러한 문제점을 해결하기 위해 "패딩"이라는 트릭을 적용한다. : 패딩이란 기존의 input이미지에서 겉의 테두리에 0으로 도배를 한다.
# --> filter가 움직이는 보폭을 스트라이드라고 함.
# --> 방금까지 한 것은 흑백이미지
# --> 컬러 이미지는 RGB값을 가지고 있기 때문에 input이 층이 3개이다. : 3차원 --> 필터도 3개로 늘어남.
# --> 3차원에서 C는 Channel을 의미함(층을 의미)
# --> 중요한 것은 filter를 잘 학습해야함.
# --> filter를 여러개 사용하면 output 역시 다차원으로 나온다.
# --> 그러면 각 필터별로 bias가 있어야함.
# --> 이러한 Convolution 작업이 feature를 추출하는 과정이다.

#  Pooling 개념
# --> convolution이 끝난 후 output을 input으로 받는다.
# --> convolution 결과 메트릭스에서 각 window마다 최대값을 찾는다. : MAX pooling(Mean pooling도 있음)
# --> 이것은 학습 대상은 아님.


#  이렇게 convolution과 pooling을 여러번 반복하다보면 최종적으로 하나의 벡터로 나오는데 이 과정을 flatten이라고 함.
# --> 하나의 층층을 map이라고 함.


#  3.2 Why CNN?
# --> 가중치 공유를 통해 학습 대상 가중치의 개수가 현저히 감소 : 속도가 빠르고, 


#  3.3 CNN 사례
# --> VGG-16 에서 16은 층 갯수 : 얼마나 깊이 들어가는가 정도~



library(keras)

mnist <- dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist

train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255

test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,                    # 필터 갯수를 32개로 함.
                kernel_size = c(3, 3),           # 필터를 3*3 matrix로 함.
                activation = "relu",
                input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%  # window size를 2*2 matrix로 해라.
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3), 
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, 
                kernel_size = c(3, 3), 
                activation = "relu")

summary(model)
# --> 32*(3*3+1)의 학습 대상
# --> 64*(3*3+1)
# --> 64*(3*3+1)
# --> 모두 더하면 55744개가 나옴.



model <- model %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(
  train_images, train_labels,
  epochs = 5, batch_size = 100
)

model %>% evaluate(test_images, test_labels)





pred <- model %>% predict_classes(test_images)
actual <- apply(test_labels, 1, which.max) - 1
table(actual, pred)








## 4. Recurrent Neural Network
#  4.1 구조
#  그림을 보면 됨
# --> 그림에서 보면 매 순간마다 hidden이 있고 y(output)이 있다.
# --> 자세히 보면 전 시점의 히든이 다음 시점의 히든에 영향을 줌.
# --> 현 시점의 output을 내기 위해서는 과거의 모든 시점이 영향을 준다.

# library(tensorflow)
# install_tensorflow(version = "nightly")

library(keras)

max_features <- 10000 # Number of words to consider as features
maxlen <- 120 # Cuts off texts after this many words 

imdb <- dataset_imdb(num_words = max_features)

c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

cat(length(x_train), "train sequences\n")


cat(length(x_test), "test sequences")

cat("Pad sequences (samples x time)\n")



x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
cat("x_train shape:", dim(x_train), "\n")


cat("x_test shape:", dim(x_test), "\n")

#  Embedding and classification by a feed-forward network
#  Embedding은 모든 단어를 나열해서 신호체계로 만듦 -> 1번 단어, 2번단어 등등
# --> 굉장히 비효율적 ; 이를 극복한 것이 Word2Vector : 64차원 공간에 단어들을 뿌릴 때, 비슷한, 유사한 단어들끼리 비슷한 위치에 뿌린다.
# --> Glove : W2V이랑 비슷한데 계산속도가 빨라서 큰 데이터에 유리함.


library(keras)

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features,
                  output_dim = 256,                 # 256차원으로 word embedding을 실시하겠다.
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

summary(model)


history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 100,
  validation_split = 0.2
)
plot(history)


model %>% evaluate(x_test, y_test)

prob <- model %>% predict_proba(x_test)
Predicted <- model %>% predict_classes(x_test)
table(Predicted = Predicted, Actual = y_test)







#  Simple RNN

library(keras)

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, 
                  output_dim = 256) %>%
  layer_simple_rnn(units = 64) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)


model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 100,
  validation_split = 0.2
)
plot(history)


model %>% evaluate(x_test, y_test)


prob <- model %>% predict_proba(x_test)
Predicted <- model %>% predict_classes(x_test)
table(Predicted = Predicted, Actual = y_test)









#  RNN: LSTM --> RNN보다 더 좋은 방법이다.

library(keras)

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, 
                  output_dim = 256) %>%
  layer_lstm(units = 64) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)


library(keras)

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, 
                  output_dim = 256) %>%
  layer_lstm(units = 64) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)



model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 100,
  validation_split = 0.2
)
plot(history)



model %>% evaluate(x_test, y_test)



prob <- model %>% predict_proba(x_test)
Predicted <- model %>% predict_classes(x_test)
table(Predicted = Predicted, Actual = y_test)





#  1d CNN
library(keras)

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, 
                  output_dim = 256,
                  input_length = maxlen) %>%
  layer_conv_1d(filters = 32,
                kernel_size = 5, 
                activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 4) %>%
  layer_conv_1d(filters = 32,
                kernel_size = 5, 
                activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 4) %>%
  layer_conv_1d(filters = 32, 
                kernel_size = 3, 
                activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 16, activation = "sigmoid") %>%
  #layer_dropout(rate = 0.5) %>% 
  #layer_dense(units = 8, activation = "sigmoid") %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)


model %>% compile(
  optimizer = optimizer_rmsprop(lr = 1e-4),
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 12,
  batch_size = 100,
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_test, y_test)


prob <- model %>% predict_proba(x_test)
Predicted <- model %>% predict_classes(x_test)
table(Predicted = Predicted, Actual = y_test)





#  Bidirectional RNN : 반대방향으로도 읽어보는 학습방법

library(keras)

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 256) %>%
  bidirectional(
    layer_lstm(units = 64)
  ) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
  optimizer = "adam", #optimizer_rmsprop(lr = 1e-4),
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 100,
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_test, y_test)


prob <- model %>% predict_proba(x_test)
Predicted <- model %>% predict_classes(x_test)
table(Predicted = Predicted, Actual = y_test)



























