setwd("~/Desktop/9890")
library(glmnet)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(MASS)
library(randomForest)
library(tidyverse)
library(grid)
library(viridis)
rm(list = ls())    #delete objects
cat("\014")

# read data
c17 <- read.csv("COMBO17.csv", header=T)
head(c17)
str(c17)

# clean data
c17[,"e.W420FE"] <- as.numeric(gsub(" ", "", gsub("E", "e", c17[,"e.W420FE"])))

# remove uneccessary columns
c17 <- c17[,-c(1, 7, 8, 9)]

# remove NA values
c17 <- na.omit(c17)
rownames(c17) <- c(seq(1, nrow(c17), 1))

# change order of columns so that target variable will be the first column
c17 <- c17[,c(5,1,2,3,4,6:ncol(c17))]

# standardize all predictor columns
# c17[,-1] <- scale(c17[,-1])
for (i in 2:ncol(c17)) {
    c17[,i] <- c17[,i]/sd(c17[,i])
}
head(c17)

# define dimension
p                  <-  dim(c17)[2]-1
n                  <-  dim(c17)[1]
train.rate         <- 0.8
# number of iteration and K folds
iterations         <- 100
K                  <- 10

# train and test R2
train.r2           <-  matrix(0, nrow = iterations, ncol = 4)
colnames(train.r2) <-  c("Lasso", "Elastic", "Ridge", "RF")
test.r2            <-  matrix(0, nrow = iterations, ncol = 4)
colnames(test.r2)  <-  c("Lasso", "Elastic", "Ridge", "RF")

# Matrix of 100 train coefficients - NOT USE
matrix.coef        <- list(lasso=matrix(0, nrow = iterations, ncol = p+1),
                           elastic=matrix(0, nrow = iterations, ncol = p+1),
                           ridge=matrix(0, nrow = iterations, ncol = p+1))
lasso.coef         <-  matrix(0, nrow = iterations, ncol = p+1)
el.coef            <-  matrix(0, nrow = iterations, ncol = p+1)
rid.coef           <-  matrix(0, nrow = iterations, ncol = p+1)

# RF
rf.importance      <- matrix(0, nrow = iterations, ncol = p)

# time for cv and model fit
time.cv            <-  matrix(0, nrow = iterations, ncol = 3)
colnames(time.cv)  <-  c("Lasso", "Elastic", "Ridge")
time.fit           <-  matrix(0, nrow = iterations, ncol = 4)
colnames(time.fit) <-  c("Lasso", "Elastic", "Ridge", "RF")

# mean cross validation error - NOT USE
cv.error           <- matrix(0, nrow = iterations, ncol = 3)

# train and test risiduals - NOT USE
train.resid        <-  matrix(0, nrow = floor(n*train.rate), ncol = 3)
colnames(train.resid)  <-  c("Lasso", "Elastic", "Ridge")
test.resid         <-  matrix(0, nrow = n-floor(n*train.rate), ncol = 3)
colnames(test.resid)  <-  c("Lasso", "Elastic", "Ridge")

# shuffle data
set.seed(3)
samp.ind    <- sample(n, n)
X           <- as.matrix(c17[samp.ind, -1])
y           <- c17[samp.ind,1]
train.rate  <- 0.8

for(m in 1:iterations){
  
  cat(sprintf("iteration = %3.f \n", m))
    
  train  <-  sample(n, n*train.rate)
  
  ############ lasso regression
  # begin cross validate
  start.time  <- proc.time()
  cv.lasso  <-  cv.glmnet(X[train,], y[train],
                          alpha = 1, family = "gaussian",
                          intercept = T)
  end.time  <-  proc.time() - start.time
  time.cv[m,1]   <-  end.time["elapsed"]
  cv.error[m,1] = min(cv.lasso$cvm)
  # begin model fit
  start.time  <- proc.time()
  lasso.fit = glmnet(X[train,], y[train],
                     alpha = 1, family = "gaussian",
                     intercept = T,
                     lambda = cv.lasso$lambda.min,
                     standardize = F)
  end.time  <-  proc.time() - start.time
  time.fit[m,1]  = end.time["elapsed"]
  # calculate trainr2, testr2, train residuals and testresduals
  lasso.coef[m,] = coef(lasso.fit)[,1]
  train.pred = as.vector(predict(lasso.fit, newx = X[train,], type ="response"))
  lasso.train.resid = y[train] - train.pred
  train.r2[m,1] = 1 - mean((y[train] - train.pred)^2)/mean((y[train]-mean(y[train]))^2)
  
  test.pred = as.vector(predict(lasso.fit, newx = X[-train,], type="response"))
  lasso.test.resid = y[-train] - test.pred
  test.r2[m,1] = 1 - mean((y[-train] - test.pred)^2)/mean((y[-train]-mean(y[-train]))^2)

  ############# elastic net
  # begin cross-validation
  start.time  <- proc.time()
  cv.el = cv.glmnet(X[train,], y[train],
                    alpha = 0.5, family = "gaussian",
                    intercept = T)
  end.time  <-  proc.time() - start.time
  time.cv[m,2]  = end.time["elapsed"]
  cv.error[m,2] = min(cv.el$cvm)
  # begin model fit
  start.time  <- proc.time()
  el.fit = glmnet(X[train,], y[train],
                  alpha = 0.5, family = "gaussian",
                  intercept = T,
                  lambda = cv.el$lambda.min,
                  standardize = F)
  end.time  <-  proc.time() - start.time
  time.fit[m,2]  = end.time["elapsed"]
  # calculate trainr2, testr2, train residuals and testresduals
  el.coef[m,] = coef(el.fit)[,1]
  train.pred = as.vector(predict(el.fit, newx = X[train,], type ="response"))
  el.train.resid = y[train] - train.pred
  train.r2[m,2] = 1 - mean((y[train] - train.pred)^2)/mean((y[train]-mean(y[train]))^2)

  test.pred = as.vector(predict(el.fit, newx = X[-train,], type="response"))
  el.test.resid = y[-train] - test.pred
  test.r2[m,2] = 1 - mean((y[-train] - test.pred)^2)/mean((y[-train]-mean(y[-train]))^2)
    
  ############## ridge 
  # begin cross-validation
  start.time  <- proc.time()
  cv.rid = cv.glmnet(X[train,], y[train],
                     alpha = 0, family = "gaussian",
                     intercept = T)
  end.time  <-  proc.time() - start.time
  time.cv[m,3]  = end.time["elapsed"]
  cv.error[m,3] = min(cv.rid$cvm)
  # begin model fit
  start.time  <- proc.time()
  rid.fit = glmnet(X[train,], y[train],
                   alpha = 0, family = "gaussian", 
                   intercept = T, lambda = cv.rid$lambda.min,
                   standardize = F)
  end.time  <-  proc.time() - start.time
  time.fit[m,3]  = end.time["elapsed"]
  # calculate trainr2, testr2, train residuals and testresduals
  rid.coef[m,] = coef(rid.fit)[,1]
  train.pred = as.vector(predict(rid.fit, newx = X[train,], type = "response"))
  rid.train.resid = y[train] - train.pred
  train.r2[m,3] = 1 - mean((y[train] - train.pred)^2)/mean((y[train]-mean(y[train]))^2)

  test.pred = as.vector(predict(rid.fit, newx = X[-train,], type = "response"))
  rid.test.resid = y[-train] - test.pred
  test.r2[m,3] = 1 - mean((y[-train] - test.pred)^2)/mean((y[-train]-mean(y[-train]))^2)

  ############### random forest 
  # begin model fit
  start.time  <- proc.time()
  rf = randomForest(X[train,], y[train], mtry = sqrt(p), importance = T)
  end.time  <-  proc.time() - start.time
  time.fit[m,4]  = end.time["elapsed"]
  # calculate trainr2, testr2, train residuals and testresduals
  train.pred = predict(rf, newdata = X[train,])
  rf.train.resid = y[train] - train.pred
  train.r2[m,4] = 1 - mean((y[train] - train.pred)^2)/mean((y[train]-mean(y[train]))^2)
    
  test.pred = predict(rf, newdata = X[-train,])
  rf.test.resid = y[-train] - test.pred
  test.r2[m,4] = 1 - mean((y[-train] - test.pred)^2)/mean((y[-train]-mean(y[-train]))^2)
  rf.importance[m,]  <- rf$importance[,1] 
}

## NOT USE - WON'T BE ABLE TO GET THE CV CURVE
# alphas  <- list(1, 0.5, 0)

# for(m in 1:iterations) {
#     cat(sprintf("iteration = ", m, "\n"))
#     train  <-  sample(n, n*train.rate)

#     for(i in seq_along(alphas)) {

#         # lasso/elastic/ridge regression
#         start.time     <- proc.time()
#         cv             <-  cv.glmnet(X[train,], y[train], alpha = alphas[[i]], family = "gaussian", intercept = T)
#         end.time       <-  proc.time() - start.time
#         time.cv[m,i]   <-  end.time["elapsed"]

#         cv.error[m,i]  <-  min(cv$cvm)
#         #bestlam = cv$lambda.min

#         # record lasso fit time
#         start.time     <- proc.time()
#         mod            <-  glmnet(X[train,], y[train], alpha = alphas[[i]], family = "gaussian",
#                                   intercept = T, lambda = cv$lambda.min, standardize = F)
#         end.time       <-  proc.time() - start.time
#         time.fit[m,i]  <-  end.time["elapsed"]

#         matrix.coef[[i]][m,]  <-  coef(mod)[,1]

#         train.pred     <-  as.vector(predict(mod, newx = X[train,], type ="response"))
#         #train.resid   <-  y[train] - train.pred
#         train.r2[m,i]  <-  1 - mean((y[train] - train.pred)^2)/mean((y[train]-mean(y[train]))^2)

#         test.pred      <-  as.vector(predict(mod, newx = X[-train,], type="response"))
#         #test.resid    <-  y[-train] - test.pred
#         test.r2[m,i]   <-  1 - mean((y[-train] - test.pred)^2)/mean((y[-train]-mean(y[-train]))^2)
#         if (m == 100) {
#             train.resid[,i] <-  y[train] - train.pred
#             test.resid[,i]  <-  y[-train] - test.pred
#         }
#     }
#     # RF
#     start.time      <- proc.time()
#     rf              <-  randomForest(X[train,], y[train], mtry = sqrt(p), importance = T)
#     end.time        <-  proc.time() - start.time
#     time.fit[m,4]   <-  end.time["elapsed"]

#     rf.pred.train   <-  predict(rf, newdata = X[train,])
#     rf.train.resid  <-  y[train] - rf.pred.train
#     train.r2[m,4]   <-  1 - mean((y[train] - rf.pred.train)^2)/mean((y[train]-mean(y[train]))^2)
#     rf.pred.test    <-  predict(rf, newdata = X[-train,])
#     rf.test.resid   <-  y[-train] - rf.pred.test
#     test.r2[m,4]    <-  1 - mean((y[-train] -rf.pred.test)^2)/mean((y[-train]-mean(y[-train]))^2)
# }

train.resid <- data.frame(Lasso=lasso.train.resid, Elastic=el.train.resid,
                          Ridge=rid.train.resid, RF=rf.train.resid)
test.resid  <- data.frame(Lasso=lasso.test.resid, Elastic=el.test.resid,
                          Ridge=rid.test.resid, RF=rf.test.resid)

colnames(lasso.coef)    <- c(c("intercept"), colnames(c17[,-1]))
colnames(el.coef)       <- c(c("intercept"), colnames(c17[,-1]))
colnames(rid.coef)      <- c(c("intercept"), colnames(c17[,-1]))
colnames(rf.importance) <- c(colnames(c17[,-1]))

#write.csv(lasso.coef, file ="lasso_coef.csv")
#write.csv(el.coef, file = "el_coef.csv")
#write.csv(rid.coef, file = "rid_coef.csv")
#write.csv(rf.importance, file = "rf_importance.csv")

#write.csv(cv.error, file = "cv_error.csv")
#write.csv(test.r2, file = "testR2.csv")
#write.csv(train.r2, file = "trainR2.csv")

#write.csv(time.cv, file = "timeCV.csv")
#write.csv(time.fit, file = "timefit.csv")

#write.csv(train.resid, file = "train_resid.csv")
#write.csv(test.resid, file = "test_resid.csv")

train.resid <- read.csv("train_resid.csv")[,-1]
test.resid <- read.csv("test_resid.csv")[,-1]

train.r2 <- read.csv("trainR2.csv")[,-1]
test.r2 <- read.csv("testR2.csv")[,-1]

time.cv <- read.csv("timeCV.csv")[,-1]
time.fit <- read.csv("timefit.csv")[,-1]


# resid plot
#options(repr.plot.width=25, repr.plot.height=10)

limits <- c(-0.3, 0.9)
breaks <- seq(limits[1], limits[2], by=.2)
# should be in dataframe format
g1 <- train.resid %>%
  gather(key=models, value=residuals) %>%
  ggplot(aes(x=models, y=residuals, fill=models)) +
  geom_boxplot() +
  scale_y_continuous(limits=limits, breaks=breaks) + 
  scale_fill_viridis(discrete = TRUE, alpha=1, option="D") +
  scale_color_viridis(discrete = TRUE) +
  theme_bw() +
  theme(
    legend.position="none",
    plot.title = element_text(size=20, face = "bold"),
    axis.text.x = element_text(angle = 90, hjust = 1, size=18,color="darkred"),
    axis.text.y = element_text(hjust = 1, size=18,color="darkred"),
    axis.title.y = element_text(size = rel(1.8), angle = 90)
  ) +
  ggtitle("Train residuals") +
  xlab("")

g2 <- test.resid %>%
  gather(key=models, value=residuals) %>%
  ggplot(aes(x=models, y=residuals, fill=models)) +
  geom_boxplot() +
  scale_y_continuous(limits=limits, breaks=breaks) + 
  scale_fill_viridis(discrete = TRUE, alpha=1, option="D") +
  scale_color_viridis(discrete = TRUE) +
  theme_bw() +
  theme(
    legend.position="none",
    plot.title = element_text(size=20, , face = "bold"),
    axis.text.x = element_text(angle = 90, hjust = 1, size=18,color="darkred"),
    axis.text.y = element_text(hjust = 1, size=18,color="darkred"),
    axis.title.y = element_text(size = rel(1.8), angle = 90)
  ) +
  ggtitle("Test residuals") +
  xlab("")

grid.arrange(g1, g2, ncol=2)

# r2 plot
#options(repr.plot.width=25, repr.plot.height=10)
train.r2 <- data.frame(train.r2)
test.r2 <- data.frame(test.r2)
limits <- c(0.8, 1)
breaks <- seq(limits[1], limits[2], by=.05)
# should be in dataframe format
g1 <- train.r2 %>%
  gather(key=models, value=R.square) %>%
  ggplot(aes(x=models, y=R.square, fill=models)) +
  geom_boxplot() +
  scale_y_continuous(limits=limits, breaks=breaks) + 
  scale_fill_viridis(discrete = TRUE, alpha=0.8, option="D") +
  scale_color_viridis(discrete = TRUE) +
  theme_bw() +
  theme(
    legend.position="none",
    plot.title = element_text(size=20, face = "bold"),
    axis.text.x = element_text(angle = 90, hjust = 1, size=18,color="darkred"),
    axis.text.y = element_text(hjust = 1, size=18,color="darkred"),
    axis.title.y = element_text(size = rel(1.8), angle = 90)
  ) +
  ggtitle("Train R.square") +
  xlab("")

g2 <- test.r2 %>%
  gather(key=models, value=R.square) %>%
  ggplot(aes(x=models, y=R.square, fill=models)) +
  geom_boxplot() +
  scale_y_continuous(limits=limits, breaks=breaks) + 
  scale_fill_viridis(discrete = TRUE, alpha=0.8, option="D") +
  scale_color_viridis(discrete = TRUE) +
  theme_bw() +
  theme(
    legend.position="none",
    plot.title = element_text(size=20, , face = "bold"),
    axis.text.x = element_text(angle = 90, hjust = 1, size=18,color="darkred"),
    axis.text.y = element_text(hjust = 1, size=18,color="darkred"),
    axis.title.y = element_text(size = rel(1.8), angle = 90)
  ) +
  ggtitle("Test R.square") +
  xlab("")

grid.arrange(g1, g2, ncol=2)

# total time: cv + fit
# should be in dataframe format
# time.total %>%
#   gather(key=models, value=time) %>%
#   ggplot(aes(x=models, y=time, fill=models)) +
#   geom_boxplot() +
#   #scale_y_continuous(limits=limits, breaks=breaks) + 
#   scale_fill_viridis(discrete = TRUE, alpha=0.8, option="C") +
#   scale_color_viridis(discrete = TRUE) +
#   theme_bw() +
#   theme(
#     legend.position="none",
#     plot.title = element_text(size=20, face = "bold"),
#     axis.text.x = element_text(angle = 90, hjust = 1, size=18,color="darkred"),
#     axis.text.y = element_text(hjust = 1, size=18,color="darkred"),
#     axis.title.y = element_text(size = rel(1.8), angle = 90)
#   ) +
#   ggtitle("Total time") +
#   xlab("")

# time cv
# should be in dataframe format
time.cv <- data.frame(time.cv)
time.cv %>%
  gather(key=models, value=time) %>%
  ggplot(aes(x=models, y=time, fill=models)) +
  geom_boxplot() +
  #scale_y_continuous(limits=limits, breaks=breaks) + 
  scale_fill_viridis(discrete = TRUE, alpha=0.8, option="C") +
  scale_color_viridis(discrete = TRUE) +
  theme_bw() +
  theme(
    legend.position="none",
    plot.title = element_text(size=20, face = "bold"),
    axis.text.x = element_text(angle = 90, hjust = 1, size=18,color="darkred"),
    axis.text.y = element_text(hjust = 1, size=18,color="darkred"),
    axis.title.y = element_text(size = rel(1.8), angle = 90)
  ) +
  ggtitle("Time CV") +
  xlab("")

# make cv plot

par(mfrow=c(1,3))
options(repr.plot.width=25, repr.plot.height=10)
plot(cv.lasso, main='LASSO' )
plot(cv.el, main='ELASTICNET')
plot(cv.rid, main="RIDGE")
#dev.copy(jpeg, filename="cv_curve.jpg");
#dev.off();

# perform 100 boostrapping samples to find sd of estimated coefficients
p               <-  dim(c17)[2]-1
n               <-  dim(c17)[1]
bootstrapSamples =     100
time.bs          <-    matrix(0, nrow = bootstrapSamples, ncol=4)
beta.rf.bs       =     matrix(0, nrow = bootstrapSamples, ncol=p)    
beta.ls.bs       =     matrix(0, nrow = bootstrapSamples, ncol=p)         
beta.en.bs       =     matrix(0, nrow = bootstrapSamples, ncol=p)         
beta.rd.bs       =     matrix(0, nrow = bootstrapSamples, ncol=p)         

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]

  #############lasso
  a                =     1 # lasso
  start.time       <-    proc.time()
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)
  end.time         <-    proc.time() - start.time
  time.bs[m,1]     <-    end.time['elapsed'] 
  beta.ls.bs[m,]   =     as.vector(fit$beta)
  
  ############el
  a                =     0.5 # elastic-net
  start.time       <-    proc.time()
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)
  end.time         <-    proc.time() - start.time
  time.bs[m,2]     <-    end.time['elapsed']
  beta.en.bs[m,]   =     as.vector(fit$beta)

  ############ridge
  a                =     0 # ridge
  start.time       <-    proc.time()
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)  
  end.time         <-    proc.time() - start.time
  time.bs[m,3]     <-    end.time['elapsed']
  beta.rd.bs[m,]   =     as.vector(fit$beta)

  ##############rf
  start.time       <-    proc.time()
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[m,]   =     as.vector(rf$importance[,1])
  end.time         <-    proc.time() - start.time
  time.bs[m,4]     <-    end.time['elapsed']
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}


colnames(time.bs)  <- c("Lasso", "Elastic", "Ridge", "RF")
colnames(beta.rf.bs) <- colnames(c17[,-1])
colnames(beta.ls.bs) <- colnames(c17[,-1])
colnames(beta.en.bs) <- colnames(c17[,-1])
colnames(beta.rd.bs) <- colnames(c17[,-1])

#write.csv(time.bs, file ="time_bs.csv")
#write.csv(beta.ls.bs, file ="beta_ls_bs.csv")
#write.csv(beta.en.bs, file ="beta_en_bs.csv")
#write.csv(beta.rd.bs, file ="beta_rd_bs.csv")
#write.csv(beta.rf.bs, file ="beta_rf_bs.csv")

beta.ls.bs <- read.csv("beta_ls_bs.csv")[,-1]
beta.en.bs <- read.csv("beta_en_bs.csv")[,-1]
beta.rd.bs <- read.csv("beta_rd_bs.csv")[,-1]
beta.rf.bs <- read.csv("beta_rf_bs.csv")[,-1]
time.bs <- read.csv("time_bs.csv")[,-1]

ls.bs.sd <- apply(beta.ls.bs, 2, sd)
en.bs.sd <- apply(beta.en.bs, 2, sd)
rd.bs.sd <- apply(beta.rd.bs, 2, sd)
rf.bs.sd <- apply(beta.rf.bs, 2, sd)

# fit lasso to the whole data
a=1 # lasso
start.time       <-    proc.time()
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit              =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
end.time         <-    proc.time() - start.time
time.ls          <-    end.time['elapsed'] 
betaS.ls               =     data.frame(colnames(X), as.vector(fit$beta), 2*ls.bs.sd)
colnames(betaS.ls)     =     c( "feature", "value", "err")

# fit en to the whole data
a=0.5 # elastic-net
start.time       <-    proc.time()
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit              =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
end.time         <-    proc.time() - start.time
time.el          <-    end.time['elapsed']
betaS.en               =     data.frame(colnames(X), as.vector(fit$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

# fit ridge to the whole data
a=0 # ridge
start.time       <-    proc.time()
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit              =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)
end.time         <-    proc.time() - start.time
time.rd          <-    end.time['elapsed']
betaS.rd               =     data.frame(colnames(X), as.vector(fit$beta), 2*rd.bs.sd)
colnames(betaS.rd)     =     c( "feature", "value", "err")

# fit rf to the whole data
start.time       <-    proc.time()
rf               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)
end.time         <-    proc.time() - start.time
time.rf          <-    end.time['elapsed']
betaS.rf               =     data.frame(colnames(X), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

# we need to change the order of factor levels by specifying the order explicitly.
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.ls$feature     =  factor(betaS.ls$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.rd$feature     =  factor(betaS.rd$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])

options(repr.plot.width=25, repr.plot.height=15)
en =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  labs(x = element_blank(), y = "Coefficients", title = expression(Elastic)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=18,color="darkred"),
        axis.text.y = element_text(hjust = 1, size=20, color="darkred"),
        plot.title = element_text(size=20, , face = "bold"))

ls =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  labs(x = element_blank(), y = "Coefficients", title = expression(LASSO)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=18,color="darkred"),
        axis.text.y = element_text(hjust = 1, size=20, color="darkred"),
        plot.title = element_text(size=20, , face = "bold"))

rd =  ggplot(betaS.rd, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  labs(x = element_blank(), y = "Coefficients", title = expression(Ridge)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=18,color="darkred"),
        axis.text.y = element_text(hjust = 1, size=20, color="darkred"),
        plot.title = element_text(size=20, , face = "bold"))

rf =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  labs(x = element_blank(), y = "Importance", title = expression(Randon~Forest)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=18,color="darkred"),
        axis.text.y = element_text(hjust = 1, size=20, color="darkred"),
        plot.title = element_text(size=20, , face = "bold"))

grid.arrange(en, ls, rd, rf, nrow = 4)

print(time.ls)
print(time.el)
print(time.rd)
print(time.rf)

quantile(test.r2[,1], c(.10, .90))
quantile(test.r2[,2], c(.10, .90)) 
quantile(test.r2[,3], c(.10, .90)) 
quantile(test.r2[,4], c(.10, .90)) 

colMeans(test.r2)
apply(test.r2, 2, sd)
colMeans(time.bs)
apply(time.bs, 2, sd)
colMeans(test.r2) + (qnorm(.95)*apply(test.r2, 2, sd)/sqrt(nrow(test.r2)))
colMeans(test.r2) - (qnorm(.95)*apply(test.r2, 2, sd)/sqrt(nrow(test.r2)))
colMeans(test.r2) + (qt(.95, nrow(test.r2)-1)*apply(test.r2, 2, sd)/sqrt(nrow(test.r2)))
colMeans(test.r2) - (qt(.95, nrow(test.r2)-1)*apply(test.r2, 2, sd)/sqrt(nrow(test.r2)))
colMeans(time.bs) + (qt(.95, nrow(time.bs)-1)*apply(time.bs, 2, sd)/sqrt(nrow(time.bs)))
colMeans(time.bs) - (qt(.95, nrow(time.bs)-1)*apply(time.bs, 2, sd)/sqrt(nrow(time.bs)))

#plot(c17$UbMAG, c17$Mcz, xlab="Bessell U luminosity", ylab="Redshift Magnitude")
options(repr.plot.width=10, repr.plot.height=8)
ggplot(c17, aes(x = UbMAG, y = Mcz)) +
  geom_point() + geom_smooth() +
  theme(
      plot.title = element_text(size=20, face = "bold"),
      axis.title.y = element_text(size = rel(1.8), angle = 90),
      axis.title.x = element_text(size = rel(1.8))) +
  ggtitle("Redshift vs Bessell U Luminosity")
