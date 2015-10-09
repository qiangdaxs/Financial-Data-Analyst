# Bias variance tradeoff illustration
rm(list=ls(all=TRUE)) #remove all the variables stored in the memory
set.seed(2)
n <- 1000
p <- 100
sigma <- .5
betazero <- (1:p)^(-2) #1/2^(1:p)#
plot(betazero)
x <- matrix(rnorm(n*p), nrow=n)
y <- x %*% betazero + sigma * rnorm(n) #rnorm's is N(0,1), the second term is the error term

n.test <- 10000
x.test <- matrix(rnorm(n.test*p), nrow=n.test)
y.test <- x.test %*% betazero + sigma * rnorm(n.test)


MSE <- matrix(NA,p,2)
for(k in 1:p){
  lm.result <- lm(y~x[,1:k])
  #This is the traing data set prediction
  y.hat <- predict(lm.result)
  #This is the testing data set prediction
  y.test.hat <- cbind(1,x.test[,1:k]) %*% coef(lm.result)
  MSE[k,1] <- mean((y-y.hat)^2)
  MSE[k,2] <- mean((y.test-y.test.hat)^2)
}

plot(MSE[,1],col=1,ylim = range(MSE))
points(MSE[,2],col=2)
points(which.min(MSE[,2]),min(MSE[,2]),col=3,cex=2)
abline(v=which.min(MSE[,2]))

###########################################################################
# subset selection


rm(list=ls(all=TRUE))
# read data
#prostate <- read.table("C:\\Users\\dyang\\Dropbox\\588\\HTF\\data\\prostate.data")
prostate <- read.table("~/Documents/Financial\ Data\ Mining /prostate.data.txt")



install.packages("leaps")
# to use function regsubsets
library(leaps)

# check data
names(prostate)
# lpsa is the response
# the last column indicates training or test
dim(prostate)
plot(prostate)

# prepare data
train <- subset( prostate, train )
test  <- subset( prostate, train==FALSE )
train <- train[,-10]
test <- test[,-10]

x.train <- as.matrix(train[,-9])
y.train <- train[,9]
x.test <- as.matrix(test[,-9])
y.test <- test[,9]

dim(train)
dim(test)

# Best Subset Selection
regfit.full <- regsubsets(lpsa~.,train)
summary(regfit.full)
reg.summary <- summary(regfit.full)
names(reg.summary)
reg.summary$rsq

# If you were to do the subset selection by yourself
for(i in 1:8){
      cat(i,(summary(lm(y.train~x.train[,i]))$sigma)^2*(nrow(train)-2),"\n")
}

for(i in 1:7){
      for(j in (i+1):8){
            cat(i,j,(summary(lm(y.train~x.train[,c(i,j)]))$sigma)^2*(nrow(train)-3),"\n")
      }
}

par(mfrow=c(2,2))
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(which.max(reg.summary$adjr2),reg.summary$adjr2[which.max(reg.summary$adjr2)], col="red",cex=2,pch=20)
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(which.min(reg.summary$cp),reg.summary$cp[which.min(reg.summary$cp)],col="red",cex=2,pch=20)
plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
which.min(reg.summary$bic)
points(which.min(reg.summary$bic),reg.summary$bic[which.min(reg.summary$bic)],col="red",cex=2,pch=20)

# built-in plot() command which can be used to display the selected variables
# for the best model with a given number of predictors, ranked according to
# the BIC, Cp, adjusted R2, or AIC
quartz()
par(mfrow=c(2,2))
plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")

# find coefficient
coef(regfit.full,6)

# Forward and Backward Stepwise Selection

regfit.fwd<-regsubsets(lpsa~.,data=train,method="forward")
summary(regfit.fwd)
regfit.bwd<-regsubsets(lpsa~.,data=train,method="backward")
summary(regfit.bwd)
k=6
coef(regfit.full,k)
coef(regfit.fwd,k)
coef(regfit.bwd,k)

###########################################################################

# Ridge Regression

library(glmnet)
lambda.grid <- 10^seq(4,-3,length=100)
ridge.mod <- glmnet(x.train,y.train,alpha=0,lambda=lambda.grid)   #alpha=0:ridge
# the defaulty is standardize = TRUE, intercept=TRUE
coeff.matrix <- coef(ridge.mod)
dim(coeff.matrix)
# To get Figure 3.8: ridge path
plot(coeff.matrix[2,],ylim=c(min(coeff.matrix[-1,]),max(coeff.matrix[-1,])),col=2,type="o",ylab="coefficients")
for(i in 2:9){
  lines(coeff.matrix[i,],col=i,type="o")
}
windows()
# or
plot(ridge.mod)

# check coefficients
ridge.mod$lambda[50]
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2))

ridge.mod$lambda[60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))
#The larger the lambda, the smaller the l2 norm of the coefficient vector 


# we can obtain the ridge regression coefficients for a new value of lambda by setting s, which is the same as lambda
predict(ridge.mod,s=2,type="coefficients")

# we get predictions for a test set, by replacing type="coefficients" with the 
# newx argument.
ridge.pred<-predict(ridge.mod,s=.5,newx=x.test)
mean((ridge.pred-y.test)^2)

# if we had instead simply fit a model with just an intercept
mean((mean(y.train)-y.test)^2)

# We could also get the same result by fitting a ridge regression model with
# a very large value of lambda
ridge.pred <- predict(ridge.mod,s=1e10,newx=x.test)
mean((ridge.pred-y.test)^2)

# Recall that least squares is simply ridge regression with lambda = 0
ridge.pred <- predict(ridge.mod,s=0,newx=x.test)
mean((ridge.pred-y.test)^2)
lm(y.train~x.train)
predict(ridge.mod,s=0, exact = TRUE,type="coefficients")

# check the effect of centering and standardization
ridge.mod2 <- glmnet(x.train,y.train,alpha=0,lambda=lambda.grid, standardize = TRUE, intercept=FALSE)   #alpha=0:ridge
ridge.mod3 <- glmnet(x.train,y.train,alpha=0,lambda=lambda.grid, standardize = FALSE, intercept=TRUE)   #alpha=0:ridge
ridge.mod4 <- glmnet(x.train,y.train,alpha=0,lambda=lambda.grid, standardize = FALSE, intercept=FALSE)   #alpha=0:ridge
predict(ridge.mod,s=2,type="coefficients")
predict(ridge.mod2,s=2,type="coefficients")
predict(ridge.mod3,s=2,type="coefficients")
predict(ridge.mod4,s=2,type="coefficients")
mean((predict(ridge.mod,s=2,newx=x.test)-y.test)^2)
mean((predict(ridge.mod2,s=2,newx=x.test)-y.test)^2)
mean((predict(ridge.mod3,s=2,newx=x.test)-y.test)^2)
mean((predict(ridge.mod4,s=2,newx=x.test)-y.test)^2)


# choose lambda
test.error <- rep(NA, 100)
for(i in 1:100){
  ridge.pred <- predict(ridge.mod,s=lambda.grid[i],newx=x.test)
  test.error[i] <- mean((ridge.pred-y.test)^2)
}
plot(log(lambda.grid),test.error)
which.min(test.error)  
lambda.grid[which.min(test.error)]
test.error[which.min(test.error)]

# CV: come back when teaching Chap. 7
set.seed(1)
cv.out <- cv.glmnet(x.train,y.train,alpha=0)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam
ridge.pred <- predict(ridge.mod,s=bestlam,newx=x.test)
mean((ridge.pred-y.test)^2)
























