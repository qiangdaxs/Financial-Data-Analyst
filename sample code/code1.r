rm(list=ls(all=TRUE))
# read data
#load("C:\\Users\\dyang\\Dropbox\\588\\HTF\\data\\ESL.mixture.rda")
load("D:/Dropbox/588/HTF/data/ESL.mixture.rda")
dataall <- ESL.mixture
x.train <- dataall$x
y.train <- dataall$y

#x	       200 x 2 matrix of training predictors
#y	       class variable; logical vector of TRUES and 
#	       FALSES - 100 of each
#xnew	       matrix 6831 x 2 of lattice points in predictor space
#prob	       vector of 6831 probabilities (of class TRUE) at each 
#	       lattice point
#marginal       marginal probability at each lattice point
#px1	       69 lattice coordinates for x.1
#px2	       99 lattice values for x.2  (69*99=6831)
#means	       20 x 2 matrix of the mixture centers, first ten for one
#	       class, next ten for the other


# plot the raw data
#pdf("raw_data.pdf")
plot(x.train,col=c("blue","orange")[y.train+1],xlab="x1",ylab="x2")
#dev.off()

# plot the bayes decition boundary with heat map
windows()
image(dataall$px1,dataall$px2,matrix(dataall$prob,length(dataall$px1),length(dataall$px2)))
contour(dataall$px1,dataall$px2,matrix(dataall$prob,length(dataall$px1),length(dataall$px2)),add=TRUE,levels=.5)

# plot the bayes decition boundary with raw data
windows()
plot(x.train,col=c("blue","orange")[y.train+1],xlab="x1",ylab="x2")
contour(dataall$px1,dataall$px2,matrix(dataall$prob,length(dataall$px1),length(dataall$px2)),add=TRUE,levels=.5)

# generate the test data
n.test <- 10000
ind <- sample(1:10,n.test,replace=TRUE)
means <- rbind(dataall$means[ind[1:(n.test/2)],],dataall$means[ind[(n.test/2+1):n.test]+10,])
x.test <- means + matrix(rnorm(n.test*2)/sqrt(5),n.test,2)
y.test <- c(rep(0,n.test/2),rep(1,n.test/2))
windows(width=10,height=6)
par(mfrow=c(1,2))
plot(x.train,col=c("blue","orange")[y.train+1],xlab="x1",ylab="x2",main="training data")
plot(x.test,col=c("blue","orange")[y.test+1],xlab="x1",ylab="x2",main="testing data")

# OLS
lm1 <- lm(y.train~x.train)
y.hat <- predict(lm1)
y.test.hat <- cbind(1,x.test) %*% coef(lm1)
test.error.lm <- mean((y.test.hat>.5)!=y.test); test.error.lm
train.error.lm <- mean((y.hat>.5)!=y.train); train.error.lm

# NN
#install.packages("class")
library(class)
k.vec <- c(1,3,5,7,9,11,21,31,45,69,101,151)
n.k <- length(k.vec)
test.error <- rep(NA, n.k)
train.error <- rep(NA, n.k)
for(i in 1:n.k){
      y.hat<- knn(x.train, rbind(x.train,x.test), y.train, k = k.vec[i], prob=FALSE)
      train.error[i] <- mean(y.hat[1:200]!=y.train)
      test.error[i] <- mean(y.hat[201:(200+n.test)]!=y.test)
      print(i)
}

# plot the errors
plot(x=200/k.vec, test.error,col="orange",ylim=c(0,max(test.error)),type="o")
points(x=200/k.vec, train.error,col="blue",type="o")
points(3,test.error.lm, col="orange",pch=2)
points(3,train.error.lm, col="blue",pch=2)
















