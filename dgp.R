library(SuperLearner)
library(ggplot2)

# for reproducibility
set.seed(777) 

# Generated Data
n <- 500
noise <- rnorm(n)
X <- runif(n, -10, 10)

Y1 <- (-2*(ifelse(X<(-5),1,0))+4*(ifelse(X>(-2),1,0))
       -2.4*(ifelse(X>0,1,0))+7*(ifelse(X>6,1,0))+noise)

Y2 <- (4+2*cos(X)+0.125*(X^2)+4*sin(X^5)+noise)

par(mfrow=c(1,2)) 
plot(X,Y1)
plot(X,Y2)

X = as.data.frame(X)

sl_lib = c("SL.xgboost", "SL.randomForest", "SL.svm", 
           "SL.kernelKnn", "SL.mean", "SL.glm")

SL.1 <- SuperLearner(Y=Y1, X=X, family=gaussian(), method="method.NNLS", 
                     SL.library=sl_lib, verbose=TRUE)

SL.2 <- SuperLearner(Y=Y2, X=X, family=gaussian(), method="method.NNLS",
                     SL.library=sl_lib, verbose=TRUE)

SL.1
SL.2


#META-FEATURES
SL.1$Z
SL.2$Z

#CV SuperLearner
CV.SL.1 <- CV.SuperLearner(Y=Y1, X=X, family=gaussian(), method="method.NNLS", 
                           SL.library=sl_lib, verbose=TRUE)

CV.SL.2 <- CV.SuperLearner(Y=Y2, X=X, family=gaussian(), method="method.NNLS", 
                           SL.library=sl_lib, verbose=TRUE)

CV.SL.1
CV.SL.2

par(mfrow=c(1,2)) 
plot(CV.SL.1)
plot(CV.SL.2)





