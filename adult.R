library(SuperLearner)
library(ggplot2)
library(caret)

# for reproducibility
set.seed(777) 

#ADULT DATASET
adult <- read.table('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 
                    sep = ',', fill = F, strip.white = T, na.strings = "?")
colnames(adult) <- c('age', 'workclass', 'fnlwgt', 'educatoin', 
                     'educatoin_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 
                     'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income')
adult = as.data.frame(adult)
adult = na.omit(adult)
adult$income <- as.numeric(factor(adult$income,levels = unique(adult$income)))-1 #BINARY
adult$workclass <- as.factor(adult$workclass)
adult$educatoin <- as.factor(adult$educatoin)
adult$marital_status <- as.factor(adult$marital_status)
adult$occupation <- as.factor(adult$occupation)
adult$relationship <- as.factor(adult$relationship)
adult$race <- as.factor(adult$race)
adult$sex <- as.factor(adult$sex)
adult$native_country <- as.factor(adult$native_country)

# TRAINING E TEST
Y = adult$income #RESPONSE
X = adult[-15] 

n = 500

train = sample.int(n=nrow(adult), size=n, replace=F)
test = sample.int(n=nrow(adult[-train]), size=n, replace=F)

# TRAINING
X_train = X[train,]
Y_train = Y[train]

# TEST
X_test = X[test,]
Y_test = Y[test]

# SUPERLEARNER
tune = list(ntrees = c(50, 200),
            max_depth = 3:5,
            shrinkage = c(0.001, 0.01))

learners = create.Learner("SL.xgboost", tune=tune, detailed_names = TRUE, 
                          name_prefix="xgb")

sl_lib = c("SL.xgboost", "SL.randomForest", "SL.mean", "SL.bartMachine", 
           "SL.glmnet", learners$names)

SL.Adult = SuperLearner(Y=Y_train, X=X_train, family=binomial(), 
                        SL.library=sl_lib, method="method.AUC", 
                        verbose=TRUE)
SL.Adult

# PREDICTION
predict <- predict(SL.Adult, X_test, onlySL=T)
predict <- as.factor(ifelse(predict$pred>=0.50, 1,0))
cm <- confusionMatrix(data=predict, as.factor(Y_test))
cm
cm <- as.data.frame(cm$table)
cm$Prediction <- factor(cm$Prediction, levels=rev(levels(cm$Prediction)))
ggplot(cm,aes(Reference, Prediction, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194")

# CV RISK
n = 500

train = sample.int(n=nrow(adult), size=n, replace=F)
test = sample.int(n=nrow(adult[-train]), size=n, replace=F)

# TRAINING
X_train = X[train,]
Y_train = Y[train]

CV.SL.Adult = CV.SuperLearner(Y=Y_train, X=X_train, family=binomial(), 
                              SL.library=sl_lib,method="method.AUC", 
                              parallel="multicore", verbose=TRUE)
summary(CV.SL.Adult)

plot(CV.SL.Adult) + theme_minimal()



