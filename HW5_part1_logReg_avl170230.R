## Step 1 Load the data
  
# your code here
library(caret)
titanProj <- read.csv('titanic_project.csv')



## Step 2 Divide train/test


train <- titanProj[1:900,] # first 900 elements of array
test <- titanProj[901:length(titanProj[,1]) ,] #remaining elements of array

## Step 3 Build a Model (Logistic regression) 


# your code here
ptm <- proc.time()
glm1 <- glm(survived~pclass, data=train, family=binomial)
proc.time() - ptm
summary(glm1)


## Step 4 Predict probabilities
# your code here


#probs of surviving 
probs <- predict(glm1, newdata=test, type="response")

#prediction & accuracy
pred <- ifelse(probs>0.5, 1, 0)
acc <- mean(pred==test$survived)
print(paste("accuracy = ", acc))

#for sensitivity and specificity, compare pred with test$survived
#if(!require("caret")) {
#  install.packages("caret")
#}
library(caret)

lm1Sensitivity <- sensitivity(as.factor(pred), as.factor(test$survived))
lm1Specificity <- specificity(as.factor(pred), as.factor(test$survived))

print(paste("logistic regression sensitivity: ", lm1Sensitivity))
print(paste("logistic regression specificity: ", lm1Specificity))


