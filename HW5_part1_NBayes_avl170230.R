
  ## Step 1 Load the data
  

#if(!require("caret")) {
#  install.packages("caret")
#}
library(caret)
# your code here
titanProj <- read.csv('titanic_project.csv')


## Step 2 Divide train/test


train <- titanProj[1:900,] # first 900 elements of array
test <- titanProj[901:length(titanProj[,1]) ,] #remaining elements of array

## Step 3 Build a Model (Logistic regression) 


# your code here
library(e1071)
ptm <- proc.time()
nb1 <- naiveBayes(survived~as.factor(pclass)+sex+age, data = train)
proc.time() - ptm
print(nb1)

## Step 4 Predict probabilities
# your code here


#probs of surviving 
probs <- predict(nb1, newdata=test)

#prediction & accuracy
pred <- ifelse(probs==1, 1, 0)
acc <- mean(pred==test$survived)
print(paste("accuracy = ", acc))

#for sensitivity and specificity, compare pred with test$survived
if(!require("caret")) {
  install.packages("caret")
}
library(caret)

nb1Sensitivity <- sensitivity(as.factor(pred), as.factor(test$survived))
nb1Specificity <- specificity(as.factor(pred), as.factor(test$survived))

print(paste("Naive Bayes sensitivity: ", nb1Sensitivity))
print(paste("Naive Bayes specificity: ", nb1Specificity))



