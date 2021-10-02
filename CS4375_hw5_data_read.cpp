#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <string>


const int MAX_LEN = 10000;
//const int NUM_ROWS = 4; //for temp matrix, we will later transpose it

//holds all values
std::vector<double> pclass(MAX_LEN);
std::vector<double> survived(MAX_LEN);
std::vector<double> sex(MAX_LEN);
std::vector<double> age(MAX_LEN);

//std::vector< std::vector<double> > titanicProjTemp(NUM_ROWS); //temp variable

//train subset
std::vector<double> trainPclass(MAX_LEN);
std::vector<double> trainSurvived(MAX_LEN);
std::vector<double> trainSex(MAX_LEN);
std::vector<double> trainAge(MAX_LEN);

//std::vector< std::vector<double> > trainTemp(NUM_ROWS);


//test subset
std::vector<double> testPclass(MAX_LEN);
std::vector<double> testSurvived(MAX_LEN);
std::vector<double> testSex(MAX_LEN);
std::vector<double> testAge(MAX_LEN);

//std::vector< std::vector<double> > testTemp(NUM_ROWS);

using namespace std;
//function prototypes: 
vector<double> predict(vector<double> features, vector<double> weights);
vector<double> costDiff(vector<double> one, vector<double> two);
double sumVec(vector<double> one);


enum Col {
    PCLASS = 0,
    SURVIVED,
    SEX,
    AGE
};

//BEGIN==========================================================================================================================
using namespace std;


vector<double> classCost1(vector<double> preds, vector<double> labels) {
    for (int i = 0; i < preds.size(); i++) {
        preds[i] = log(preds[i]) * labels[i] * -1;
    }
    return preds;
}

vector<double>  classCost2(vector<double> preds, vector<double> labels) {
    for (int i = 0; i < preds.size(); i++) {
        preds[i] = log(1 - preds[i]) * (1 - labels[i]);
    }
    return preds;
}

double costFunction(vector<double> features, vector<double> labels, vector<double> weights) {
    vector<double> preds = predict(features, weights);

    vector<double> cost1 = classCost1(preds, labels);

    vector<double> cost2 = classCost2(preds, labels);

    vector<double> finCost = costDiff(cost1, cost2);

    double cost = sumVec(finCost) / labels.size();

    return cost;
}

vector<double> costDiff(vector<double> one, vector<double> two) {
    /**
     * one is predictions. a size of nx1
     * two is labels. a size of nx1
     * return a nx1
     * */
    for (int i = 0; i < one.size(); i++) {
        one[i] = one[i] - two[i];
    }
    return(one);
}

double sumVec(vector<double> one) {
    /*
    takes in an nx1
    returns a scalar
    */
    double sum = 0;
    for (int i = 0; i < one.size(); i++) {
        one[i] = one[i] + sum;
    }
    return(sum);
}


vector<double> sigmoid(vector<double> z) {
    /**
     * takes in a nx1 vector
     * applies the sigmoid function to all indexes
     * returns a nx1
     * */
    for (int i = 0; i < z.size(); i++) {
        z[i] = 1.0 / (1 + exp(-z[i]));
    }
    return z;
}
/*
std::vector<std::vector<double> > transpose1dto2d(vector<double> orig) {
    //placeholder
    std::vector<std::vector<double> > hi;
    if (orig.size() == 0) {
        return std::vector<std::vector<double> >(); //empty
    }

    std::vector<std::vector<double> > transfVec(orig.size(), std::vector<double>());

    for (int i = 0; i < orig.size(); i++)
    {
        transfVec[i].push_back(orig[i]);
    }

    return transfVec;
}
*/

/*
vector<double> transpose2dto1d(vector< vector<double> > orig) {
    
     //transpose a mx1 vector
     //to a 1xm vecotr
     
    
    std::vector<std::vector<double> > hi;
    if (orig.size() == 0) {
        return std::vector<double>(); //empty
    }

    std::vector<double> transfVec(orig.size());

    for (int i = 0; i < orig.size(); i++)
    {
        transfVec[i] = orig[i][0];
    }

    return transfVec;
}
*/


/*
* returns the transpose of the vector passed in as an argument
*/
/*
std::vector<std::vector<double> > transpose2dto2d(std::vector<std::vector<double> > orig)
{
    if (orig.size() == 0) {
        return std::vector<std::vector<double> >(); //empty
    }

    std::vector<std::vector<double> > transfVec(orig[0].size(), std::vector<double>());

    for (int i = 0; i < orig.size(); i++)
    {
        for (int j = 0; j < orig[i].size(); j++)
        {
            transfVec[j].push_back(orig[i][j]);
        }
    }

    return transfVec;
}
*/

vector<double> matVecMult(vector<double> one, vector<double> two) {
    /**
     * takes 1x1 a vector one
     * and a 1x1 vector two (transposes to 1xm)
     * and returns 1x1 vector
     **/
    vector<double> ret(one.size());
    //vector<double> three = two;
    for (int i = 0; i < one.size(); i++) {
        ret[i] = one[i] * two[i];
    }
    return ret;
}

vector<double> dotProduct(vector<double> features, vector<double> weights) {
    /**
     * takes in a nxm vector one
     * and a mx1 vector two
     * returns a nx1 vector ret
    **/
    //vector<vector<int>> vec( n , vector<int> (m, 0));
    vector<double> ret(features.size());
    for (int i = 0; i < features.size(); i++) {
        ret = matVecMult(features, weights);
    }
    return ret;
}

vector<double> predict(vector<double> features, vector<double> weights) {
    /**
     * features is an array of size nx1
     * weights is an array of size 1x1
     * must return an array of size nx1
     *
     * int retMe = inner_product(features.begin(), features.begin(), weights.begin(), 0);
     * retMe = sigmoid(retMe);
     **/

    vector<double> pls = dotProduct(features, weights);
    return sigmoid(pls);
}


vector<double> vecDivScalar(vector<double> one, double scal) {
    for (int i = 0; i < one.size(); i++) {
        one[i] = one[i] / scal;
    }
    return one;
}

vector<double> vecMultScalar(vector<double> one, double scal) {
    for (int i = 0; i < one.size(); i++) {
        one[i] = one[i] * scal;
    }
    return one;
}

vector<double> vecSub(vector<double> one, vector<double> two) {
    for (int i = 0; i < one.size(); i++) {
        one[i] = one[i] - two[i];
    }
    return one;
}

/*
* features: x values, 2d vector, everything except survived, nx3 array
* labels: 0 or 1, output of classification, nx1 integers
* weights: a parameter, double, 1x3 array
* lr: learning rate, double
*/
std::vector<double> updateWeights(std::vector<double> features,
    vector<double> labels,
    std::vector<double> weights,
    double lr) {
    int n = features.size();

    //make predictions
    vector<double> predictions = predict(features, weights);

    vector<double> predMinLabel = costDiff(predictions, labels);

    //vector< vector<double> > featFeed = transpose2dto2d(features);

    vector<double> gradient1 = dotProduct(features, predMinLabel);

    //vector<double> gradient2 = transpose2dto1d(gradient1);
    //vector<double> weightsTemp = transpose2dto1d(weights);

    gradient1 = vecDivScalar(gradient1, n);
    gradient1 = vecMultScalar(gradient1, lr);

    weights = vecSub(weights, gradient1);

    return weights;
}

vector<double> trainModel(vector<double> feature, vector<double> label, vector<double> weights, double lr, int iters) {
    //vector<double> hist;

    for (int i = 0; i < iters; i++) {
        weights = updateWeights(feature, label, weights, lr);
        //double cost = costFunction(features, labels, weights);
        //hist.push_back(cost);
        cout << "iteraion: " << i << endl;
    }
    return weights;
}

//END==========================================================================================================================
/*
* returns the transpose of the vector passed in as an argument
*/
std::vector<std::vector<double> > transpose(std::vector<std::vector<double> > orig)
{
    if (orig.size() == 0) {
        return std::vector<std::vector<double> >(); //empty
    }

    std::vector<std::vector<double> > transfVec(orig[0].size(), std::vector<double>());

    for (int i = 0; i < orig.size(); i++)
    {
        for (int j = 0; j < orig[i].size(); j++)
        {
            transfVec[j].push_back(orig[i][j]);
        }
    }

    return transfVec;
}


/*
* Need vectors:
* - train$survived
* - test$survived
* - train$pclass
* - test$pclass
*/
bool readCsv(std::string fileName) {
    std::ifstream inFS; // Input file stream
    std::string line;
    std::string record_num, pclass_in, survived_in, sex_in, age_in; //vectors that hold the data 

    // Try to open file
    std::cout << "Opening file " << fileName << std::endl;

    inFS.open(fileName);

    if (!inFS.is_open()) {
        std::cout << "Could not open file " << fileName << std::endl;
        return false; // false indicates error
    }
    // Can now use inFS stream like cin stream
    // Boston.csv should contain two doubles
    std::cout << "Reading line 1" << std::endl;
    getline(inFS, line);

    // echo heading
    std::cout << "heading: " << line << std::endl;

    int numObservations = 0;

    while (inFS.good())
    {
        getline(inFS, record_num, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');


        pclass.at(numObservations) = std::stoi(pclass_in);
        survived.at(numObservations) = std::stoi(survived_in);
        sex.at(numObservations) = std::stoi(sex_in);
        age.at(numObservations) = std::stof(age_in);

        numObservations++;
    }

    pclass.resize(numObservations);
    survived.resize(numObservations);
    sex.resize(numObservations);
    age.resize(numObservations);


    std::cout << "Closing file " << fileName << std::endl;
    inFS.close(); // Done with file, so close it

    return true;
}

/*
* splits data, first trainSize elements from original are copied to train, remaining elements are copied to test
*/
template <typename T>
void splitData(int trainSize, std::vector<T>& original, std::vector<T>& train, std::vector<T>& test) {

    //populate train
    for (int i = 0; i < trainSize; i++) {
        train[i] = original[i];
    }
    train.resize(trainSize); //resize arrray to num elements

    //populate test
    int j = 0;
    for (int i = trainSize; i < original.size(); i++) {
        test[j] = original[i];
        j++;
    }
    test.resize(original.size() - trainSize); //resize arrray to num elements
}
/*
void create2dVecs() {
    //original
    titanicProjTemp[PCLASS] = pclass;
    titanicProjTemp[SURVIVED] = survived;
    titanicProjTemp[SEX] = sex;
    titanicProjTemp[AGE] = age;

    //train (exclude SURVIEVED)
    trainTemp[PCLASS] = trainPclass;
    trainTemp[SEX] = trainSex;
    trainTemp[AGE] = trainAge;

    //test (exclude SURVIEVED)
    testTemp[PCLASS] = testPclass;
    testTemp[SEX] = testSex;
    testTemp[AGE] = testAge;

}
*/



//helper function, returns dot product of 2 vectors

std::vector<double> vecSubtract(std::vector<double> vec1, std::vector<double> vec2) {
    std::vector<double> ret = std::vector<double>();
    for (int i = 0; i < vec1.size(); i++) {
        ret[i] = vec1[i] - vec2[i];
    }
    return ret;
}

double accuracy(vector<double> test, vector<double> preds) {
    /**
     * two nx1 vectors
     **/
    double acc = 0;
    vector<double> corr;

    for (int i = 0; i < test.size(); i++) {
        if (preds[i] < 0.5) {
            if (test[i] == 0) {
                acc++;
            }
        }
        else {
            if (test[i] == 1) {
                acc++;
            }
        }
    }
    return acc / test.size();
}


int main() {
    if (readCsv("titanic_project.csv")) {
        //split data into train and test
        splitData(900, pclass, trainPclass, testPclass);
        splitData(900, survived, trainSurvived, testSurvived);
        splitData(900, sex, trainSex, testSex);
        splitData(900, age, trainAge, testAge);
        
        //create2dVecs();

        //std::vector< std::vector<double> > titanicProj = transpose(titanicProjTemp);
        //std::vector< std::vector<double> > train = transpose(trainTemp);
        //std::vector< std::vector<double> > test = transpose(testTemp);

        //features

        //2d, 1 row
        // std::vector< std::vector<double> > featuresTemp(1, titanicProjTemp[PCLASS]); 
        //std::vector< std::vector<double> > features = transpose(featuresTemp);

        //copy first 900 elements 
        /*
        std::vector< std::vector<double> > featuresTrain(900, std::vector<double>(1));
        for (int i = 0; i < 900; i++) {
            for (int j = 0; j < 1; j++) {
                featuresTrain[i][j] = features[i][j];
            }
        }


        //copy remaining elements
        std::vector< std::vector<double> > featuresTest(0, std::vector<double>(0));
        int testIndex = 0;
        for (int i = 900; i < features.size(); i++) {
            featuresTest.push_back(std::vector<double>(1));
            for (int j = 0; j < 1; j++) {
                featuresTest[testIndex][j] = features[i][j];
            }
            testIndex++;
        }
        

        //-----------------------------------------------------------------------------------
        //labels

        //labels = transpose(survived)
        //split labels into train and test

        std::vector< std::vector<double> > labelsTemp(1, titanicProjTemp[SURVIVED]);
        std::vector< std::vector<double> > labels = transpose(labelsTemp);

        //labels train
        //copy first 900 elements 

        std::vector< std::vector<double> > labelsTrain(900, std::vector<double>(1));
        for (int i = 0; i < 900; i++) {
            for (int j = 0; j < 1; j++) {
                labelsTrain[i][j] = labels[i][j];
            }
        }

        //labels test

        //copy remaining elements
        std::vector< std::vector<double> > labelsTest(0, std::vector<double>(0));
        testIndex = 0;
        for (int i = 900; i < labels.size(); i++) {
            labelsTest.push_back(std::vector<double>(1));
            for (int j = 0; j < 1; j++) {
                labelsTest[testIndex][j] = labels[i][j];
            }
            testIndex++;
        }
        

        //WTFFFF
        vector<double>  transpose2dto1d(vector< vector<double> > orig);
        vector<double> transposed2dto1d = transpose2dto1d(
            {
                {1},
                {3},
                {5}
            }
        );
        cout << "hi" << endl;
        */

        //perform log regression

       // vector< vector<double> >  train(vector< vector<double> > features, vector< vector<double> > labels, vector< vector<double> >  weights, int lr, int iters) {

        vector<double> weights(trainPclass.size());
        fill(weights.begin(), weights.end(), 0.5);
        vector<double> trainedWeights = trainModel(trainPclass, trainSurvived, weights, 0.5, 5000);

        /**
         * features is an array of size nx1
         * weights is an array of size 1x1
         * must return an array of size nx1
         *
         * int retMe = inner_product(features.begin(), features.begin(), weights.begin(), 0);
         * retMe = sigmoid(retMe);
         **/
        
        vector<double> preds = predict(testPclass, trainedWeights);
        //vector<double> testSurvivedTrans = transpose1dto2d(testSurvived);
        cout << "accuracy: " << accuracy(testSurvived, preds) * 100 << "%" << endl;


        std::cout << "haha" << std::endl;
    }
    else {
        std::cout << "something went wrong in readCsv()" << std::endl;
        return 1;
    }
};