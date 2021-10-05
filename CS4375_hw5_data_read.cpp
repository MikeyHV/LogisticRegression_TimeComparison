#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <string>
#include <chrono>


const int MAX_LEN = 10000;
//const int NUM_ROWS = 4; //for temp matrix, we will later transpose it

//holds all values
std::vector<double> pclass(MAX_LEN);
std::vector<double> survived(MAX_LEN);
std::vector<double> sex(MAX_LEN);
std::vector<double> age(MAX_LEN);

//train subset
std::vector<double> trainPclass(MAX_LEN);
std::vector<double> trainSurvived(MAX_LEN);
std::vector<double> trainSex(MAX_LEN);
std::vector<double> trainAge(MAX_LEN);

//test subset
std::vector<double> testPclass(MAX_LEN);
std::vector<double> testSurvived(MAX_LEN);
std::vector<double> testSex(MAX_LEN);
std::vector<double> testAge(MAX_LEN);

using namespace std;


double sensitivity(vector<double> predicted, vector<double> actual) {
    double numTruePos = 0;
    double numActuallyTrue = 0;
    for (int i = 0; i < predicted.size(); i++) {
        if (predicted[i] == actual[i] == 1) {
            numTruePos++;
        }
        if (actual[i] == 1) {
            numActuallyTrue++;
        }
    }
    return numTruePos / numActuallyTrue * 100;
}

double specificity(vector<double> predicted, vector<double> actual) {
    double numTrueNeg = 0;
    double numActuallyNeg = 0;
    for (int i = 0; i < predicted.size(); i++) {
        if (predicted[i] == actual[i] == 0) {
            numTrueNeg++;
        }
        if (actual[i] == 0) {
            numActuallyNeg++;
        }
    }
    return numTrueNeg / numActuallyNeg * 100;
}

//function prototypes: 
vector<double> predict(vector<double> features, vector<double> weights);
vector<double> costDiff(vector<double> one, vector<double> two);
double sumVec(vector<double> one);

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
        ret[i] = features[i] * weights[i]; //matVecMult(features, weights);
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

vector<double> vecSums(vector<double> one, vector<double> two) {
    for (int i = 0; i < one.size(); i++) {
        one[i] = one[i] + two[i];
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

    //vector<double> error = costDiff(predictions, labels);
    vector<double> error = costDiff(labels, predictions);
    //vector< vector<double> > featFeed = transpose2dto2d(features);

    vector<double> gradient1 = dotProduct(features, error);

    //gradient1 = vecDivScalar(gradient1, n);
    gradient1 = vecMultScalar(gradient1, lr);

    //weights = vecSub(weights, gradient1);
    weights = vecSums(weights, gradient1);

    return weights;
}

vector<double> trainModel(vector<double> feature, vector<double> label, vector<double> weights, double lr, int iters) {
    //vector<double> hist;

    for (int i = 0; i < iters; i++) {
        weights = updateWeights(feature, label, weights, lr);
        cout << "iteration: " << i << endl;
    }
    return weights;
}

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
    vector<double> predictionsAsFactor(0);

    for (int i = 0; i < test.size(); i++) {
        if (preds[i] < 0.70) {
            if (test[i] == 1) {
                acc++;
            }
            predictionsAsFactor.push_back(1);
        }
        else {
            if (test[i] == 0) {
                acc++;
            }
            predictionsAsFactor.push_back(0);
        }
    }
    std::cout << "sensitivity: " << sensitivity(predictionsAsFactor, test) << std::endl;
    std::cout << "specificity: " << specificity(predictionsAsFactor, test) << std::endl;
    return acc / test.size();
}

double accuracy2(vector<double> test, vector<double> pred) {
    double numCorrect = 0;
    vector<int> testAsFactor(test.size());
    vector<int> predAsFactor(test.size());

    //convert to integer factor 
    for (int i = 0; i < test.size(); i++) {
        if (test[i] > 0) {
            testAsFactor[i] = 1;
        }
        else {
            testAsFactor[i] = 0;
        }
    }

    for (int i = 0; i < pred.size(); i++) {
        if (pred[i] > 0.5) {
            predAsFactor[i] = 1;
        }
        else {
            predAsFactor[i] = 0;
        }
    }

    for (int i = 0; i < pred.size(); i++) {
        if (predAsFactor[i] == testAsFactor[i]) {
            numCorrect++;
        }
    }

    return numCorrect / test.size();
}


int main() {
    if (readCsv("titanic_project.csv")) {
        //split data into train and test
        splitData(900, pclass, trainPclass, testPclass);
        splitData(900, survived, trainSurvived, testSurvived);
        splitData(900, sex, trainSex, testSex);
        splitData(900, age, trainAge, testAge);

        vector<double> weights(trainPclass.size());
        fill(weights.begin(), weights.end(), 0.5);

        //algorithm start time
        auto start = chrono::high_resolution_clock::now();

        vector<double> trainedWeights = trainModel(trainPclass, trainSurvived, weights, 0.001, 50);
        //cout << trainedWeights[0] << endl;

        //end/stop algorithm time
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        std::cout << "Time taken by function: " << duration.count() << " microseconds" << endl;

        vector<double> preds = predict(testPclass, trainedWeights);

        for (int i = 0; i < preds.size(); i++) {
            cout << preds[i] << endl;
        }
        cout << "accuracy: " << accuracy(testSurvived, preds) * 100 << "%" << endl;

        std::cout << "haha" << std::endl;
    }
    else {
        std::cout << "something went wrong in readCsv()" << std::endl;
        return 1;
    }
};