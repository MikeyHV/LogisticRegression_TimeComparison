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
//function prototypes: 

using namespace std;


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
    vector<double> corr;

    for (int i = 0; i < test.size(); i++) {
        if (preds[i] < 0.5) {
            if (test[i] == 1) {
                acc++;
            }
        }
        else {
            if (test[i] == 0) {
                acc++;
            }
        }
    }
    return acc / test.size();
}

vector< vector<double> > posteriorDiscretePClass(vector<double> pclass, vector<double> survived){
    /**
     * 
     * lh_pclass[survived+1, passClass] = sum(pclass && survived)/sum(survived)
     * 
     */
    double ones = 0;
    double twos = 0;
    double threes = 0;
    double oned = 0;
    double twod = 0;
    double threed = 0;
    double diedNum = 0;
    double surviveNum = 0;
    for(int i = 0; i < survived.size(); i++){
        if(pclass[i] == 1){
            if(survived[i] == 1){
                ones++;
                surviveNum++;
            }else{
                oned++;
                diedNum++;
            }
        }
        if(pclass[i] == 2){
            if(survived[i] == 1){
                twos++;
                surviveNum++;
            }else{
                twod++;
                diedNum++;
            }
        }
        if(pclass[i] == 3){
            if(survived[i] == 1){
                threes++;
                surviveNum++;
            }else{
                threed++;
                diedNum++;
            }
        }
    }
    double prob1S = ones/surviveNum;
    double prob2S = twos/surviveNum;
    double prob3S = threes/surviveNum;
    double prob1D = oned/diedNum;
    double prob2D = twod/diedNum;
    double prob3D = threed/diedNum;
    vector< vector<double> > fin { { prob1S, prob2S, prob3S }
                                    { prob1D, prob2D, prob3D } };
    return fin;
}

/**
 * 
 * calculate apriori later
 */

int main() {
    bool readSuccess = readCsv("titanic_project.csv");
    if (!readSuccess) {
        std::cout << "something went wrong in readCsv()" << std::endl;
        return 1;
    }
    //split data into train and test
    splitData(900, pclass, trainPclass, testPclass);
    splitData(900, survived, trainSurvived, testSurvived);
    splitData(900, sex, trainSex, testSex);
    splitData(900, age, trainAge, testAge);
};