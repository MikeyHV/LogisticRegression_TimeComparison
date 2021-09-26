#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <string>


const int MAX_LEN = 10000;
const int NUM_COLS = 4;

//holds all values
std::vector<double> pclass(MAX_LEN);
std::vector<double> survived(MAX_LEN);
std::vector<double> sex(MAX_LEN);
std::vector<double> age(MAX_LEN);

std::vector< std::vector<double> > titanicProj(NUM_COLS);


//train subset
std::vector<double> trainPclass(MAX_LEN);
std::vector<double> trainSurvived(MAX_LEN);
std::vector<double> trainSex(MAX_LEN);
std::vector<double> trainAge(MAX_LEN);

std::vector< std::vector<double> > train(NUM_COLS);


//test subset
std::vector<double> testPclass(MAX_LEN);
std::vector<double> testSurvived(MAX_LEN);
std::vector<double> testSex(MAX_LEN);
std::vector<double> testAge(MAX_LEN);

std::vector< std::vector<double> > test(NUM_COLS);

enum Col {
    PCLASS = 0,
    SURVIVED,
    SEX,
    AGE
};

/*
* Need vectors:
*     - train
*     - test
*     - pclass
*     - survived
*     - sex
*     - age
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
    for (int i = trainSize; i < original.size(); i++) {
        test[i] = original[i];
    }
    test.resize(original.size() - trainSize); //resize arrray to num elements
}
void create2dVecs() {
    //original
    titanicProj[PCLASS] = pclass;
    titanicProj[SURVIVED] = survived;
    titanicProj[SEX] = sex;
    titanicProj[AGE] = age;

    //train (exclude SURVIEVED)
    train[PCLASS] = trainPclass;
    train[SEX] = trainSex;
    train[AGE] = trainAge;
    
    //test (exclude SURVIEVED)
    test[PCLASS] = testPclass;
    test[SEX] = testSex;
    test[AGE] = testAge;

}

/*
* helper function, returns dot product of 2 vectors
*/
std::vector<double> vecSubtract(std::vector<double> vec1, std::vector<double> vec2) {
    std::vector<double> ret = std::vector<double>();
    for (int i = 0; i < vec1.size(); i++) {
        ret[i] = vec1[i] - vec2[i];
    }
    return ret;
}
/*
* features: x values, 2d vector, everything except survived, nx3 array
* labels: 0 or 1, output of classification, nx1 integers
* weights: a parameter, double, 1x3 array
* lr: learning rate, double
*/
std::vector<double> updateWeights(std::vector< std::vector<double> >& features, 
                                std::vector<double>& labels, 
                                std::vector<double>& weights, 
                                double lr) {
    int n = features.size();

    //make predictions
    std::vector<double> predictions = predict(features, weights);
    

    double gradient = dotProduct(transpose(features[0]), vecSubtract(predictions, labels));
    gradient = gradient / n;
    gradient = gradient * lr;
    weights = vecSubtract(weights, gradient);

    return weights;
}


int main() {
    if (readCsv("titanic_project.csv")) {
        //split data into train and test
        splitData(900, pclass, trainPclass, testPclass);
        splitData(900, survived, trainSurvived, testSurvived);
        splitData(900, sex, trainSex, testSex);
        splitData(900, age, trainAge, testAge);

        create2dVecs();

        //perform log regression

        std::cout << "haha" << std::endl;
    }
    else {
        std::cout << "something went wrong in readCsv()" << std::endl;
        return 1;
    }
};