#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <string>


const int MAX_LEN = 10000;

//holds all values
std::vector<int> pclass(MAX_LEN);
std::vector<int> survived(MAX_LEN);
std::vector<int> sex(MAX_LEN);
std::vector<double> age(MAX_LEN);

//train subset
std::vector<int> trainPclass(MAX_LEN);
std::vector<int> trainSurvived(MAX_LEN);
std::vector<int> trainSex(MAX_LEN);
std::vector<double> trainAge(MAX_LEN);

//test subset
std::vector<int> testPclass(MAX_LEN);
std::vector<int> testSurvived(MAX_LEN);
std::vector<int> testSex(MAX_LEN);
std::vector<double> testAge(MAX_LEN);

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


int main() {
    if (readCsv("titanic_project.csv")) {
        //split data into train and test
        splitData(900, pclass, trainPclass, testPclass);
        splitData(900, survived, trainSurvived, testSurvived);
        splitData(900, sex, trainSex, testSex);
        splitData(900, age, trainAge, testAge);

        //perform log regression

        std::cout << "haha" << std::endl;
    }
    else {
        std::cout << "something went wrong in readCsv()" << std::endl;
        return 1;
    }
};