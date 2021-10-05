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

double accuracy(vector<double> test, vector< vector<double> > preds) {
    /**
     * two nx1 vectors
     **/
    double acc = 0;
    vector<double> corr;

    for (int i = 0; i < test.size(); i++) {
        double fir = preds[i][0];
        double sec = preds[i][1];
        double max = fir;
        if (fir > sec) {
            max = fir;
        }
        else {
            max = sec;
        }
        if (max <= 0.776) {
            if (test[i] == 0) {
                acc++;
            }
        }else {
            if (test[i] == 1) {
                acc++;
            }
        }
        cout << max << " " << test[i]  << " " << acc << endl;
    }
    return acc / test.size();
}

vector< vector<double> > posteriorDiscreteSex(vector<double> sex, vector<double> survived){
    double ms = 0;
    double fs = 0;
    double md = 0;
    double fd = 0;
    double diedNum = 0;
    double surviveNum = 0;
    for(int i = 0; i < survived.size(); i++){
        if(sex[i] == 1){
            if(survived[i] == 1){
                ms++;
                surviveNum++;
            }else{
                md++;
                diedNum++;
            }
        }
        if(sex[i] == 0){
            if(survived[i] == 1){
                fs++;
                surviveNum++;
            }else{
                fd++;
                diedNum++;
            }
        }
    }
    double probmS = ms/surviveNum;
    double probfS = fs/surviveNum;
    double probmD = md/diedNum;
    double probfD = fd/diedNum;
    vector< vector<double> > fin { { probmD, probfD },
                                    { probmS, probfS } };
    return fin;
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
    vector< vector<double> > fin { { prob1D, prob2D, prob3D },
                                    { prob1S, prob2S, prob3S } };
    return fin;
}

//returns summation of parameter inputVector 
double vectorSum(const std::vector<double>& inputVector) {
    double sum = 0;
    for (double curr : inputVector) {
        sum += curr;
    }
    return sum;
}

//returns mean of parameter inputVector 
double vectorMean(const std::vector<double>& inputVector) {
    double mean = 0;
    double sum = vectorSum(inputVector);
    mean = sum / inputVector.size();
    return mean;
}

//returns variance of vector parameter inputVector
double vectorVariance(const std::vector<double>& inputVector) {
    double summation = 0;
    double xAvg = 0;
    double xSum = 0;
    int n = inputVector.size();
    double variance = 0;

    //calculate avg
    for (int i = 0; i < n; i++) {
        xSum += inputVector[i];
    }
    xAvg = xSum / n;

    //calculate variance
    for (int i = 0; i < n; i++) {
        summation += ((inputVector[i] - xAvg) * (inputVector[i] - xAvg)); //(xi - xAvg)
    }
    variance = summation / (n - 1);
    return variance;
}

//computes likelihood of continuous vector
vector<double> likelihoodContinuous1d(vector<double> x) {
    double mean = vectorMean(x);
    double variance = vectorVariance(x);
    double myPi = atan(1) * 4; //pi

    vector<double> likelihoods(0);
    double expNumerator;
    double expDenominator;
    double temp; 
    for (double instance : x) {
        expNumerator = -1 * pow((instance - mean), 2);
        expDenominator = 2 * variance;
        temp = 1 / sqrt(2 * myPi * variance) * exp(expNumerator / expDenominator);
        likelihoods.push_back(temp);
    }
    cout << "age mean " << mean << endl;
    cout << "age variance " << variance << endl;
    return likelihoods;
}

vector< vector<double> > likelihoodContinuous(vector<double> age, vector<double> survived) {
    vector<double> survivedAges(0);
    vector<double> notSurvivedAges(0);

    //populate survived and notSurvived vectors
    for (int i = 0; i < survived.size(); i ++) {
        if (survived[i] == 1) { //if person survived, add their age to survivedAges
            survivedAges.push_back(age[i]);
        }
        else {
            notSurvivedAges.push_back(age[i]);
        }
    }

    //calculate likelihoods
    vector<double> agesMean = { vectorMean(notSurvivedAges), vectorMean(survivedAges) };
    vector<double> agesVar = { vectorVariance(notSurvivedAges), vectorVariance(survivedAges) };

    vector< vector<double> > ret(0);
    ret.push_back(agesMean);
    ret.push_back(agesVar);

    return ret; //ret[0] = likelihoodAgeSurvived, ret[1] = likelihoodAgeNotSurvived
}

vector<double> apriori(vector<double> data){
    double survived = 0;
    double notsurvived = 0;
    for(int i = 0; i < data.size(); i++){
        if(data[i] == 1){
            survived++;
        }else{
            notsurvived++;
        }
    }
    vector<double> apriori = { notsurvived / data.size(), survived / data.size() };
    return apriori;
}

double calcPAge(double age, double mean, double var){
    double expNumerator = -1 * pow((age - mean), 2);
    double expDenominator = 2 * var;
    double myPi = atan(1) * 4;
    return 1 / sqrt(2 * myPi * var) * exp(expNumerator / expDenominator);
}

vector <double> naiveBayes(double pclass, double sex, double age, 
                            vector< double > survived,
                            vector< vector<double> > weightsPclass, 
                            vector< vector<double> > weightsSex, 
                            vector< vector<double> > weightsAge){

    vector<double> agesMean = weightsAge[0];
    vector<double> agesVar = weightsAge[1];

    int intPclass = pclass;
    int intSex = sex;

    double pclassS = weightsPclass[1][intPclass-1];
    double sexS = weightsSex[1][intSex];
    double ageS = calcPAge(age, agesMean[1], agesVar[1]);

    double pclassD = weightsPclass[0][intPclass-1];
    double sexD = weightsSex[0][intSex];
    double ageD = calcPAge(age, agesMean[0], agesVar[0]);

    double pS = survived[1];
    double pD = survived[0];

    double featureProbsS = pclassS * sexS * ageS * pS;
    double featureProbsD = pclassD * sexD * ageD * pD;

    double denom = featureProbsS + featureProbsD;

    return { featureProbsS / denom, featureProbsD / denom };
}

/**
 * 
 * putting it all together:
 * 
 * nums = psurvivedPclass * psurvivedsex * psurvived(apriori) * p_agecalculation
 * numd = pdiedPclass * pdiedsec * pdied(apriori) * p_agecalculation
 * i dont fully understand above yet. what do i do about the 3 different classes and 2 different sexes?
 * 
 * denom = psurvivedPclass * psurvivedsex * psurvivedage * apriorisurvived +
 *         all of the above but died haha
 * 
 * return list(probablySurvived = nums/denom, probablityDeadDead = numd/denom)
 */

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

    vector< vector<double> > weightsPclass = posteriorDiscretePClass(trainPclass, trainSurvived);
    vector< vector<double> > weightsSex = posteriorDiscreteSex(trainSex, trainSurvived);
    vector< vector<double> > weightsAge = likelihoodContinuous(trainAge, trainSurvived);
    vector< double > aprioriS = apriori(trainSurvived);
    cout << "=================================================" << endl;
    for (auto i : weightsPclass) {
        for (auto j : i) {
            cout << j << " ";
        }
    }
    cout << endl;
    for (auto i : weightsSex) {
        for (auto j : i) {
            cout << j << " ";
        }
    }
    cout << endl;

    for (auto i : weightsAge) {
        for (auto j : i) {
            cout << j << " ";
        }
    }
    cout << endl;

    for (auto i : aprioriS) {
            cout << i << " ";
    }
    cout << endl;
    cout << "=================================================" << endl;

    

    vector< vector<double> > testProbs;
    // this is a nx2 vector. each row is an instance, column 1 is dead, 2 is survived.

    for(int i = 0; i < testSex.size(); i++){
        double sexi = testSex[i];
        double agei = testAge[i];
        double pclassi = testPclass[i];
        testProbs.push_back(naiveBayes(pclassi, sexi, agei, aprioriS, weightsPclass, weightsSex, weightsAge));
        //cout << "age " << agei << " pclass " << pclassi << endl;
    }
    cout << endl;
    for (int i = 0; i < testProbs.size()-1; i++) {
        
        //cout << testProbs[i][0] << ", " << testProbs[i][1] << endl;
    }
    cout << accuracy(testSurvived, testProbs) << endl;

};