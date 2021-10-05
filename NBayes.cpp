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
std::vector<double> pclassGlobal(MAX_LEN);
std::vector<double> survivedGlobal(MAX_LEN);
std::vector<double> sexGlobal(MAX_LEN);
std::vector<double> ageGlobal(MAX_LEN);

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
    vector<double> likelihoodAgeSurvived = likelihoodContinuous1d(survivedAges);
    vector<double> likelihoodAgeNotSurvived = likelihoodContinuous1d(notSurvivedAges);

    vector< vector<double> > ret(0);
    ret.push_back(likelihoodAgeSurvived);
    ret.push_back(likelihoodAgeNotSurvived);

    return ret; //ret[0] = likelihoodAgeSurvived, ret[1] = likelihoodAgeNotSurvived

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


        pclassGlobal.at(numObservations) = std::stoi(pclass_in);
        survivedGlobal.at(numObservations) = std::stoi(survived_in);
        sexGlobal.at(numObservations) = std::stoi(sex_in);
        ageGlobal.at(numObservations) = std::stof(age_in);

        numObservations++;
    }

    pclassGlobal.resize(numObservations);
    survivedGlobal.resize(numObservations);
    sexGlobal.resize(numObservations);
    ageGlobal.resize(numObservations);


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
<<<<<<< Updated upstream
        if (preds[i] < 0.5) {
            if (test[i] == 1) {
                acc++;
            }
        }
        else {
            if (test[i] == 0) {
                acc++;
            }
=======
        double fir = preds[i][0];
        double sec = preds[i][1];
        
        double max = fir;
        if (fir > sec) {
            max = fir;
        }
        else {
            max = sec;
        }
        //std::cout << max << ", " << test[i] << endl;
        if (max < 0.775) {
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
>>>>>>> Stashed changes
        }
    }
    std::cout << "sensitivity: " << sensitivity(predictionsAsFactor, test) << std::endl;
    std::cout << "specificity: " << specificity(predictionsAsFactor, test) << std::endl;
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
        if(pclass[i] == 0){
            if(survived[i] == 1){
                ms++;
                surviveNum++;
            }else{
                md++;
                diedNum++;
            }
        }
        if(pclass[i] == 1){
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
    vector< vector<double> > fin { { probmS, probfS }
                                    { probmD, probfD } };
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
    vector< vector<double> > fin { { prob1S, prob2S, prob3S }
                                    { prob1D, prob2D, prob3D } };
    return fin;
}

<<<<<<< Updated upstream
vector <double> naiveBayes(vector<double> pclass, vector<double> sex, vector<double> age, vector<double> survived){
    vector< vector<double> > postSex = posteriorDiscreteSex(sex, survived);
    
    vector< vector<double> > postClass = posteriorDiscretePClass(pclass, survived);

    
=======
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
    long n = inputVector.size();
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
    long one = static_cast<long>(1);
    variance = summation / (n - one);
    return variance;
    //return sqrt(variance);
   
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
    std::cout << "age mean " << mean << endl;
    std::cout << "age variance " << variance << endl;
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

    //cout << "featuresProbs " << featureProbsS << endl;
    //cout << "denom " << denom << endl;
    //cout << "featreProbsD " << featureProbsD << endl;



    return { featureProbsS / denom, featureProbsD / denom };
>>>>>>> Stashed changes
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
    splitData(900, pclassGlobal, trainPclass, testPclass);
    splitData(900, survivedGlobal, trainSurvived, testSurvived);
    splitData(900, sexGlobal, trainSex, testSex);
    splitData(900, ageGlobal, trainAge, testAge);

    //algorithm start time
    auto start = chrono::high_resolution_clock::now();

<<<<<<< Updated upstream
=======
    vector< vector<double> > weightsPclass = posteriorDiscretePClass(trainPclass, trainSurvived);
    vector< vector<double> > weightsSex = posteriorDiscreteSex(trainSex, trainSurvived);
    vector< vector<double> > weightsAge = likelihoodContinuous(trainAge, trainSurvived);
    vector< double > aprioriS = apriori(trainSurvived);
   
    std::cout << "=================================================" << endl;
    std::cout << endl;
    std::cout << "Likelihood for p(pclass|survived)" << endl;
    for (int i = 0; i < 2; i++) {
        std::cout << weightsPclass[i][0] << " " << weightsPclass[i][1] << " " << weightsPclass[i][2] << endl;
    }
    std::cout << endl;
    std::cout << "Likelihood for p(sex|survived)" << endl;
    for (int i = 0; i < 2; i++) {
        std::cout << weightsSex[i][0] << " " << weightsSex[i][1] << endl;
    }
    std::cout << endl;
    std::cout << "Mean" << endl;
    std::cout << weightsAge[0][0] << " " << weightsAge[0][1] << endl;
    std::cout << endl;

    std::cout << "Variance" << endl;
    std::cout << weightsAge[1][0] << " " << weightsAge[1][1] << endl;
    std::cout << endl;

    std::cout << "Apirori" << endl;
    std::cout << aprioriS[0] << " " << aprioriS[1] << endl;

    std::cout << endl;
    std::cout << "=================================================" << endl;

   

    vector< vector<double> > testProbs;
    // this is a nx2 vector. each row is an instance, column 1 is dead, 2 is survived.

    for(int i = 0; i < testSex.size(); i++){
        double sexi = testSex[i];
        double agei = testAge[i];
        double pclassi = testPclass[i];
        testProbs.push_back(naiveBayes(pclassi, sexi, agei, aprioriS, weightsPclass, weightsSex, weightsAge));
        //cout << "age " << agei << " pclass " << pclassi << endl;
    }

    //end/stop algorithm time
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    /*
    for (auto i : weightsSex) {
        for (auto j : i) {
            std::cout << j;
        }
    }
    std::cout << endl;
    for (int i = 0; i < testProbs.size()-1; i++) {
        
        std::cout << testProbs[i][0] << ", " << testProbs[i][1] << ", " << testSurvived[i] << endl;
    }
    */
    std::cout << accuracy(testSurvived, testProbs) << endl;
    
    //test 
    double acc = 0;
    vector<double> corr;

    for (int i = 0; i < testSurvived.size(); i++) {
        double fir = testProbs[i][0];
        double sec = testProbs[i][1];

        double max = fir;
        if (fir > sec) {
            max = fir;
        }
        else {
            max = sec;
        }
        std::cout << max << ", " << testSurvived[i] << endl;
        if (max < 0.5) {
            if (testSurvived[i] == 1) {
                acc++;
            }
        }
        else {
            if (testSurvived[i] == 0) {
                acc++;
            }
        }
    }
>>>>>>> Stashed changes

};