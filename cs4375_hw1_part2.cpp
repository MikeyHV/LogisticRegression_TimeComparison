#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <string>

/*
* CS 4375 Homework 1 Part 2
* Name: Austin Li  
* Date: 9/5/2021
* File IO Based on code from Dr. Mazidi
*/

//function prototypes
double vectorSum(const std::vector<double>& inputVector);
double vectorMean(const std::vector<double>& inputVector);
double vectorMedian(const std::vector<double>& inputVector);
std::string vectorRange(const std::vector<double>& inputVector);
double vectorCorrelation(const std::vector<double>& inputVectorX, const std::vector<double>& inputVectorY);
double vectorVariance(const std::vector<double>& inputVector);
double vectorCovariance(const std::vector<double>& inputVectorX, const std::vector<double>& inputVectorY);

//returns summation of parameter inputVector 
double vectorSum(const std::vector<double>& inputVector) {
    double sum = 0;
    for (double curr: inputVector) {
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

//returns median of parameter inputVector
double vectorMedian(const std::vector<double>& inputVector) {
    double medianValue = 0;
    int numElements = inputVector.size();
    std::vector<double> sortedInputVector = inputVector; //copy of inputVector
    std::sort(sortedInputVector.begin(), sortedInputVector.end());;


    //if num elements in vector is even
    if (numElements % 2 == 0) {
        int leftIndex = numElements / 2 - 1;
        int rightIndex = numElements / 2;
        double leftOfMiddle = sortedInputVector[leftIndex]; //x[n/2] - 1
        double rightOfMiddle = sortedInputVector[rightIndex];  //x[n/2]
        medianValue = (leftOfMiddle + rightOfMiddle) / 2;
    }
    else {
        //if num elements in vector is odd
        medianValue = sortedInputVector[numElements / 2];
    }
    return medianValue;
}

//returns range of parameter inputVector
std::string vectorRange(const std::vector<double>& inputVector) {
    double min = inputVector[0];
    double max = inputVector[0];

    //finding min and max
    for (double curr : inputVector) {
        if (curr < min) {
            min = curr;
        }
        if (curr > max) {
            max = curr;
        }
    }
    std::string range = std::to_string(min) +  " " + std::to_string(max);
    return range; 
}

//retruns correlation for vector parameters inputVectorX and inputVectorY
double vectorCorrelation(const std::vector<double>& inputVectorX, const std::vector<double>& inputVectorY) {
    double covariance = vectorCovariance(inputVectorX, inputVectorY);

    //standard deviation is square root of variance
    double stdDevX = sqrt(vectorVariance(inputVectorX));
    double stdDevy = sqrt(vectorVariance(inputVectorY));

    double correlation = covariance / (stdDevX * stdDevy);
    return correlation;
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

//returns covariance of parameters inputVectorX and inputVectorY
double vectorCovariance(const std::vector<double>& inputVectorX, const std::vector<double>& inputVectorY) {
    double summation = 0;
    int n = inputVectorX.size();
    double xAvg = 0;
    double yAvg = 0;
    double xSum = 0;
    double ySum = 0;

    //compute x average and y average
    for (int i = 0; i < n; i++) {
        xSum += inputVectorX[i];
        ySum += inputVectorY[i];
    }
    xAvg = xSum / n;
    yAvg = ySum / n;

    //calculate covariance
    for (int i = 0; i < n; i++) {
        summation += (inputVectorX[i] - xAvg) * (inputVectorY[i] - yAvg);
    }
    double covariance = summation / (n - 1);
    return covariance;
}




using namespace std;
int main(int argc, char** argv)
{
    ifstream inFS; // Input file stream
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);
    // Try to open file
    cout << "Opening file Boston.csv." << endl;

    inFS.open("Boston.csv");

    if (!inFS.is_open()) {
        cout << "Could not open file Boston.csv." << endl;
        return 1; // 1 indicates error
    }
    // Can now use inFS stream like cin stream
    // Boston.csv should contain two doubles
    cout << "Reading line 1" << endl;
    getline(inFS, line);

    // echo heading
    cout << "heading: " << line << endl;

    int numObservations = 0;

    while (inFS.good())
    {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);


    cout << "Closing file Boston.csv." << endl;
    inFS.close(); // Done with file, so close it

    //Call calculation functions
    cout << "Number of records: " << numObservations << endl;

    std::cout << "===============================" << std::endl;
    std::cout << std::endl;

    //sum
    std::cout << "sum of rm is: " << vectorSum(rm) << std::endl;
    std::cout << "sum of medv is: " << vectorSum(medv) << std::endl;
    std::cout << std::endl;

    //mean
    std::cout << "mean of rm is: " << vectorMean(rm) << std::endl;
    std::cout << "mean of medv is: " << vectorMean(medv) << std::endl;
    std::cout << std::endl;

    //medain 
    std::cout << "median of rm is: " << vectorMedian(rm) << std::endl;
    std::cout << "median of medv is: " << vectorMedian(medv) << std::endl;
    std::cout << std::endl;

    //range 
    std::cout << "range of rm is: " << vectorRange(rm) << std::endl;
    std::cout << "range of medv is: " << vectorRange(medv) << std::endl;
    std::cout << std::endl;

    //covariance
    std::cout << "covariance between rm and medv is: " << vectorCovariance(rm, medv) << std::endl;
    std::cout << std::endl;

    //correlation
    std::cout << "correlation between rm and medv is: " << vectorCorrelation(rm, medv) << std::endl;
    std::cout << std::endl;
}