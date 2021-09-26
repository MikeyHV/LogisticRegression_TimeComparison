#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <math.h>
using namespace std;

vector<double> updateWeights(vector<double> features,vector<int> labels,vector<double> weights,int lr){

}

double costFunction(vector<double> features, vector<int>labels, vector<double>weights){

}

double sigmoid(int z){
    return 1.0/(1 + exp(-z));
}

vector<double> train(vector<double> features,vector<int> labels,vector<double> weights,int lr, int iters){
    vector<double> hist;

    for(int i = 0; i < iters; i++){
        weights = updateWeights(features, labels, weights, lr);

        double cost = costFunction(features, labels, weights);
        hist.push_back(cost);
    }
    return weights;
}

int main(int argc, char **argv)
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
    
    if (!inFS.is_open()){
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
    
}
