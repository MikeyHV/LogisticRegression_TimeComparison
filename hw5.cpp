#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <math.h>
#include <numeric>
using namespace std;

double sigmoid(int z){
    return 1.0/(1 + exp(-z));
}

vector<double> predict(vector< vector<double> > features,vector< vector<double> >  weights){
    int retMe = inner_product(features.begin(), features.begin(), weights.begin(), 0);
    retMe = sigmoid(retMe);
    vector<double> pls;
    return pls;
}

vector<double> updateWeights(vector< vector<double> > features,vector<double> labels,vector< vector<double> > weights,int lr){
    int obs = labels.size();

    vector<double> preds = predict(features, weights);

    vector<double> class1Cost = labels * log(preds);

}

vector<double> classCost1(vector<double> preds, vector<double> labels){
    for(int i = 0; i < preds.size(); i++){
        preds[i] = log(preds[i]) * labels[i] * -1;
    }
    return preds;
}

vector<double> classCost2(vector<double> preds, vector<double> labels){
    for(int i = 0; i < preds.size(); i++){
        preds[i] = log(1 - preds[i]) * (1 - labels[i]);
    }
    return preds;
}

double costFunction(vector< vector<double> > features, vector<double>labels,vector< vector<double> > weights){
    vector<double>  preds = predict(features, weights);
    
    vector<double> cost1 = classCost1(preds, labels);

    vector<double>  cost2 = classCost2(preds, labels);
    
    vector<double> finCost = costDiff(cost1, cost2);

    double cost = sumVec(finCost)/labels.size();

    return cost;
}

vector<double> costDiff(vector<double> one, vector<double> two){
    for(int i = 0; i < one.size(); i++){
        one[i] = one[i] - two[i];
    }
    return(one);
}

double sumVec(vector<double> one){
    double sum = 0;
    for(int i = 0; i < one.size(); i++){
        one[i] = one[i] + sum;
    }
    return(sum);
}

vector<double> train(vector<double> features,vector<double> labels,vector<double> weights,int lr, int iters){
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
