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

vector<double> transpose1d(vector<double> one){
    //placeholder
    vector<double> hi;
    return hi;
}

vector< vector<double> > transpose2d(vector< vector<double> > one){
    vector< vector<double> > hi;
    return hi;
}

vector<double>  transpose2dto1d(vector< vector<double> > one){
    vector<double>  hi;
    return hi;
}

vector< vector<double> > transpose1dto2d(vector<double> one){
    vector< vector<double> >  hi;
    return hi;
}

vector<double> matVecMult(vector<double> one, vector<double> two){
    vector<double> ret(one.size());
    vector<double> three = transpose1d(two);
    for(int i = 0; i < one.size(); i++){
        ret[i] = one[i] * three[i];
    }
    return ret;
}

vector< vector<double> > dotProduct(vector< vector<double> > one, vector<double> two){
    /**
     * takes in a nxm vector one
     * and a mx1 vector two
     * returns a nx1 vector ret
    **/
    //vector<vector<int>> vec( n , vector<int> (m, 0));
    vector< vector<double> > ret(one.size());
    for(int i = 0; i < one.size(); i++){
        ret[i] = matVecMult(one[i], two);
    }
    return ret;
}

vector<double> vecDivScalar(vector<double> one, double scal){
    for(int i = 0; i < one.size(); i++){
        one[i] = one[i]/scal;
    }
    return one;
}

vector<double> vecMultScalar(vector<double> one, double scal){
    for(int i = 0; i < one.size(); i++){
        one[i] = one[i]*scal;
    }
    return one;
}

vector<double> vecSub(vector<double> one, vector<double> two){
    for(int i = 0; i < one.size(); i++){
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
std::vector< std::vector<double> > updateWeights(std::vector< std::vector<double> > features, 
                                std::vector<double> labels, 
                                std::vector< std::vector<double> > weights, 
                                double lr) {
    int n = features.size();

    //make predictions
    vector<double> predictions = predict(features, weights);
    
    vector<double> predMinLabel = costDiff(predictions, labels);

    vector< vector<double> > featFeed = transpose2d(featFeed);

    vector< vector<double> > gradient1 = dotProduct(featFeed, predMinLabel);
    vector<double> gradient2 = transpose2dto1d(gradient1);
    gradient2 = vecDivScalar(gradient2, n);
    gradient2 = vecMultScalar(gradient2, lr);
    //gradient3 = transpose1dto2d(gradient2);
    vector<double> weightsTemp = transpose2dto1d(weights);
    weightsTemp = vecSub(weightsTemp, gradient2);
    weights = transpose1dto2d(weightsTemp);

    return weights;
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
