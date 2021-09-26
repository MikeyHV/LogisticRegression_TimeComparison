#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <math.h>
#include <numeric>
using namespace std;

vector< vector<double> > sigmoid(vector< vector<double> > z){
    /**
     * takes in a nx1 vector
     * applies the sigmoid function to all indexes
     * returns a nx1
     * */
    for(int i = 0; i < z[0].size(); i++){
        z[0][i] = 1.0/(1 + exp(-z[0][i]));
    }
    return z;
}

std::vector<std::vector<double> > transpose1dto2d(vector<double> orig){
    //placeholder
    std::vector<std::vector<double> > hi;
        if (orig.size() == 0) {
        return std::vector<std::vector<double> >(); //empty
    }

    std::vector<std::vector<double> > transfVec(orig.size(), std::vector<double>());

    for (int i = 0; i < orig.size(); i++)
    {
        transfVec[0][i] = orig[i];
    }

    return transfVec;
}


vector<double>  transpose2dto1d(vector< vector<double> > orig){
    /**
     * transpose a mx1 vector
     * to a 1xm vecotr
     **/
    std::vector<std::vector<double> > hi;
        if (orig.size() == 0) {
        return std::vector<double>(); //empty
    }

    std::vector<double> transfVec(orig.size());

    for (int i = 0; i < orig.size(); i++)
    {
        transfVec[i] = orig[0][i];
    }

    return transfVec;
}

/*
* returns the transpose of the vector passed in as an argument
*/
std::vector<std::vector<double> > transpose2dto2d(std::vector<std::vector<double> > orig)
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

vector<double> matVecMult(vector<double> one,vector< vector<double> > two){
    /**
     * takes 1xm a vector one
     * and a mx1 vector two
     * and returns 1x1 vector
     **/
    vector<double> ret(one.size());
    vector<double> three = transpose2dto1d(two);
    for(int i = 0; i < one.size(); i++){
        ret[i] = one[i] * three[i];
    }
    return ret;
}

vector< vector<double> > dotProduct(vector< vector<double> > one,vector< vector<double> > two){
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


std::vector<std::vector<double> > predict(vector< vector<double> > features,vector< vector<double> >  weights){
    /**
     * features is an array of size nx3
     * weights is an array of size 3x1
     * must return an array of size nx1
     * 
     * int retMe = inner_product(features.begin(), features.begin(), weights.begin(), 0);
     * retMe = sigmoid(retMe);
     **/

    vector< vector<double> > pls = dotProduct(features, weights);
    return sigmoid(pls);
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
                                vector< vector<double> > labels, 
                                std::vector< std::vector<double> > weights, 
                                double lr) {
    int n = features.size();

    //make predictions
    vector< vector<double> > predictions = predict(features, weights);
    
    std::vector<std::vector<double> > predMinLabel = costDiff(predictions, labels);

    vector< vector<double> > featFeed = transpose2dto2d(featFeed);

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

vector< vector<double> >  classCost1(vector< vector<double> > preds,vector< vector<double> > labels){
    for(int i = 0; i < preds.size(); i++){
        preds[0][i] = log(preds[0][i]) * labels[0][i] * -1;
    }
    return preds;
}

vector< vector<double> >  classCost2(vector< vector<double> > preds,vector< vector<double> > labels){
    for(int i = 0; i < preds.size(); i++){
        preds[0][i] = log(1 - preds[0][i]) * (1 - labels[0][i]);
    }
    return preds;
}

double costFunction(vector< vector<double> > features, vector< vector<double> > labels,vector< vector<double> > weights){
    std::vector<std::vector<double> >  preds = predict(features, weights);
    
    std::vector<std::vector<double> > cost1 = classCost1(preds, labels);

    std::vector<std::vector<double> >  cost2 = classCost2(preds, labels);
    
    std::vector<std::vector<double> > finCost = costDiff(cost1, cost2);

    double cost = sumVec(finCost)/labels.size();

    return cost;
}

std::vector<std::vector<double> > costDiff(std::vector<std::vector<double> > one, std::vector<std::vector<double> > two){
    /**
     * one is predictions. a size of nx1
     * two is labels. a size of nx1
     * return a nx1
     * */
    for(int i = 0; i < one.size(); i++){
        one[0][i] = one[0][i] - two[0][i];
    }
    return(one);
}

double sumVec(std::vector<std::vector<double> > one){
    /*
    takes in an nx1
    returns a scalar
    */
    double sum = 0;
    for(int i = 0; i < one.size(); i++){
        one[0][i] = one[0][i] + sum;
    }
    return(sum);
}

vector< vector<double> >  train(vector< vector<double> > features,vector< vector<double> > labels,vector< vector<double> >  weights,int lr, int iters){
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
