#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
using namespace std;


int count_survived(vector<double> data){
    int survived = 0;
    for(int i = 0; i < data.size(); i++){
        if(data[0] == 0){
            survived++;
        }
    }
    return(survived);
}

double likelihoods(vector< vector<double> > data){
    double pc = 0;
    double sx = 0;
    double age = 0;
}