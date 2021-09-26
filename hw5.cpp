#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
using namespace std;

std::vector<X> train(features, labels, weights, lr, iters){
    

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
    
    while (inFS.good())
    {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');
        
        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);
        
        numObservations++;
    }

    cout << "Closing file Boston.csv." << endl;
    inFS.close(); // Done with file, so close it
    
    rm.resize(numObservations);
    medv.resize(numObservations);
}
