#include <iostream>
#include <vector>
#include "vec_operations.h"
#include <string>
#include <random>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>

using namespace std;

class Net{
    public:
        vector<int> nodes = {3};
        int num_layers;
        unsigned iterations = 1000;
        double lr = 0.001;
        double momentum = 0.05;
        double weight_decay = 0.04;
        double threshold = 0.95;
        string activation = "relu";
        bool verbose = false;
        unordered_map<int, vector<vector<double> > > weights;
        unordered_map<int, vector<vector<double> > > prev_grad;
        unordered_map<int, vector<vector<double> > > derivatives;
        unordered_map<int, vector<vector<double> > > outputs;

        void init_params(int N, int D, int M);
        vector<vector<double> > sigmoid(vector<vector<double> > &x);
        vector<vector<double> > relu(vector<vector<double> > &x);
        void forward(vector<vector<double> > &x);
        vector<vector<double> > linearForward(vector<vector<double> > &layer_input, vector<vector<double> > &param, string activation);
        
        void backward(vector<vector<double> > &x, vector<vector<double> > &y);
        vector<vector<double> > linearBackward(vector<vector<double> > &received_msg, int layer_id, bool is_output_layer);
        void updateParam() {};
        double calc_loss(vector<double> ybatch) {};
        double calc_error(vector<double> ybatch) {};
        double predictions(vector<double> ybatch) {};
};

vector<vector<double>> normalInit(int m, int n){
    unsigned seed = 0;
    default_random_engine generator (seed);
    normal_distribution<double> distribution (0.0,1.0);
    vector<vector<double>> output(m, vector<double>(n));
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            output[i][j] = .1*distribution(generator);
        }
    }
    return output;
}

void Net::init_params(int N, int D, int M){
    vector<int> num_nodes;
    num_nodes.push_back(D);
    for(int i=0;i<nodes.size();i++){
        num_nodes.push_back(nodes[i]);
    }
    
    num_nodes.push_back(M);
    
    nodes = num_nodes;
    if(verbose){
        cout << "yes" <<endl;
    }
    
    num_layers = nodes.size();
    for(int i=1;i<num_layers;i++){
        weights[i]    = normalInit(nodes[i-1]+1, nodes[i]);
        derivatives[i]    = normalInit(nodes[i-1]+1, nodes[i]);
        prev_grad[i]  = normalInit(nodes[i-1]+1, nodes[i]);
    }
    return;
}

vector<vector<double> > Net::sigmoid(vector<vector<double> > &x) {
    int m = x.size();
    int n = x[0].size();
    vector<vector<double>> result(m, vector<double>(n));
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            result[i][j] = exp(x[i][j]);
        }
    }
    return result;
}

vector<vector<double> > Net::relu(vector<vector<double> > &x) {
    int m = x.size();
    int n = x[0].size();
    vector<vector<double>> result = x;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if (x[i][j]<=0) result[i][j] = 0;
        }
    }
    return result;
}

vector<vector<double> > Net::linearForward(vector<vector<double> > &layer_input, vector<vector<double> > &param, string activation){
    int D = param.size() - 1;
    vector<vector<double> > layer_output;
    vector<vector<double> > bias = sliceMatrix(param, D, 0, D+1, 0);
    vector<vector<double> > temp = matadd(matmul(layer_input, sliceMatrix(param,0,0,D,0)), bias);
    if (activation=="relu"){        
        layer_output = relu(temp);
    }
    else{
        layer_output = sigmoid(temp);
    }
    return layer_output;
}

void Net::forward(vector<vector<double> > &x){
    outputs[0] = x;
    for(int i=1;i<num_layers;i++){
        outputs[i] = linearForward(outputs[i-1], weights[i], activation);
    }
    return;
}

vector<vector<double> > Net::linearBackward(vector<vector<double> > &received_msg, int layer_id, bool is_output_layer){
    vector<vector<double> > layer_param  = weights[layer_id];
    vector<vector<double> > layer_output = outputs[layer_id];
    vector<vector<double> > layer_input  = outputs[layer_id-1];
    int N = layer_input.size();
    int Di = layer_input[0].size();
    int Do = layer_output[0].size();
    if(is_output_layer){
        if(activation=="relu"){
            for(int i=0;i<received_msg.size();i++){
                for(int j=0;j<received_msg[0].size();j++){
                    if(layer_output[i][j]<=0) received_msg[i][j] = 0;
                }
            }
        }
        else{
            received_msg = elementwisemult(received_msg, elementwisemult(layer_output, 1 - (-1*layer_output)));
        }
    }
    // printMatrix(received_msg);
    cout << layer_input.size() << " " << layer_input[0].size() << " " << received_msg.size() << " " << received_msg[0].size() << endl;
    vector<vector<double> > d1 = (1/N)*matmul(layer_input, received_msg, true, false);
    // printMatrix(d1);
    vector<double> d2 = (1/N)*matsum(received_msg, 0);
    cout << d2.size() << endl;
    // printVector(d2);
    for(int i=0;i<Di+1;i++){
        for(int j=0;j<Do;j++){
            if(i<Di) derivatives[layer_id][i][j] = d1[i][j];
            else derivatives[layer_id][i][j] = d2[j];
        }
    }    
    printMatrix(derivatives[layer_id]);

    vector<vector<double> > sent_msg = matmul(received_msg, sliceMatrix(layer_param,0,0,Di,Do), false, true);
    return sent_msg;
    
}

void Net::backward(vector<vector<double> > &x, vector<vector<double> > &y){
    bool is_output_layer = true;
    vector<vector<double> > send_msg = matadd(outputs[num_layers-1],-1*y);
    for(int i=num_layers-1;i>=0;i--){
        outputs[i] = linearBackward(send_msg, i, is_output_layer);
        printMatrix(outputs[i]);
        is_output_layer = false;
    }
    return;
}


int main(){
    Net net;
    vector<vector<double> > X = {{1,2,3},
                                {4,5,6}};
    
    vector<vector<double> > Y = {{1},
    {4}};
    
    int N = X.size();
    int D = X[0].size();
    int M = Y[0].size();
    
    net.init_params(2,3,2);
    net.forward(X);
    net.backward(X,Y);
    
    // printMatrix(net.outputs[0]);
    // printMatrix(net.weights[1]);
    // printMatrix(net.outputs[1]);
    // printMatrix(net.weights[2]);
    // printMatrix(net.outputs[2]);

    vector<vector<double> > a = {{1,2,3},{3,4,7},{7,8,9}};
    vector<vector<double> > b = {{1,3},{1,3}};
    vector<vector<double> > c = {{1,3,4}};
    // vector<vector<double> > d = indexVector(a,1,0,1+1,0);
    // printMatrix(d);
    // {"row1_s","col1_s","row1_e","col1_e","row2_s","col2_s","row2_e","row2_e"}
    // vector<vector<double> > c = matmul(a,b, vector<int> {1,1,0,0,0,0,0,0});
    // printMatrix(a);
    // printMatrix(b);
    // printMatrix(c);

    // printMatrix(net.layers[1]);
    // vector<double> a;
    // a.push_back(3);
    // a.push_back(2);
    // a.push_back(1);

    // vector<double> b;
    // b.push_back(1);
    // b.push_back(7);
    // b.push_back(14);

    // vector<double> c = a + b;
    // for(double i=0;i<c.size();i++){
    //     cout << c[i] << endl;
    // }
}