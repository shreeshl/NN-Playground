#include "vec_operations.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <random>
#include <cstdlib>
#include <chrono>
#include <cassert>
#include <ctime>

using namespace std;

typedef vector<vector<double> > mat2d;

struct Data{
    mat2d x;
    mat2d y;
};

class TrainingData{
    public:
        Data data;
        void load_data(string filename, int no_classes);
        Data getBatch(int batchsize);
};

Data TrainingData::getBatch(int batchsize){
    if (batchsize==data.x.size()) return data;
    mat2d x;
    mat2d y;
    Data batch;
    while(batchsize!=0){
        int index = rand() % data.x.size();
        x.push_back(data.x[index]);
        y.push_back(data.y[index]);
        batchsize--;
    }
    batch.x = x;
    batch.y = y;
    return batch;
}

void TrainingData::load_data(string filename, int no_classes){
    ifstream infile(filename);
    int count = 0;
    while (infile){
        string s;
        if (!getline(infile, s)) break;

        istringstream ss(s);
        vector<double> ytemp;
        vector<double> xtemp;
        bool first = true;

        while (ss){
            string d;
            if (!getline( ss, d, ',' )) break;
            if (first) ytemp.push_back(stod(d));
            else xtemp.push_back( stod(d) );
            first = false;
        }
        data.y.push_back(ytemp);
        data.x.push_back(xtemp);
        count++;
        if(count%10000==0) cout << count << " lines read!" << endl;
    }
    if (!infile.eof()){
        cerr << "Fooey!\n";
    }
}

class Net{
    public:
        vector<int> nodes = {16,8,4};
        int num_layers;
        double lr = 0.3;
        double momentum = 0.9;
        unsigned iterations = 1000;
        double weight_decay = 0.04;
        double threshold = 0.95;
        string activation = "relu";
        bool verbose = false;
        unordered_map<int, mat2d > weights;
        unordered_map<int, mat2d > prev_grad;
        unordered_map<int, mat2d > derivatives;
        unordered_map<int, mat2d > outputs;

        void init_params(int N, int D, int M);
        void forward(mat2d &x);
        void linearForward(mat2d &layer_input, mat2d &param, string activation, int layer_id);
        void sigmoid(const mat2d &x, int layer_id);
        void relu(const mat2d &x, int layer_id);
        void relu2(const mat2d &x, const mat2d &w, const mat2d &b, int layer_id);
        void backward(mat2d &y);
        void linearBackward(mat2d &received_msg, int layer_id, bool is_output_layer);
        
        void updateParam();
        
        mat2d predictions(mat2d &x);
        double calc_loss(mat2d &y);
        double calc_error(mat2d &ytrue);

};

void Net::sigmoid(const mat2d &x, int layer_id) {
    int m = x.size();
    int n = x[0].size();
    mat2d result(m, vector<double>(n));
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            result[i][j] = 1/(1+exp(-1*x[i][j]));
        }
    }
    outputs[layer_id] = result;
    return;
}

void Net::relu(const mat2d &x, int layer_id) {
    int m = x.size();
    int n = x[0].size();
    mat2d result = x;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if (x[i][j]<=0) result[i][j] = 0;
        }
    }
    outputs[layer_id] = result;
    return;
}

void Net::relu2(const mat2d &x, const mat2d &w, const mat2d &b, int layer_id) {
    int m = x.size();
    int n = w[0].size();
    mat2d result(m,vector<double>(n));
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            for(int k=0;k<x[0].size();k++){
                result[i][j] += w[k][j]*x[i][k] + b[0][j];
            }
            if (result[i][j]<=0) result[i][j] = 0;
        }
    }
    outputs[layer_id] = result;
    return;
}

mat2d normalInit(int m, int n){
    // static unsigned seed = 0;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);
    normal_distribution<double> distribution (0.0,1.0);
    mat2d output(m, vector<double>(n));
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            output[i][j] = .1*distribution(generator);
        }
    }
    // seed++;
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
    
    num_layers = nodes.size();
    cout << "Topology:" << endl;
    cout << "Input: " << N << " x " <<  D << endl;
    for(int i=1;i<num_layers;i++){
        mat2d zeros(nodes[i-1]+1, vector<double>(nodes[i]));
        weights[i]    = normalInit(nodes[i-1]+1, nodes[i]);
        derivatives[i]= zeros;
        prev_grad[i]  = zeros;
        cout << "Weight" << i << ": " << nodes[i-1] << " x " << nodes[i] << endl;
    }
    return;
}

void Net::linearForward(mat2d &layer_input, mat2d &param, string activation, int layer_id){
    clock_t start;
    int D = param.size() - 1;
    start=clock();
    mat2d bias = sliceMatrix(param, D, 0, D+1, 0);
    // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
    start=clock();
    mat2d weight_sans_bias = sliceMatrix(param,0,0,D,0);
    // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
    start=clock();
    if (activation=="relu" && param[0].size()!=1) relu(matadd(matmul(layer_input, weight_sans_bias), bias), layer_id);
    // if (activation=="relu" && param[0].size()!=1) relu2(layer_input, weight_sans_bias, bias, layer_id);
    else sigmoid(matadd(matmul(layer_input, weight_sans_bias), bias),layer_id);
    // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
}

void Net::forward(mat2d &x){
    outputs[0] = x;
    for(int i=1;i<num_layers;i++) linearForward(outputs[i-1], weights[i], activation, i);
    return;
}

void Net::linearBackward(mat2d &received_msg, int layer_id, bool is_output_layer){
    
    mat2d layer_output = outputs[layer_id];
    double N = outputs[layer_id-1].size();
    double Di = outputs[layer_id-1][0].size();
    double Do = layer_output[0].size();
    if(!is_output_layer){
        if(activation=="relu"){
            for(int i=0;i<received_msg.size();i++){
                for(int j=0;j<received_msg[0].size();j++){
                    if(layer_output[i][j]<=0) received_msg[i][j] = 0;
                }
            }
        }
        else{
            received_msg = elementwisemult(received_msg, elementwisemult(layer_output, 1 - layer_output));
        }
    }

    mat2d d1 = (1/N)*matmul(outputs[layer_id-1], received_msg, true, false);
    vector<double> d2 = (1/N)*matsum(received_msg, 0);

    for(int i=0;i<Di+1;i++){
        for(int j=0;j<Do;j++){
            if(i<Di) derivatives[layer_id][i][j] = d1[i][j];
            else derivatives[layer_id][i][j] = d2[j];
        }
    }    
    mat2d param_sans_bias = sliceMatrix(weights[layer_id],0,0,Di,Do);
    received_msg = matmul(received_msg, param_sans_bias, false, true);
    return;
    
}

void Net::backward(mat2d &y){
    bool is_output_layer = true;
    mat2d send_msg = matadd(outputs[num_layers-1],-1*y);
    for(int i=num_layers-1;i>0;i--){
        linearBackward(send_msg, i, is_output_layer);
        is_output_layer = false;
    }
    return;
}

double Net::calc_loss(mat2d &y){
    mat2d ypred = outputs[num_layers-1];
    
    double cls_loss = 0;
    mat2d total_loss = elementwisemult(y,log2d(ypred)) + elementwisemult(1-y,log2d(1-ypred));
    
    for(int i=0;i<total_loss.size();i++){
        cls_loss = cls_loss - total_loss[i][0];
    }
    cls_loss /= total_loss.size();
    double reg_loss = 0;
    // for (int i=1;i<num_layers;i++){
    //     vector<vector<double> > param = sliceMatrix(weight[layer_id],0,0,num_nodes[layer_id-1]-1,0);
    //     reg_loss = reg_loss + weight_decay*matsum(param)
    // }
    return cls_loss + reg_loss;
    
}

void Net::updateParam(){
    for(int i=num_layers-1;i>0;i--){
        prev_grad[i] = momentum*prev_grad[i] + (-1*lr*derivatives[i]);
        weights[i] = weights[i] + prev_grad[i];
    }
    return;
}

double Net::calc_error(mat2d &ytrue){
    mat2d ypred = outputs[num_layers-1];
    double error = 0;
    for(int i=0;i<ypred.size();i++){
        if (ypred[i][0]>=threshold) ypred[i][0] = 1;
        else ypred[i][0] = 0;
        if(ypred[i][0]!=ytrue[i][0]) error++;
    }
    return error/ypred.size();
}

mat2d Net::predictions(mat2d &x){
    forward(x);
    mat2d ypred = outputs[num_layers-1];
    for(int i=0;i<ypred.size();i++){
        if (ypred[i][0]>=threshold) ypred[i][0] = 1;
        else ypred[i][0] = 0;
    }
    return ypred;
}


int main(){
    
    // mat2d X = {{1,1},
    //            {1,0},
    //            {0,0},
    //            {0,1}};
    
    // mat2d Y = {{0},
    //            {1},
    //            {0},
    //            {1}};

    TrainingData train_data, test_data;
    
    train_data.load_data("train.csv",2);
    cout << "Training data read!" << endl;
    test_data.load_data("test.csv",2);
    cout << "Test data read!" << endl;
    Net net;
    
    int N = train_data.data.x.size();
    int D = train_data.data.x[0].size();
    int M = train_data.data.y[0].size();
    int batchsize = N;
    net.init_params(N,D,M);
    cout << "Neural Network Built!" << endl;
    int iters = 0;
    clock_t start;
    double duration;

    while(iters<net.iterations){      
        
        Data batch = train_data.getBatch(batchsize);
        start = clock();
        net.forward(batch.x);
        // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
        start = clock();
        net.backward(batch.y);
        // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
        start = clock();
        net.updateParam();
        // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
        cout << "Iteration: " << iters << "| Batch Loss: " << net.calc_loss(batch.y) << " | Batch Error: " << net.calc_error(batch.y) << endl;
        
        iters++;
    }
}