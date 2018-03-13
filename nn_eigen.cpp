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
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

struct Data{
    MatrixXd x;
    MatrixXd y;
    Data() {};
    Data(int data_size, int data_dimension){
        x = MatrixXd::Zero(data_size, data_dimension);
        y = MatrixXd::Zero(data_size, 1);
    };
};

double sigmoid_unary(double x){
    return 1/(1+exp(-x));
}
double relu_unary(double x){
    if(x<=0) return 0;
    else return x;
}
double log_unary(double x){
    return log(x);
}

class TrainingData{
    public:
        Data data;
        int data_size = 0;
        int data_dimension = 0;
        void load_data(string filename);
        void get_data_size(string filename);
        Data getBatch(int batchsize);
};

Data TrainingData::getBatch(int batchsize){
    if (batchsize==data_size) return data;
    MatrixXd x;
    MatrixXd y;
    Data batch(batchsize, data_dimension);
    while(batchsize!=0){
        int index = rand() % data_size;
        batch.x.row(batchsize) = data.x.row(index);
        batch.y.row(batchsize) = data.y.row(index);
        batchsize--;
    }
    return batch;
}

void TrainingData::get_data_size(string filename){
    ifstream infile(filename);
    while (infile){
        
        string s;
        if (!getline(infile, s)) break;
        
        if (data_dimension==0){
            for(int i = 0;i<s.size();i++){
                if(s[i]==',') data_dimension++;
            }
        }
        data_size++;
    }
    if (!infile.eof()){
        cerr << "Fooey!\n";
    }
}

void TrainingData::load_data(string filename){
    TrainingData::get_data_size(filename);
    cout << "Data Size: " << data_size << " Data Dimensions: " << data_dimension << endl;
    data.x = MatrixXd::Zero(data_size, data_dimension);
    data.y = MatrixXd::Zero(data_size, 1);
    ifstream infile(filename);
    int row = 0;
    while (infile){
        string s;
        if (!getline(infile, s)) break;

        istringstream ss(s);
        int col = 0;

        while (ss){
            string d;
            if (!getline( ss, d, ',' )) break;
            if(col==0) data.y(row,col) = stod(d);
            else data.x(row,col-1) = stod(d);
            col++;
        }

        row++;
        // if(row%10000) cout << row << " lines read!" << endl;
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
        unordered_map<int, MatrixXd > weights;
        unordered_map<int, MatrixXd > prev_grad;
        unordered_map<int, MatrixXd > derivatives;
        unordered_map<int, MatrixXd > outputs;
        
        void sigmoid(const MatrixXd &x, int layer_id);
        void relu(const MatrixXd &x, int layer_id);
        void init_params(int N, int D, int M);

        void forward(MatrixXd &x);
        void linearForward(MatrixXd &layer_input, MatrixXd &param, string activation, int layer_id);
        void backward(MatrixXd &y);
        void linearBackward(MatrixXd &received_msg, int layer_id, bool is_output_layer);
        void updateParam();
        
        double calc_loss(MatrixXd &y);
        double calc_error(MatrixXd &ytrue);
        MatrixXd predictions(MatrixXd &x);

        void relu_unary(double x) {};
};

void Net::sigmoid(const MatrixXd &x, int layer_id) {
    int m = x.rows();
    int n = x.cols();
    MatrixXd result(m, n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            result(i,j) = 1/(1+exp(-1*x(i,j)));
        }
    }
    outputs[layer_id] = result;
    return;
}

void Net::relu(const MatrixXd &x, int layer_id) {
    int m = x.rows();
    int n = x.cols();
    MatrixXd result = x;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if (x(i,j)<=0) result(i,j) = 0;
        }
    }
    outputs[layer_id] = result;
    return;
}

MatrixXd normalInit(int m, int n){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);
    normal_distribution<double> distribution (0.0,1.0);
    MatrixXd output(m, n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            output(i,j) = .1*distribution(generator);
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
    
    num_layers = nodes.size();
    cout << "Topology:" << endl;
    cout << "Input: " << N << " x " <<  D << endl;
    for(int i=1;i<num_layers;i++){
        weights[i]    = normalInit(nodes[i-1]+1, nodes[i]);
        derivatives[i]= MatrixXd::Zero(nodes[i-1]+1, nodes[i]);
        prev_grad[i]  = MatrixXd::Zero(nodes[i-1]+1, nodes[i]);
        cout << "Weight" << i << ": " << nodes[i-1]+1 << " x " << nodes[i] << endl;
    }
    return;
}

void Net::linearForward(MatrixXd &layer_input, MatrixXd &param, string activation, int layer_id){
    // clock_t start;
    int D = param.rows() - 1;
    // start=clock();
    VectorXd bias = param.row(D);
    // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
    // start=clock();
    MatrixXd weight_sans_bias = param.block(0,0,D,param.cols());
    // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
    // start=clock();
    
    if (activation=="relu" && param.rows()!=1) relu((layer_input*weight_sans_bias).rowwise() - bias.transpose(), layer_id);
    else sigmoid((layer_input*weight_sans_bias).rowwise() - bias.transpose(), layer_id);
    // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
}

void Net::forward(MatrixXd &x){
    outputs[0] = x;
    for(int i=1;i<num_layers;i++) linearForward(outputs[i-1], weights[i], activation, i);
    return;
}

void Net::linearBackward(MatrixXd &received_msg, int layer_id, bool is_output_layer){

    // clock_t start;
    MatrixXd layer_output = outputs[layer_id];
    double N = outputs[layer_id-1].rows();
    double Di = outputs[layer_id-1].cols();
    double Do = layer_output.cols();
    // start=clock();
    if(!is_output_layer){
        if(activation=="relu"){
            for(int i=0;i<received_msg.rows();i++){
                for(int j=0;j<received_msg.cols();j++){
                    if(layer_output(i,j)<=0) received_msg(i,j) = 0;
                }
            }
        }
        else{
            MatrixXd temp = (1-layer_output.array()).matrix();
            temp = layer_output.matrix().cwiseProduct(temp);
            received_msg = layer_output.cwiseProduct(temp);
        }
    }
    // cout << "1st" <<( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
    // start=clock();
    MatrixXd d1 = (1/N)*outputs[layer_id-1].transpose()*received_msg;
    MatrixXd d2 = (1/N)*received_msg.colwise().sum();
    
    // cout <<"2nd" << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
    // start=clock();
    for(int i=0;i<Di+1;i++){
        for(int j=0;j<Do;j++){
            if(i<Di) derivatives[layer_id](i,j) = d1(i,j);
            else derivatives[layer_id](i,j) = d2(0,j);
        }
    }    
    // cout << "3rd" <<( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
    // start=clock();
    received_msg = received_msg*weights[layer_id].block(0,0,Di,Do).transpose();
    // cout << "4th" <<( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
    return;
    
}

void Net::backward(MatrixXd &y){
    bool is_output_layer = true;
    // clock_t start;
    // start=clock();
    MatrixXd send_msg = outputs[num_layers-1] - y;
    // cout << "backward" << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
    for(int i=num_layers-1;i>0;i--){
        linearBackward(send_msg, i, is_output_layer);
        is_output_layer = false;
    }
    return;
}

double Net::calc_loss(MatrixXd &y){
    MatrixXd ypred = outputs[num_layers-1];
    MatrixXd ones = MatrixXd::Ones(y.rows(),y.cols());
    double cls_loss = 0;
    
    MatrixXd total_loss = y.cwiseProduct(ypred.unaryExpr(&log_unary)) + (ones-y).cwiseProduct((ones-ypred).unaryExpr(&log_unary));
    cls_loss = -1*total_loss.colwise().mean()(0,0);
    
    double reg_loss = 0;
    // for (int i=1;i<num_layers;i++){
    //     vector<vector<double> > param = sliceMatrix(weight[layer_id],0,0,num_nodes[layer_id-1]-1,0);
    //     reg_loss = reg_loss + weight_decay*matsum(param)
    // }
    return cls_loss + reg_loss;
    
}

void Net::updateParam(){
    for(int i=num_layers-1;i>0;i--){
        prev_grad[i] = momentum*prev_grad[i] - lr*derivatives[i];
        weights[i] = weights[i] + prev_grad[i];
    }
    return;
}

double Net::calc_error(MatrixXd &ytrue){
    MatrixXd ypred = outputs[num_layers-1];
    double error = 0;
    for(int i=0;i<ypred.size();i++){
        if (ypred(i,0)>=threshold) ypred(i,0) = 1;
        else ypred(i,0) = 0;
        if(ypred(i,0)!=ytrue(i,0)) error++;
    }
    return error/ypred.rows();
}

// MatrixXd Net::predictions(MatrixXd &x){
//     forward(x);
//     MatrixXd ypred = outputs[num_layers-1];
//     for(int i=0;i<ypred.size();i++){
//         if (ypred(i,0)>=threshold) ypred(i,0) = 1;
//         else ypred(i,0) = 0;
//     }
//     return ypred;
// }


int main(){
    
    // MatrixXd X = MatrixXd::Zero(4,2);    
    // MatrixXd Y = MatrixXd::Zero(4,1);

    TrainingData train_data, test_data;
    train_data.load_data("train.csv");
    cout << "Training data read!" << endl;
    test_data.load_data("test.csv");
    cout << "Test data read!" << endl;
    Net net;
    int N = train_data.data.x.rows();
    int D = train_data.data.x.cols();
    int M = train_data.data.y.cols();
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
        // start = clock();
        net.backward(batch.y);
        // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
        // start = clock();
        net.updateParam();
        // cout << ( clock() - start ) / (double) CLOCKS_PER_SEC << endl;
        cout << "Iteration: " << iters << "| Batch Loss: " << net.calc_loss(batch.y) << " | Batch Error: " << net.calc_error(batch.y) << endl;
        
        iters++;
    }
}