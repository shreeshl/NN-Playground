#include <iostream>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <cmath>

using namespace std;

vector<double> operator+(const vector<double>& lhs, const vector<double>& rhs){
    vector<double> result;
    for(int i=0;i<lhs.size();i++){
        result.push_back(lhs[i]+rhs[i]);
    }
    return result;
}


vector<double> operator*(const double& lhs, const vector<double>& rhs){
    vector<double> result;
    for(int i=0;i<rhs.size();i++){
        result.push_back(lhs*rhs[i]);
    }
    return result;
}

template<typename T>
void printMatrix(const T& mat) {
    cout<<"\nPrinting Matrix : \n";
    for(int i=0 ; i<mat.size() ; i++) {
        for(int j=0 ; j<mat[0].size() ; j++)
            cout<< mat[i][j] << " ";
        cout<<endl;
    }
}

template<typename T>
void printVector(const T& mat) {
    cout<<"\nPrinting Vector : \n";
    for(int i=0 ; i<mat.size() ; i++) {
        cout<< mat[i] << " ";
    }
    cout<<endl;
}

template<typename T>
vector<vector<T> > Transpose(const vector<vector<T> > & mat) {
    int m = mat.size();
    int n = mat[0].size();
    vector<vector<T> > result(n, vector<T>(m));
    for(int i=0 ; i<m ; i++) {
        for(int j=0 ; j<n ; j++){
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

template<typename T>
vector<vector<T> > sliceMatrix(const vector<vector<T> >& mat, int r1, int c1, int r2, int c2) {
    if(r2==0) r2 = mat.size();
    if(c2==0) c2 = mat[0].size();

    vector<vector<T> > result(r2-r1,vector<T>(c2-c1));
    
    for(int i=0 ; i<r2-r1; i++) {
        for(int j=0 ; j<c2-c1 ; j++){
            result[i][j] = mat[i+r1][j+c1];    
        }
    }
    return result;
}

template<typename T>
vector<vector<T> > matadd(const vector<vector<T> >& mat1, const vector<vector<T> >& mat2){
    int r1 = mat1.size();
    int c1 = mat1[0].size();
    int r2 = mat2.size();
    int c2 = mat2[0].size();

    assert(r1==r2 || c1==c2);

    vector<vector<T> > result = mat1;
    for(int i=0;i<r1;i++){
        for(int j=0;j<c1;j++){
            if(r1==r2){
                if(c2==1) result[i][j] = result[i][j] + mat2[i][0];
                else result[i][j] = result[i][j] + mat2[i][j];
            }
            else{
                if(r2==1) result[i][j] = result[i][j] + mat2[0][j];
                else result[i][j] = result[i][j] + mat2[i][j];
            }
        }
    } 
    return result;
}

// template<class T>
// vector<vector<T> > matmul(vector<vector<T> >& mat1, vector<vector<T> >& mat2, vector<int> vals, bool t1 = false, bool t2 = false) {
    
//     // General Matrix Multiplier function

//     if (t1) mat1 = Transpose(mat1);
//     if (t2) mat2 = Transpose(mat2);

//     int row1_s = vals[0];
//     int col1_s = vals[1];
//     int row1_e = vals[2];
//     int col1_e = vals[3];

//     int row2_s = vals[4];
//     int col2_s = vals[5];
//     int row2_e = vals[6];
//     int col2_e = vals[7];

//     if(row1_e==0) row1_e = mat1.size();
//     if(row2_e==0) row2_e = mat2.size();
//     if(col1_e==0) col1_e = mat1[0].size();
//     if(col2_e==0) col2_e = mat2[0].size();
//     assert(col1_e-col1_s==row2_e-row2_s);

    
//     vector<vector<T> > result(row1_e-row1_s, vector<T>(col2_e-col2_s));
    
//     for(int i=0 ; i<row1_e-row1_s ; i++) {
//         for(int j=0; j<col2_e-col2_s ; j++) {
//             for(int k=0 ; k<col1_e-col1_s ; k++) {
//                 // cout << i << " " << j <<" " <<  k << " " << mat1[i][col1_s+k]*mat2[row2_s+k][j] << endl;
//                 result[i][j] = result[i][j] + mat1[row1_s+i][col1_s+k]*mat2[row2_s+k][col2_s+j];
//             }
//         }
//     }

//     if (t1) mat1 = Transpose(mat1);
//     if (t2) mat2 = Transpose(mat2);

//     return result;
// }

template<class T>
vector<vector<T> > matmul(vector<vector<T> >& mat1, vector<vector<T> >& mat2, bool t1 = false, bool t2 = false) {
    
    // General Matrix Multiplier function

    if (t1) mat1 = Transpose(mat1);
    if (t2) mat2 = Transpose(mat2);

    assert(mat1[0].size()==mat2.size());
    vector<vector<T> > result(mat1.size(), vector<T>(mat2[0].size()));
    
    for(int i=0 ; i<mat1.size() ; i++) {
        for(int j=0; j<mat2[0].size() ; j++) {
            for(int k=0 ; k<mat1[0].size() ; k++) {
                result[i][j] = result[i][j] + mat1[i][k]*mat2[k][j];
            }
        }
    }
    if (t1) mat1 = Transpose(mat1);
    if (t2) mat2 = Transpose(mat2);

    return result;
}

template<class T>
vector<vector<T> > elementwisemult(const vector<vector<T> >& mat1,const vector<vector<T> >& mat2) {
    int m = mat1.size();
    int n = mat1[0].size();
    vector<vector<T> > result = mat1;
    
    for(int i=0 ; i<m ; i++) {
        for(int j=0; j<n ; j++) {
            result[i][j] = result[i][j]*mat2[i][j];
        }
    }
    return result;
}

template<class T>
vector<T> matsum(const vector<vector<T> >& mat, int axis = 0) {
    int m = mat.size();
    int n = mat[0].size();
    unordered_map<double,double> count;
    vector<T> result;
    
    for(int i=0 ; i<m ; i++) {
        for(int j=0; j<n ; j++) {
            if(axis==0) count[j] = count[j]+mat[i][j];
            else count[j] = count[j]+mat[i][j];
        }
    }
    if(axis==0) for(int i=0;i<n;i++) result.push_back(count[i]);
    else for(int j=0;j<m;j++) result.push_back(count[j]);
    
    return result;
}

vector<vector<double>> operator*(const double lhs, const vector<vector<double>>& rhs){
    int m = rhs.size();
    int n = rhs[0].size();
    vector<vector<double>> result = rhs;
    for(int i=0 ; i<m ; i++) {
        for(int j=0; j<n ; j++) {
            result[i][j] = result[i][j]*lhs;
        }
    }
    return result;
}

vector<vector<double>> operator-(const double lhs, const vector<vector<double>>& rhs){
    int m = rhs.size();
    int n = rhs[0].size();
    vector<vector<double>> result = rhs;
    for(int i=0 ; i<m ; i++) {
        for(int j=0; j<n ; j++) {
            result[i][j] = lhs - result[i][j];
        }
    }
    return result;
}