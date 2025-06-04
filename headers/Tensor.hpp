#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
using namespace std;

class Tensor { 
private:
    int dim_4; // batch_size  // filters
    int dim_3; // channels    // kernels
    int dim_2; // rows        // rows
    int dim_1; // cols        // cols

    vector<double> data;

    bool valid_dimensions(const Tensor& other) const;

    bool valid_dimensions(int d4, int d3, int d2, int d1) const;

public:
    Tensor(int d4 = 1, int d3 = 1, int d2 = 1, int d1 = 1);

    Tensor(const Tensor& other);

    Tensor(Tensor&& other) noexcept;
    
    Tensor& operator=(Tensor&& other) noexcept;

    int get_dim4() const;
    int get_dim3() const;
    int get_dim2() const;
    int get_dim1() const;
    int get_size() const;
    double* get_data(); // for read and write
    const double* get_data() const; // for read only

    Tensor& operator=(const Tensor& other);

    // Indexing for setting
    double& operator()(int d4, int d3, int d2, int d1); // zero_indexing

    // Indexing for getting
    const double& operator()(int d4, int d3, int d2, int d1) const; // zero_indexing

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    Tensor operator+(double scalar) const;
    Tensor operator-(double scalar) const;
    Tensor operator*(double scalar) const;

    Tensor& operator+=(double scalar);
    Tensor& operator-=(double scalar);
    Tensor& operator*=(double scalar);

    Tensor sqrt() const;

    Tensor matmult(const Tensor& other) const;

    Tensor matmult_broadcast(const Tensor& other) const;

    Tensor reshape(int new_d4, int new_d3, int new_d2, int new_d1) const;

    Tensor transpose(int dim_a, int dim_b) const;

    Tensor sum_axis(int axis) const;

    Tensor sum_across_samples() const;

    void fill(double value);
    void zero();
    bool empty() const;
    vector<int> shape() const;
    void print(const string& name = "") const;
};