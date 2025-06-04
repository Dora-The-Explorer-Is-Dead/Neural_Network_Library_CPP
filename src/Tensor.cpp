#include "..\headers\Tensor.hpp"

bool Tensor::valid_dimensions(const Tensor& other) const {
    if (this->dim_4 != other.dim_4 ||
        this->dim_3 != other.dim_3 ||
        this->dim_2 != other.dim_2 ||
        this->dim_1 != other.dim_1) return false;

    else return true;
}

bool Tensor::valid_dimensions(int d4, int d3, int d2, int d1) const {
    if ((d4 < 0 || d4 >= dim_4) ||
        (d3 < 0 || d3 >= dim_3) ||
        (d2 < 0 || d2 >= dim_2) ||
        (d1 < 0 || d1 >= dim_1)) return false;

    else return true;
}

Tensor::Tensor(int d4, int d3, int d2, int d1) : dim_4(d4), dim_3(d3), dim_2(d2), dim_1(d1) {
    if (d4 <= 0 || d3 <= 0 || d2 <= 0 || d1 <= 0) {
        throw std::invalid_argument("All tensor dimensions must be positive");
    }
    data.resize((dim_4 * dim_3 * dim_2 * dim_1), 0.0);
}

Tensor::Tensor(const Tensor& other) {
    this->dim_4 = other.dim_4;
    this->dim_3 = other.dim_3;
    this->dim_2 = other.dim_2;
    this->dim_1 = other.dim_1;

    this->data = other.data;
}

Tensor::Tensor(Tensor&& other) noexcept : dim_4(other.dim_4), dim_3(other.dim_3), dim_2(other.dim_2), dim_1(other.dim_1), data(move(other.data)) {

    other.dim_4 = other.dim_3 = other.dim_2 = other.dim_1 = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        dim_4 = other.dim_4;
        dim_3 = other.dim_3;
        dim_2 = other.dim_2;
        dim_1 = other.dim_1;
        data = move(other.data);
        
        other.dim_4 = other.dim_3 = other.dim_2 = other.dim_1 = 0;
    }
    return *this;
}

int Tensor::get_dim4() const {return dim_4;}
int Tensor::get_dim3() const {return dim_3;}
int Tensor::get_dim2() const {return dim_2;}
int Tensor::get_dim1() const {return dim_1;}
int Tensor::get_size() const {return data.size();}
double* Tensor::get_data() {return data.data();} // for read and write
const double* Tensor::get_data() const {return data.data();} // for read only

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        this->dim_4 = other.dim_4;
        this->dim_3 = other.dim_3;
        this->dim_2 = other.dim_2;
        this->dim_1 = other.dim_1;

        this->data = other.data;
    }
    return *this;
}

double& Tensor::operator()(int d4, int d3, int d2, int d1) { // zero_indexing
    if (!valid_dimensions(d4, d3, d2, d1)) throw out_of_range("Tensor index out of range");
    return data[((d4 * dim_3 + d3) * dim_2 + d2) * dim_1 + d1];
}

const double& Tensor::operator()(int d4, int d3, int d2, int d1) const { // zero_indexing
    if (!valid_dimensions(d4, d3, d2, d1)) throw out_of_range("Tensor index out of range");
    return data[((d4 * dim_3 + d3) * dim_2 + d2) * dim_1 + d1];
}

Tensor Tensor::operator+(const Tensor& other) const {
    if(!valid_dimensions(other)) throw invalid_argument("Tensor dimensions don't match for element-wise addition");
    
    Tensor result(*this); 
    result += other;    
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if(!valid_dimensions(other)) throw invalid_argument("Tensor dimensions don't match for element-wise subtraction");
    
    Tensor result(*this); 
    result -= other;     
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if(!valid_dimensions(other)) throw invalid_argument("Tensor dimensions don't match for element-wise multiplication");
    
    Tensor result(*this); 
    result *= other;      
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if(!valid_dimensions(other)) throw invalid_argument("Tensor dimensions don't match for element-wise division");
    
    Tensor result(*this); 
    result /= other;      
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if(!valid_dimensions(other)) throw invalid_argument("Tensor dimensions don't match for addition");
    
    for (size_t i = 0; i < data.size(); i++) {
        this->data[i] += other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if(!valid_dimensions(other)) throw invalid_argument("Tensor dimensions don't match for subtraction");
    
    for (size_t i = 0; i < data.size(); i++) {
        this->data[i] -= other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if(!valid_dimensions(other)) throw invalid_argument("Tensor dimensions don't match for multiplication");
    
    for (size_t i = 0; i < data.size(); i++) {
        this->data[i] *= other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if(!valid_dimensions(other)) throw invalid_argument("Tensor dimensions don't match for division");
    
    for (size_t i = 0; i < data.size(); i++) {
        if (other.data[i] == 0.0) throw runtime_error("Division by Zero");
        this->data[i] /= other.data[i];
    }
    return *this;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result(*this);
    double* result_data = result.get_data();
    const size_t size = result.get_size();

    for (size_t i = 0; i < size; ++i) {
        result_data[i] *= scalar;
    }
    return result;
}

Tensor& Tensor::operator*=(double scalar) {
    double* data_ptr = this->get_data();
    const size_t size = this->get_size();
    
    for (size_t i = 0; i < size; ++i) {
        data_ptr[i] *= scalar;
    }
    return *this;
}

Tensor Tensor::operator-(double scalar) const {
    Tensor result(*this);
    double* result_data = result.get_data();
    const size_t size = result.get_size();

    for (size_t i = 0; i < size; ++i) {
        result_data[i] -= scalar;
    }
    return result;
}

Tensor& Tensor::operator-=(double scalar) {
    double* data_ptr = this->get_data();
    const size_t size = this->get_size();
    
    for (size_t i = 0; i < size; ++i) {
        data_ptr[i] -= scalar;
    }
    return *this;
}

Tensor Tensor::operator+(double scalar) const {
    Tensor result(*this);
    double* result_data = result.get_data();
    const size_t size = result.get_size();

    for (size_t i = 0; i < size; ++i) {
        result_data[i] += scalar;
    }
    return result;
}

Tensor& Tensor::operator+=(double scalar) {
    double* data_ptr = this->get_data();
    const size_t size = this->get_size();
    
    for (size_t i = 0; i < size; ++i) {
        data_ptr[i] += scalar;
    }
    return *this;
}

Tensor Tensor::sqrt() const {
    Tensor result(*this);
    double* data = result.get_data();
    for (int i = 0; i < this->get_size(); i++) {
        data[i] = std::sqrt(data[i]);
    }
    return result;
}

Tensor Tensor::matmult(const Tensor& other) const {
    if (this->get_dim1() != other.get_dim2()) {
        throw invalid_argument("Incompatible dimensions for matrix multiplication");
    }
    
    int batch_size = this->get_dim4();
    int channels = this->get_dim3();
    int m = this->get_dim2();  // rows of first matrix
    int k = this->get_dim1();  // cols of first matrix / rows of second matrix
    int n = other.get_dim1();  // cols of second matrix
    
    if (batch_size != other.get_dim4() || channels != other.get_dim3()) {
        throw invalid_argument("Batch or channel dimensions don't match");
    }
    
    Tensor result(batch_size, channels, m, n);

    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    
                    const double* this_row = &(*this)(b, c, i, 0);
                    for (int p = 0; p < k; p++) {
                        sum += this_row[p] * other(b, c, p, j);
                    }
                    
                    result(b, c, i, j) = sum;
                }
            }
        }
    }
    
    return result;
}

/* Tensor Tensor::matmult_broadcast(const Tensor& other) const {
    if (this->dim_1 != other.dim_2) {
        throw invalid_argument("Incompatible inner dimensions for matrix multiplication");
    }
    
    int batch_size = this->dim_4;
    int channels = this->dim_3;
    int height = this->dim_2;
    int input_features = this->dim_1;
    int output_features = other.dim_1;
    
    Tensor result(batch_size, channels, height, output_features);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int out_f = 0; out_f < output_features; ++out_f) {
                    double sum = 0.0;
                    
                    for (int in_f = 0; in_f < input_features; ++in_f) {
                        sum += (*this)(b, c, h, in_f) * other(0, 0, in_f, out_f);
                    }
                    
                    result(b, c, h, out_f) = sum;
                }
            }
        }
    }
    
    return result;
}

*/

Tensor Tensor::matmult_broadcast(const Tensor& other) const {
    if (this->dim_1 != other.dim_2) {
        throw invalid_argument("Incompatible inner dimensions for matrix multiplication");
    }
    
    int batch_size = this->dim_4;
    int channels = this->dim_3;
    int height = this->dim_2;
    int input_features = this->dim_1;
    int output_features = other.dim_1;
    
    Tensor result(batch_size, channels, height, output_features);

    const double* this_data = this->get_data();
    const double* other_data = other.get_data();
    double* result_data = result.get_data();
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int out_f = 0; out_f < output_features; ++out_f) {
                    double sum = 0.0;
                    
                    int this_base = ((b * channels + c) * height + h) * input_features;
                    for (int in_f = 0; in_f < input_features; ++in_f) {
                        int other_idx = in_f * output_features + out_f;
                        sum += this_data[this_base + in_f] * other_data[other_idx];
                    }
                    
                    int result_idx = ((b * channels + c) * height + h) * output_features + out_f;
                    result_data[result_idx] = sum;
                }
            }
        }
    }
    
    return result;
}

Tensor Tensor::reshape(int new_d4, int new_d3, int new_d2, int new_d1) const {
    int old_size = dim_4 * dim_3 * dim_2 * dim_1;
    int new_size = new_d4 * new_d3 * new_d2 * new_d1;

    if (old_size != new_size) {
        throw invalid_argument("Cannot reshape tensor: new shape has different total size. Old size: " + to_string(old_size) + ", New size: " + to_string(new_size));
    }

    Tensor result(new_d4, new_d3, new_d2, new_d1);

    for (int i = 0; i < old_size; i++) {
        result.get_data()[i] = this->get_data()[i];
    }

    return result;
}

Tensor Tensor::transpose(int dim_a, int dim_b) const {
    if (dim_a < 0 || dim_a > 3 || dim_b < 0 || dim_b > 3) {
        throw invalid_argument("Invalid dimensions for transpose. Must be 0, 1, 2, or 3");
    }

    if (dim_a == dim_b) return *this;

    int new_dims[4] = {dim_4, dim_3, dim_2, dim_1};
    swap(new_dims[dim_a], new_dims[dim_b]);

    Tensor result(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);

    const double* src_data = this->get_data();
    double* dst_data = result.get_data();

    int total_elements = dim_4 * dim_3 * dim_2 * dim_1;

    for (int i = 0; i < total_elements; i++) {
        int coords[4];
        int temp = i;
        
        coords[3] = temp % dim_1; temp /= dim_1;  // dim_1 coordinate
        coords[2] = temp % dim_2; temp /= dim_2;  // dim_2 coordinate  
        coords[1] = temp % dim_3; temp /= dim_3;  // dim_3 coordinate
        coords[0] = temp;                         // dim_4 coordinate
        
        swap(coords[dim_a], coords[dim_b]);
        
        int new_idx = ((coords[0] * new_dims[1] + coords[1]) * new_dims[2] + coords[2]) * new_dims[3] + coords[3];
        
        if (new_idx >= 0 && new_idx < total_elements) {
            dst_data[new_idx] = src_data[i];
        }
    }

    return result;
}

Tensor Tensor::sum_axis(int axis) const {
    if (axis == 0) {
        Tensor result(1, dim_3, dim_2, dim_1);
        
        const double* src_data = this->data.data();
        double* result_data = result.data.data();
        
        int elements_per_batch = dim_3 * dim_2 * dim_1;

        std::fill(result_data, result_data + elements_per_batch, 0.0);
        
        for (int b = 0; b < dim_4; ++b) {
            int batch_offset = b * elements_per_batch;
            
            for (int i = 0; i < elements_per_batch; ++i) {
                result_data[i] += src_data[batch_offset + i];
            }
        }
        
        return result;
    }
    
    throw invalid_argument("Sum axis not implemented for this dimension");
}

Tensor Tensor::sum_across_samples() const {
    int features = this->dim_1;
    Tensor result(1, 1, features, 1);

    const double* src_data = this->data.data();
    double* result_data = result.data.data();

    std::fill(result_data, result_data + features, 0.0);
    
    int feature_stride = 1;
    int height_stride = dim_1;
    int channel_stride = dim_2 * dim_1;
    int batch_stride = dim_3 * dim_2 * dim_1;
    
    for (int b = 0; b < dim_4; ++b) {
        for (int c = 0; c < dim_3; ++c) {
            for (int h = 0; h < dim_2; ++h) {
                int offset = b * batch_stride + c * channel_stride + h * height_stride;

                for (int f = 0; f < features; ++f) {
                    result_data[f] += src_data[offset + f];
                }
            }
        }
    }
    
    return result;
}

void Tensor::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

void Tensor::zero() {
    fill(0.0);
}

bool Tensor::empty() const {
    return data.empty();
}

vector<int> Tensor::shape() const {
    return {dim_4, dim_3, dim_2, dim_1};
}

void Tensor::print(const string& name) const {
    if (!name.empty()) {
        cout << name << ": ";
    }
    cout << "Tensor(" << dim_4 << ", " << dim_3 << ", " << dim_2 << ", " << dim_1 << ")\n";
    
    cout << "First 10 values: ";
    int print_count = min(10, (int)data.size());
    for (int i = 0; i < print_count; i++) {
        cout << data[i] << " ";
    }
    cout << "\n";
}