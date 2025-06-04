# Neural_Network_Library_CPP

A lightweight, efficient neural network library implemented in C++ for educational and research purposes. This library provides a clean, object-oriented approach to building and training neural networks from scratch.

## 🚀 Features

- **Pure C++ Implementation** - No external dependencies beyond standard library
- **Object-Oriented Design** - Clean, modular architecture for easy extension
- **MNIST Support** - Built-in functionality for MNIST dataset training and testing
- **Flexible Architecture** - Easily configurable network layers and parameters
- **Educational Focus** - Clear, readable code perfect for learning neural network fundamentals
- **Cross-Platform** - Works on Windows, macOS, and Linux

## 📁 Project Structure

```
Neural_Network_Library_CPP/
├── src/                    # Core library source files
├── include/                # Header files
├── main/                   # Example programs and demos
├── .vscode/               # VS Code configuration
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

- **C++ Compiler** (C++11 or later)
  - GCC 4.8+ on Linux
  - MSVC 2015+ on Windows
  - Clang 3.4+ on macOS
- **CMake** (optional, for build automation)

### Building the Project

#### Option 1: Using g++
```bash
# Clone the repository
git clone https://github.com/Dora-The-Explorer-Is-Dead/Neural_Network_Library_CPP.git
cd Neural_Network_Library_CPP

# Compile example programs
g++ -O2 -std=c++11 -I./include main/main.cpp src/*.cpp -o neural_network_demo
```

#### Option 2: Using CMake (if CMakeLists.txt available)
```bash
mkdir build
cd build
cmake ..
make
```

### Dataset Setup

This project uses the MNIST dataset for training and testing. Download the CSV files:

1. **Download MNIST CSV files:**
   - https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download

2. **Place them in the project root directory:**
   ```
   Neural_Network_Library_CPP/
   ├── mnist_train.csv
   ├── mnist_test.csv
   └── ...
   ```

3. **Alternative: Use the download script (if available):**
   ```bash
   python download_data.py
   ```

## 🔧 Usage

### Basic Example

```cpp
#include "NeuralNetwork.h"
#include "DataLoader.h"

int main() {
    // Create a neural network with 784 inputs, 128 hidden neurons, 10 outputs
    NeuralNetwork nn(784, 128, 10);
    
    // Load MNIST training data
    DataLoader loader("mnist_train.csv");
    auto training_data = loader.load();
    
    // Train the network
    nn.train(training_data, epochs=100, learning_rate=0.01);
    
    // Test the network
    DataLoader test_loader("mnist_test.csv");
    auto test_data = test_loader.load();
    double accuracy = nn.test(test_data);
    
    std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
    
    return 0;
}
```

### Advanced Configuration

```cpp
// Create a custom network architecture
NeuralNetwork nn;
nn.addLayer(784, "input");
nn.addLayer(256, "relu");
nn.addLayer(128, "relu");
nn.addLayer(64, "relu");
nn.addLayer(10, "softmax");

// Configure training parameters
TrainingConfig config;
config.learning_rate = 0.001;
config.batch_size = 32;
config.epochs = 200;
config.momentum = 0.9;

nn.train(training_data, config);
```

## 📊 Performance

Typical performance on MNIST dataset:
- **Training Time**: ~5-10 minutes (CPU, varies by architecture)
- **Test Accuracy**: 95-98% (depending on network configuration)
- **Memory Usage**: ~50-100 MB during training

## 🧪 Examples

The `main/` directory contains several example programs:

- `main.cpp` - Basic MNIST training example
- `network_demo.cpp` - Demonstrates different network architectures
- `data_visualization.cpp` - Visualizes training progress and results

To run examples:
```bash
cd main
g++ -O2 -std=c++11 -I../include main.cpp ../src/*.cpp -o example
./example
```

## 🏗️ Architecture

### Core Classes

- **`NeuralNetwork`** - Main network class handling forward/backward propagation
- **`Layer`** - Individual network layer with weights and activations
- **`ActivationFunction`** - Supports ReLU, Sigmoid, Tanh, Softmax
- **`DataLoader`** - Handles CSV data loading and preprocessing
- **`Matrix`** - Custom matrix operations for linear algebra

### Key Features

- **Modular Design** - Easy to add new activation functions and layer types
- **Efficient Matrix Operations** - Optimized for training performance
- **Memory Management** - Proper resource handling and cleanup
- **Error Handling** - Robust error checking and validation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Guidelines

- Follow C++ best practices and modern C++ standards
- Use descriptive variable and function names
- Add comments for complex algorithms
- Ensure code compiles without warnings
- Test your changes with MNIST dataset

## 📞 Contact

- **GitHub**: [Dora-The-Explorer-Is-Dead](https://github.com/Dora-The-Explorer-Is-Dead)
- **Project Link**: [Neural_Network_Library_CPP](https://github.com/Dora-The-Explorer-Is-Dead/Neural_Network_Library_CPP)

---

**Happy Neural Networking! 🧠✨**
