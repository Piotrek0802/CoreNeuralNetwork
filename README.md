# CoreNeuralNetwork 🧠

A custom, multi-layer artificial neural network (MLP) built completely from scratch in pure C++. 

This project was developed to gain a deep understanding of neural network architectures, forward propagation, and memory management in C++ without relying on external machine learning frameworks like PyTorch or TensorFlow.

## ✨ Features
* **Zero Dependencies:** Built using only the C++ Standard Library (`<vector>`, `<iostream>`, `<fstream>`, `<random>`).
* **Dynamic Architecture:** Easily construct networks with any number of hidden layers and neurons per layer.
* **Custom Serialization:** Save and load trained network weights and biases using a structured, human-readable text file format.
* **Memory Efficient:** Object-oriented design utilizing `std::vector` with precise memory pre-allocation (`reserve`).

## 🚀 Quick Start (Example Usage)

Building and running the network is incredibly simple. You can define the architecture dynamically:

~~~cpp
#include "Network.h" // Replace with your actual header/file name

int main() {
    // 1. Define hidden layers (e.g., one hidden layer with 64 neurons)
    std::vector<int> hiddenLayers = { 64 };
    
    // 2. Build the network: (File name, Hidden layers count, Input size, Output size, Hidden layers structure, Activation)
    Network myNet("model.txt", 1, 784, 10, hiddenLayers, SIGMOID);
    
    // 3. Prepare dummy data (e.g., 784 inputs)
    std::vector<double> input(784, 1.0);
    
    // 4. Feed forward
    std::vector<double> output = myNet.feedForward(input);
    
    // 5. Save the state for later
    myNet.zapiszDane();
    
    return 0;
}
~~~

## 💾 Loading a Saved Model
You can revive a previously saved network using the overloaded constructor:

~~~cpp
// Automatically reads architecture, weights, and biases from the configuration file
Network trainedNet("model.txt"); 
~~~

## 🛠️ Compilation
Compile the project using any standard C++ compiler (C++11 or higher recommended):

~~~bash
g++ main.cpp -o CoreNN -std=c++17
./CoreNN
~~~