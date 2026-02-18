#pragma once

#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <cstdlib>
#include <random>
#include <iomanip>
#include <limits>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

enum ActivationType {
	RELU,
	SIGMOID
};

inline void randomizeForSigmoid(Matrix& vec) {
	std::random_device rd;
	std::mt19937 gen(rd());

	double scale = 1.0 / sqrt(vec[0].size());
	std::uniform_real_distribution<> distribution(-scale, scale);

	for (size_t i = 0; i < vec.size(); i++) {
		for (size_t j = 0; j < vec[0].size(); j++) {
			vec[i][j] = distribution(gen);
		}
	}
}

inline double sigmoid(const double x) {
	return 1.0 / (1.0 + exp(-x));
}

class Layer {
public:
	const int inputSize, outputSize;
	Matrix weights;
	Vector bias, calculations, postActivation, E, inputVec;
	ActivationType activationType;

	Layer(int inSize, int outSize, ActivationType actType)
		: inputSize(inSize), outputSize(outSize), activationType(actType) {

		bias.resize(outputSize);
		calculations.resize(outputSize);
		postActivation.resize(outputSize);
		weights.resize(outputSize);

		for (size_t i = 0; i < outputSize; i++) {
			weights[i].resize(inputSize);
		}

		if (activationType == SIGMOID) {
			randomizeForSigmoid(weights);
		}
	}

	~Layer() {}

	double derivative(double& x) {
		if (activationType == SIGMOID) {
			return x * (1 - x);
		}
	}

	Vector feedForward(Vector& input) {
		inputVec = input;

		for (int i = 0; i < outputSize; i++) {
			double result = 0;

			for (int j = 0; j < inputSize; j++) {
				result += input[j] * weights[i][j];
			}

			result += bias[i];
			calculations[i] = result;

			if (activationType == SIGMOID) {
				postActivation[i] = sigmoid(result);
			}
		}

		return postActivation;
	}

	void train(Vector& Eprevious, Matrix& weightsPrevious) {
		E.assign(outputSize, 0.0);

		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < Eprevious.size(); j++) {
				E[i] += Eprevious[j] * weightsPrevious[j][i] * derivative(postActivation[i]);
			}
		}
	}

	void updateWeights(const double& learningRate) {
		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < inputSize; j++) {
				weights[i][j] = weights[i][j] - (learningRate * E[i] * inputVec[j]);
			}

			bias[i] = bias[i] - (learningRate * E[i]);
		}
	}
};

class Network {
public:
	const std::string fileName;
	int numHiddenLayers, inSize, outSize;
	double learningRate = 0.2;
	std::vector<int> neuronsInLayers;
	ActivationType activationType;
	std::vector<Layer> layers;
	Vector realOutput, E;

	Network(std::string fName, int numHLayers, int inputS, int outputS, std::vector<int> neuronsInL, ActivationType actType)
		: fileName(fName), numHiddenLayers(numHLayers), inSize(inputS), outSize(outputS), neuronsInLayers(neuronsInL), activationType(actType) {

		layers.reserve(numHiddenLayers + 1);

		for (int i = 0; i < numHiddenLayers; i++) {
			if (i == 0) {
				layers.push_back(Layer(inSize, neuronsInLayers[i], activationType));
			}
			else {
				layers.push_back(Layer(neuronsInLayers[i - 1], neuronsInLayers[i], activationType));
			}
		}

		if (numHiddenLayers == 0) {
			layers.push_back(Layer(inSize, outSize, activationType));
		}
		else {
			layers.push_back(Layer(neuronsInLayers.back(), outSize, activationType));
		}
	}

	~Network() {}

	Vector feedForward(const Vector& input) {
		Vector tempInput = input;

		for (size_t i = 0; i <= numHiddenLayers; i++) {
			tempInput = layers[i].feedForward(tempInput);
		}

		return tempInput;
	}

	void train(const Vector& input, Vector& correctOutput) {
		realOutput = feedForward(input);
		E.assign(outSize, 0.0);

		for (int i = 0; i < realOutput.size(); i++) {
			E[i] = (realOutput[i] - correctOutput[i]) * realOutput[i] * (1 - realOutput[i]);
		}

		layers.back().E = E;

		for (int i = numHiddenLayers - 1; i >= 0; i--) {
			layers[i].train(layers[i + 1].E, layers[i + 1].weights);
		}

		for (int i = 0; i < layers.size(); i++) {
			layers[i].updateWeights(learningRate);
		}
	}

	void saveData() {
		std::fstream file;
		file.open(fileName, std::ios::out);

		if (file.good() != true || file.is_open() != true) {
			std::cout << "Error opening file for writing!" << std::endl;
			return;
		}

		file << "::::CONFIG::::\n";
		file << "Inputs (inSize): " << inSize << "\n";
		file << "Outputs (outSize): " << outSize << "\n";
		file << "Activation type (activationType): " << activationType << "\n";
		file << "Hidden layers count (numHiddenLayers): " << numHiddenLayers << "\n";
		file << "Neurons in layer count (Vector neuronsInLayers): ";

		for (size_t i = 0; i < neuronsInLayers.size(); i++) {
			file << neuronsInLayers[i] << " ";
		}
		file << "\n";
		file << "::::CONFIG::::\n";

		file << std::setprecision(std::numeric_limits<double>::max_digits10);

		file << "Weights:\n";
		for (int i = 0; i <= numHiddenLayers; i++) {
			for (int j = 0; j < layers[i].outputSize; j++) {
				file << "Neuron-w:" << i << "/" << j << ": ";
				for (int k = 0; k < layers[i].weights[j].size(); k++) {
					file << layers[i].weights[j][k] << " ";
				}
				file << "\n";
			}
			file << "\n";
		}

		file << "Bias:\n";
		for (int i = 0; i <= numHiddenLayers; i++) {
			for (int j = 0; j < layers[i].outputSize; j++) {
				file << layers[i].bias[j] << " ";
			}
			file << "\n";
		}

		file.close();
	}

	void loadData() {
		std::fstream file;
		file.open(fileName, std::ios::in);

		if (file.good() != true || file.is_open() != true) {
			return;
		}

		std::string dummy;
		int tempActivation;

		file >> dummy;

		file >> dummy >> dummy >> inSize;
		file >> dummy >> dummy >> outSize;
		file >> dummy >> dummy >> dummy >> tempActivation;
		activationType = (ActivationType)tempActivation;

		file >> dummy >> dummy >> dummy >> dummy >> numHiddenLayers;

		for (int i = 0; i < 6; i++) {
			file >> dummy;
		}

		neuronsInLayers.resize(numHiddenLayers);
		for (size_t i = 0; i < neuronsInLayers.size(); i++) {
			file >> neuronsInLayers[i];
		}

		file >> dummy;
		file >> dummy;

		for (int i = 0; i <= numHiddenLayers; i++) {
			for (int j = 0; j < layers[i].outputSize; j++) {
				file >> dummy;
				for (int k = 0; k < layers[i].weights[j].size(); k++) {
					file >> layers[i].weights[j][k];
				}
			}
		}

		file >> dummy;

		for (int i = 0; i <= numHiddenLayers; i++) {
			for (int j = 0; j < layers[i].outputSize; j++) {
				file >> layers[i].bias[j];
			}
		}

		file.close();
	}
};