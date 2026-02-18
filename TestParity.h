#pragma once
#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

inline void testParity(const int& inputSize) {
	std::vector<int> hiddenLayers = { 32 };
	Network myNetwork("file_TestParity.txt", (int)hiddenLayers.size(), inputSize, 1, hiddenLayers, SIGMOID);

	const int trainCases = 200, testCases = 10;
	Matrix trainInputs(trainCases, Vector(inputSize, 0.0)), trainExpected(trainCases, Vector(1, 0.0));
	Matrix testInputs(testCases, Vector(inputSize, 0.0)), testExpected(testCases, Vector(1, 0.0));

	for (int i = 0; i < trainCases + testCases; i++) {
		int onesCount = 0;
		Vector tempInput(inputSize);
		for (int j = 0; j < inputSize; j++) {
			tempInput[j] = rand() % 2;
			onesCount += tempInput[j];
		}

		double expected = onesCount % 2;

		if (i < trainCases) {
			trainInputs[i] = tempInput;
			trainExpected[i][0] = expected;
		}
		else {
			testInputs[i - trainCases] = tempInput;
			testExpected[i - trainCases][0] = expected;
		}
	}

	for (int epoch = 0; epoch <= 10000; epoch++) {
		for (int j = 0; j < trainCases; j++) {
			myNetwork.train(trainInputs[j], trainExpected[j]);
		}

		if (epoch % 2000 == 0) {
			std::cout << epoch << "\n";
			for (int j = 0; j < trainCases; j++) {
				Vector output = myNetwork.feedForward(trainInputs[j]);
				for (int k = 0; k < inputSize; k++) std::cout << trainInputs[j][k] << " ";
				std::cout << " result: " << output[0] << " expected: " << trainExpected[j][0] << "\n";
			}
		}
	}

	std::cout << "\nTEST ON NEW RANDOM DATA\n";
	for (int j = 0; j < testCases; j++) {
		Vector output = myNetwork.feedForward(testInputs[j]);
		for (int k = 0; k < inputSize; k++) std::cout << testInputs[j][k] << " ";
		std::cout << " result: " << output[0] << " expected: " << testExpected[j][0] << "\n";
	}
}