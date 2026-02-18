#pragma once
#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

//OVERFITTING!!!!

inline void printVec(std::vector<double>& vec) {
	for (size_t i = 0; i < vec.size(); i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << std::endl;
}

inline void testXOR() {
	std::vector<std::vector<double>> inputValue{ {1,0}, {0,1}, {0,0}, {1,1} };
	std::vector<std::vector<double>> correctOutput{ {1}, {1}, {0}, {0} };

	Network myNetwork("file_TestXOR.txt", 1, 2, 1, { 8 }, SIGMOID);

	for (size_t i = 0; i <= 10000; i++)
	{
		for (size_t j = 0; j < 4; j++) {
			myNetwork.train(inputValue[j], correctOutput[j]);
		}

		if (i % 2500 == 0)
		{
			std::cout << i << std::endl;
			for (size_t j = 0; j < 4; j++)
			{
				std::vector<double> output = myNetwork.feedForward(inputValue[j]);
				std::cout << inputValue[j][0] << " " << inputValue[j][1] << ": ";
				printVec(output);
			}

		}
	}

	std::cout << "\nOutput:\n";
	std::vector<double> testInputValue;
	testInputValue.resize(2);

	testInputValue[0] = rand() % 2;
	testInputValue[1] = rand() % 2;

	std::cout << "Random input: [" << testInputValue[0] << ", " << testInputValue[1] << "]\n";
	std::cout << "output: ";
	std::vector<double> finalOutput = myNetwork.feedForward(testInputValue);
	printVec(finalOutput);
}