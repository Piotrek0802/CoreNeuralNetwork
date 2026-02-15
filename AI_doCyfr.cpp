#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <cstdlib>
#include <random>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;
enum rodzajAktywacji
{
	RELU,
	SIGMOID
};

void randDlaSigmoidy(Matrix& wek) {
	std::random_device rd;
	std::mt19937 gen(rd());

	double skala = 1.0 / sqrt(wek[0].size());
	std::uniform_real_distribution<> dystrybucja(-skala, skala);

	for (size_t i = 0; i < wek.size(); i++)
	{
		for (size_t j = 0; j < wek[0].size(); j++)
		{
			wek[i][j] = dystrybucja(gen);
		}
	}
}
double sigmoid(const double x) {
	return 1.0 / (1.0 + exp(-x));
}

class Layer
{
public:
	const int inputSize, outputSize;
	Matrix wagi;
	Vector bias, obliczenia, poAktywacji, E, wejscie;
	rodzajAktywacji rodzAktywacji;

	Layer(int inSize, int ouSize, rodzajAktywacji rrodzAktywacji)
		:inputSize(inSize), outputSize(ouSize), rodzAktywacji(rrodzAktywacji)
	{
		bias.resize(outputSize);
		obliczenia.resize(outputSize);
		poAktywacji.resize(outputSize);
		wagi.resize(outputSize);
		for (size_t i = 0; i < outputSize; i++)
		{
			wagi[i].resize(inputSize);
		}
		if (rodzAktywacji == SIGMOID)
		{
			randDlaSigmoidy(wagi);
		}
	}

	~Layer()
	{
	}


	double pochodna(double& x) {
		if (rodzAktywacji == SIGMOID)
		{
			return x * (1 - x);
		}
	}
	Vector feedForward(Vector& input) {
		wejscie = input;
		for (int i = 0; i < outputSize; i++)
		{
			double wynik = 0;
			for (int j = 0; j < inputSize; j++)
			{
				wynik += input[j] * wagi[i][j];
			}
			wynik += bias[i];
			obliczenia[i] = wynik;
			if (rodzAktywacji == SIGMOID)
			{
				poAktywacji[i] = sigmoid(wynik);
			}
		}
		return poAktywacji;
	}
	void train(Vector& Eprevious, Matrix& wagiPrevious) {
		E.assign(outputSize, 0.0);
		for (int i = 0; i < outputSize; i++)
		{
			for (int j = 0; j < Eprevious.size(); j++)
			{
				E[i] += Eprevious[j] * wagiPrevious[j][i] * pochodna(poAktywacji[i]);
			}
		}
	}
	void noweWagi(const double& silaUczenia) {
		for (int i = 0; i < outputSize; i++)
		{
			for (int j = 0; j < inputSize; j++)
			{
				wagi[i][j] = wagi[i][j] - (silaUczenia * E[i] * wejscie[j]);
			}
			bias[i] = bias[i] - (silaUczenia * E[i]);
		}
	}
private:

};
class Network
{
public:
	const std::string nazwaPliku;
	int ileHiddenLayers, inSize, outSize;
	double silaUczenia = 0.2;
	std::vector<int>ileNeuronsInLayers;
	rodzajAktywacji rodzAktywacji;
	std::vector<Layer> layers;
	Vector realOutput, E;
	Network(std::string nnazwaPliku, int iileHiddenLayers, int iinSize, int ooutSize, std::vector<int> iileNeuronsInLayers, rodzajAktywacji rrodzAktywacji)
		: nazwaPliku(nnazwaPliku), ileHiddenLayers(iileHiddenLayers), inSize(iinSize), outSize(ooutSize), ileNeuronsInLayers(iileNeuronsInLayers), rodzAktywacji(rrodzAktywacji)
	{
		layers.reserve(ileHiddenLayers + 1);

		for (int i = 0; i < ileHiddenLayers; i++)
		{
			if (i == 0) {
				layers.push_back(Layer(inSize, ileNeuronsInLayers[i], rodzAktywacji));
			}
			else {
				layers.push_back(Layer(ileNeuronsInLayers[i - 1], ileNeuronsInLayers[i], rodzAktywacji));
			}
		}

		if (ileHiddenLayers == 0) {
			layers.push_back(Layer(inSize, outSize, rodzAktywacji));
		}
		else {
			layers.push_back(Layer(ileNeuronsInLayers.back(), outSize, rodzAktywacji));
		}
	}
	~Network() {}

	Vector feedForward(const Vector& input) {
		Vector tempInput = input;
		for (size_t i = 0; i <= ileHiddenLayers; i++)
		{
			tempInput = layers[i].feedForward(tempInput);
		}
		return tempInput;
	}
	void train(const Vector& input, Vector& correctOutput) {
		realOutput = feedForward(input);
		E.assign(outSize, 0.0);
		for (int i = 0; i < realOutput.size(); i++)
		{
			E[i] = (realOutput[i] - correctOutput[i]) * realOutput[i] * (1 - realOutput[i]);
		}
		layers.back().E = E;

		for (int i = ileHiddenLayers - 1; i >= 0; i--)
		{
			layers[i].train(layers[i + 1].E, layers[i + 1].wagi);
		}

		for (int i = 0; i < layers.size(); i++)
		{
			layers[i].noweWagi(silaUczenia);
		}
	}
	void zapiszDane() {
		std::fstream plik;
		plik.open(nazwaPliku, std::ios::out);
		if (plik.good() != true || plik.is_open() != true)
		{
			std::cout << "Blad otwarcia pliku do zapisu!" << std::endl;
			return;
		}
		//Dane do konfiguracji
		plik << "::::CONFIG::::\n";
		plik << "Wejscia (inSize): " << inSize << "\n";
		plik << "Wyjscia (outSize): " << outSize << "\n";
		plik << "Rodzaj aktywacji (rodzAktywacji): " << rodzAktywacji << "\n";
		plik << "Ile ukrytch warstw (ileHiddenLayers): " << ileHiddenLayers << "\n";
		plik << "Ile neuronow w warstwie (Vector ileNeuronsInLayers): ";
		for (size_t i = 0; i < ileNeuronsInLayers.size(); i++)
		{
			plik << ileNeuronsInLayers[i] << " ";
		}
		plik << "\n";
		plik << "::::CONFIG::::\n";


		plik << "Wagi:\n";
		for (int i = 0; i <= ileHiddenLayers; i++)
		{
			for (int j = 0; j < layers[i].outputSize; j++)
			{
				plik << "Neuron-w" << i << "/" << j << ": ";
				for (int k = 0; k < layers[i].wagi[j].size(); k++)
				{
					plik << layers[i].wagi[j][k] << " ";
				}
				plik << "\n";
			}
			plik << "\n";
		}
		plik << "Bias:\n";
		for (int i = 0; i <= ileHiddenLayers; i++)
		{
			for (int j = 0; j < layers[i].outputSize; j++)
			{
				plik << layers[i].bias[j] << " ";
			}
			plik << "\n";
		}
	}
private:

};


int main()
{
	//test czy siec zyje!
	std::cout << "Budowanie sieci\n";

	std::vector<int> ukryte = { 4 }; 
	Network mojaSiec("moj_model.txt", 1, 2, 1, ukryte, SIGMOID);

	Vector testInput = { 1, 0, 0, 0 , 0, 1, 0, 0, 1};
	Vector oczekiwanyWynik = { 1 }; 

	std::cout << "\n--- PRZED TRENINGIEM ---\n";
	Vector wynikPrzed = mojaSiec.feedForward(testInput);
	std::cout << "Wynik sieci: " << wynikPrzed[0] << " (Szef oczekuje: " << oczekiwanyWynik[0] << ")\n";

	std::cout << "\nRozpoczynam bolesny proces nauki (10 000 powtorzen)...\n";

	for (int i = 0; i <= 100000; i++)
	{
		mojaSiec.train(testInput, oczekiwanyWynik);

		if (i % 1000 == 0)
		{
			Vector podglad = mojaSiec.feedForward(testInput);
			std::cout << "Epoka " << i << " -> Aktualny wynik: " << podglad[0] << "\n";
		}
	}

	std::cout << "\n--- PO TRENINGU ---\n";
	Vector wynikPo = mojaSiec.feedForward(testInput);
	std::cout << "Wynik sieci: " << wynikPo[0] << " (Szef oczekuje: " << oczekiwanyWynik[0] << ")\n";

	return 0;
}
