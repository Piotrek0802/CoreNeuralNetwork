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
	Vector bias, obliczenia, poAktywacji;
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
		if (rodzAktywacji==SIGMOID)
		{
			randDlaSigmoidy(wagi);
		}
	}

	~Layer()
	{
	}

	Vector feedForward(Vector& input) {
		for (int i = 0; i < outputSize; i++)
		{
			double wynik = 0;
			for (size_t j = 0; j < inputSize; j++)
			{
				wynik += input[j] * wagi[i][j];
			}
			wynik += bias[i];
			obliczenia[i] = wynik;
			if (rodzAktywacji==SIGMOID)
			{
				poAktywacji[i] = sigmoid(wynik);
			}
		}
		return poAktywacji;
	}

private:
	
};
class Network
{
public:
	const std::string nazwaPliku;
	const int ileHiddenLayers, inSize, outSize;
	const std::vector<int>ileNeuronsInLayers;
	rodzajAktywacji rodzAktywacji;
	std::vector<Layer> layers;
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
	~Network(){}

	Vector feedForward(const Vector& input) {
		Vector tempInput = input;
		for (size_t i = 0; i <= ileHiddenLayers; i++)
		{
			tempInput = layers[i].feedForward(tempInput);
		}
		return tempInput;
	}
	void zapiszDane() {
		std::fstream plik;
		plik.open(nazwaPliku, std::ios::out);
		if (plik.good()!=true||plik.is_open()!=true)
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
				plik << layers[i].bias[j]<<" ";
			}
			plik << "\n";
		}
	}
private:

};


int main()
{
	std::cout << "Budowanie sieci...\n";

	// Tworzymy rozmiary warstw ukrytych (np. jedna warstwa 64 neurony)
	std::vector<int> ukryte = { 64 };

	// Tworzymy sieć: plik, 1 ukryta, wejście 784, wyjście 10, wektor ukrytych, SIGMOID
	Network mojaSiec("moj_model.txt", 1, 784, 10, ukryte, SIGMOID);

	std::cout << "Siec zbudowana! Liczba warstw (Layer): " << mojaSiec.layers.size() << "\n";

	// Tworzymy testowy wektor wejściowy (784 jedynki)
	Vector testInput(784, 1.0);

	std::cout << "Przepuszczam dane przez siec...\n";
	Vector wynik = mojaSiec.feedForward(testInput);

	// Wypisujemy wynik (powinno być 10 liczb z zakresu 0-1)
	std::cout << "Wynik sieci:\n";
	for (double w : wynik) {
		std::cout << w << "\n";
	}
}
