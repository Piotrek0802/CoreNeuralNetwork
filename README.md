# CoreNeuralNetwork (C++)

Własnoręcznie napisany od zera, lekki silnik sieci neuronowej (Multi-Layer Perceptron) w czystym języku C++. Projekt nie korzysta z żadnych zewnętrznych bibliotek matematycznych ani AI. Cała matematyka – propagacja w przód (feedforward), wsteczna propagacja błędu (backpropagation) i aktualizacja wag – została zaimplementowana od podstaw.

## 🚀 Cechy projektu
* **Header-only:** Cały silnik znajduje się w jednym pliku `NeuralNetwork.h`. Wystarczy wrzucić go do swojego folderu z projektem i załączyć. Nie wymaga skomplikowanego linkowania.
* **Dynamiczna architektura:** Konstruktor pozwala na elastyczne tworzenie dowolnej liczby warstw ukrytych oraz dowolnej liczby neuronów w każdej z warstw.
* **Zapis i odczyt modeli:** Wbudowane funkcje `saveData()` i `loadData()` do bezstratnego zapisywania wyuczonego modelu (wag i biasów z maksymalną precyzją `double`) do pliku tekstowego i błyskawicznego jego wczytywania.
* **Funkcje aktywacji:** Zaimplementowana obsługa nieliniowej funkcji Sigmoid z przygotowaną strukturą enum pod łatwe rozszerzenie o kolejne (np. ReLU, Tanh).

## 🛠️ Jak używać?

Wystarczy pobrać plik `NeuralNetwork.h`, umieścić go w folderze ze swoim kodem źródłowym i zainicjować obiekt sieci.

```cpp
#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

int main() {
    // 1. Konfiguracja architektury: 2 wejścia, 1 warstwa ukryta (8 neuronów), 1 wyjście
    std::vector<int> hiddenLayers = { 8 };
    
    // 2. Tworzenie instancji sieci (plik zapisu, liczba warstw ukrytych, wejścia, wyjścia, struktura, funkcja aktywacji)
    Network myBrain("model.txt", hiddenLayers.size(), 2, 1, hiddenLayers, SIGMOID);

    // Przykładowe wejście i oczekiwane wyjście (np. dla bramki logicznej XOR)
    std::vector<double> input = { 1.0, 0.0 };
    std::vector<double> expectedOutput = { 1.0 };

    // 3. Trening sieci
    myBrain.train(input, expectedOutput);

    // 4. Przewidywanie (feedforward)
    std::vector<double> result = myBrain.feedForward(input);
    std::cout << "Wynik sieci: " << result[0] << std::endl;

    // 5. Zapis wyuczonego modelu na dysk
    myBrain.saveData();

    return 0;
}
```

## 🧪 Przykłady i Testy
Wewnętrzna logika silnika została z sukcesem przetestowana na klasycznych problemach uczenia maszynowego:
* **Bramka XOR:** Weryfikacja zdolności sieci do rozwiązywania problemów nieliniowo separowalnych.
* **Wykrywanie parzystości bitów:** Nauka rozpoznawania wzorców na 10-elementowych ciągach wejściowych.