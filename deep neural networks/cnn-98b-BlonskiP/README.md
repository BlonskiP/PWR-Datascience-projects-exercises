# Konwolucyjna sieć neuronowa, Cześć I

## Wstęp

Poznanie możliwości sieci konwolucyjnej (CNN) w zadaniu klasyfikacji obrazów w bibliotece Tensorflow. Sieć powinna rozwiązywać problem klasyfikacji obrazów ze zbioru stylów architektonicznych.

## Lista zadań (Część I)

**1. Zaimplementuj zdefiniowaną poniżej architekturę sieci konwolucyjnej \[2 pkt\]**

| Blok | Warstwy | Liczba filtrów / Liczba neuronów | Wymiar filtra | Liczba kanałów wejściowych | Liczba kanałów wyjściowych | Przesunięcie / krok (ang. stride) |
| --- | --- | --- | --- | --- | --- | --- |
| conv1 | Konwolucja <br /> ReLU <br /> Max-pool | 64 | [5,5] <br /> - <br /> [3,3]| 3 <br /> - <br /> - | 64 <br /> - <br /> - | [1, 1] <br /> - <br /> [2, 2] |
| conv2 | Konwolucja <br /> ReLU <br /> Max-pool | 64 | [5,5] <br /> - <br /> [3,3]| 64 <br /> - <br /> - | 64 <br /> - <br /> - | [1, 1] <br /> - <br /> [2, 2] |
| flatten | Spłaszczenie | - | - | - | - | 
| fc1 | Warstwa w pełni połączona <br /> ReLU | 384 | - | - | - | - |
| fc2 | Warstwa w pełni połączona  <br /> ReLU | 192 | - | - | - | - |
| fc3 | Warstwa w pełni połączona  <br /> Softmax | 14 | - | - | - | - |

**2. Dobierz odpowiednie hiperparametry uczenia modelu oraz przetestuj różne parametry sieci konwolucyjnej (rozmiar filtra, zastosowanie innej metody pooling). \[2 pkt\]**  
Dla każdego przetestowanego modelu przedstaw wartość metryki f1 oraz dokonaj porównania tych modeli. 
Proszę o przetestowanie minimum **czterech** konfiguracji, gdzie co najmniej **jedna** z nich osiąga wartość **0.35** metryki F1 na zbiorze walidacyjnym. 

*W celu przyśpieszenia obliczeń można zmniejszyć rozmiar danych wejściowych i/lub wykorzystać platformę Google Colab https://colab.research.google.com/*

**3. Dokonaj ewaluacji znalezionej najlepszej konfiguracji hiperparametrów na zbiorze testowym i przedstaw \[1 pkt\]**:
- wykres zależności f1 od numeru epoki 
- krzywą funkcji kosztu w zależności od numeru epoki
- macierz pomyłek (confusion matrix)
- wizualizację kilku przykładów na klasę, dla których model podejmowal złą decyzję (przeanalizuj dlaczego model mógł źle sklasyfikować*)

**4. Dla ambitnych (niepunktowane)**  
Wykorzystaj następujące narzędzia w celu interpretacji decyzji sieci:
- [What-If Tool](https://pair-code.github.io/what-if-tool/index.html#features) - nakładka do Tensorboarda. Na stronie znajduje się zakładka `Smile Detection`, zawierająca wizualizację dla problemu klasyfikacji.
- `tf-explain` ([link](https://github.com/sicara/tf-explain)) - dodatek do Tensorflowa, implementujący jedne z ostatnich metod w dziedzinie wyjaśnialnej sztucznej inteligencji (explainable AI lub XAI). Metody pozwalają zwizualizować, jakie obszary obrazu najbardziej przyczyniły się do podjęcia decyzji. Ich dokładny opis znajduje się w publikacjach, można je znaleźć po nazwach metod.
