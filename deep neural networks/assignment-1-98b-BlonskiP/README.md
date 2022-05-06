# Wielowarstwowa sieć jednokierunkowa MLP w Tensorflow, Część I

## Wstęp:

Celem ćwiczenia jest wprowadzenie do biblioteki Tensorflow, przypomnienie podstawowej architektury sieci MLP oraz wpływu hiperparametrów na uczenie i na jakość otrzymywanych wyników.
Sieć powinna rozwiązywać problem klasyfikacji obrazów ze zbioru stylów architektonicznych.  
Należy zdefiniować architekturę modelu, funkcję celu i dostarczyć dane do sieci. (Tensorflow automatycznie oblicza pochodne funkcji celu).


Zadania na laboratorium będą oparte o interfejs Subclassing API. W notebooku znajduje się treść zadania wraz ze zdefiniowanymi klasami bazowymi i narzędziami pomocniczymi.

Paczki, które mogą być dodatkowo zastosowane:  
- numpy  
- scikit-learn (Metryki)  
- matplotlib, seaborn (Wykresy)  
- tqdm (Pasek postępu)


## Lista zadań (Część I): 
Klasy bazowe potrzebne do realizacji zadania oraz instrukcje pomocnicze zostały przedstawione w pliku [Assignment_1.ipynb](Assignment_1.ipynb)

1. Dokonaj przetwarzania wstępnego zbioru danych (kroki przetwarzania wstępnego zostały opisane poniżej). Następnie wykorzystując klasę bazową BaseLayer zaimplementuj warstwę w pełni połączoną. (Dodawanie zmiennych do warstwy odbywa się za pomocą metody add_weight). Źródło:  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) **\[0.5 pkt\]**.

2. Wykorzystując klasę bazową BaseModel i zaimplementowaną wcześniej warstwę w pełni połączoną, zdefiniuj architekturę sieci, funkcję uczenia sieci i funkcję ewaluacji modelu. Ewaluacja modelu ma obejmować metrykę F1 oraz logowanie wartości funkcji straty i ma następować po każdej epoce uczenia.. **\[0.5 pkt\]**.

3. Zwizualizuj otrzymane rezultaty (Można wykorzystać narzędzie TensorBoard https://www.tensorflow.org/tensorboard) **\[1 pkt\]**.:
    - Przedstaw wartości metryki F1 w zależności od epoki (dla zbioru treningowego, walidacyjnego i testowego)
    - Przedstaw krzywą funkcji kosztu w zależności od epoki (dla zbioru treningowego, walidacyjnego i testowego) 
    - Przedstaw macierz pomyłek (confusion matrix) dla zbioru testowego

4. Zwizualizuj kilka przykładów na klasę, dla których model podejmował złą decyzję i przeanalizuj dlaczego **\[1 pkt\]**.

Jakość analizy i realizacji (prawidłowość wniosków, klarowność prezentacji, rozumienie modelu, jakość kodu)  **\[2 pkt\]**.

## Architektura i hiperparametry

**Dwu-warstwowa sieć w pełni połączona**
1. Spłaszczenie obrazu do jednego wymiaru (np. z 64x64x3 do 12288). Operacja może być zdefiniowana jako krok przetwarzania wstępnego.
2. Warstwa w pełni połączona z 256 neuronami i funkcją aktywacji ReLU
3. Warstwa w pełni połączona z 14 neuronami i funkcją aktywacji Softmax

Inicjalizacja wag: **uniform** *(Funkcje aktywacji znajdują się w module `tensorflow.keras.activation`)*

**Hiperparametry uczenia**

- Wielkość paczki: 100
- Optymalizator: Adam
- Współczynnik uczenia: 0.001
- Liczba epok: 25
- Funkcja kosztu: tf.keras.losses.SparseCategoricalCrossentropy


### Ograniczenia

1. Zadania 1 i 2 muszą być oddane razem z Zadaniem 3
2. Wykorzystanie gotowych modułów implementujących warstwy sieci np. `tensorflow.keras.layers` i innych jest **zabronione**!


## Zbiór danych
Zbiór danych został udostępniony pod adresem https://dnn-lab-pwr.s3.amazonaws.com/dataset.pkl?AWSAccessKeyId=AKIAQZFHSEACSA724RVF&Signature=C9ed3e890IGjqPOR0qFRWS3JsGg%3D&Expires=1607459449 (2.1 GB) w formacie `pickle` i składa się z 14800 obrazów RGB o wymiarach 224x224 przedstawiających budynki oraz zbioru etykiet zawierającego 14 klas określających styl architektoniczny danego budynku.

Dane zostały podzielone na:
- zbiór treningowy (8870 obrazów)
- zbiór walidacyjny (2964 obrazów)
- zbiór testowy (2966 obrazów)

Nazwy etykiet zostały wcześniej zmapowane do liczb, mapowanie jest także zawarte w pliku.
 
## Przetwarzanie wstępne

W celu realizacji zadania należy przeprowadzić następujące kroki:
- Ustawienie wartości ziarna na 1234 
- Normalizacja wartości pikseli obrazów z przedziału 0-255 do przedziału 0-1
- Wczytanie mechanizmu paczkowania dla zbioru treningowego

**Opcjonalnie, w zależności od zasobów obliczeniowych** 
- Zmniejszenie obrazów z rozmiaru 224x244 do 64x64 lub innego wybranego.
