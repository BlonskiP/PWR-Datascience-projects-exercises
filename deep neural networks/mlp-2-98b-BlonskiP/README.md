# Wielowarstwowa sieć jednokierunkowa MLP w Tensorflow, Część II

## Wstęp:

Celem ćwiczenia jest wprowadzenie do biblioteki Tensorflow, przypomnienie podstawowej architektury sieci MLP oraz wpływu hiperparametrów na uczenie i na jakość otrzymywanych wyników. Sieć powinna rozwiązywać problem klasyfikacji obrazów ze zbioru stylów architektonicznych.

## Lista zadań (Część II): 
**1. Wykorzystując model przygotowany w części I, wykonaj badania następujących hiperparametrów modelu (1 pkt)**:  

| Parametr | Punktacja |
| --- | --- |
| A. Metody inicjalizacji wag i biasów | 0.5 pkt|
| B. Liczbę warstw modelu | 0.5 pkt |

wykorzystując następujące hiperparametry uczenia:

| Hiperparametr | Wartość |
| --- | --- |
| Optymalizator | Adam |
| Współczynnik uczenia | 0.001 |
| Liczba epok | 25 |
| Liczność paczki | 100 |

W badaniu powinny być przebadane następujące metryki:
**Accuracy, F-score (średnia), Wartość funkcji kosztu**

**2.  Wykorzystując znalezioną najlepszą konfigurację liczby warstw i metody inicjalizacji, przebadaj hiperparametry uczenia modelu (2 pkt)**

| Hiperparametr | Punktacja |
| --- | --- |
| A. Liczbę epok uczenia | 0.5 pkt |
| B. Metody optymalizacji modelu wraz z przebadaniem różnych wartości współczynnika uczenia | 1 pkt |
| C. Różne liczności paczek (mini batch) | 0.5 pkt | 

Hiperparametry uczenia powinny być badane łącznie tj. w rezultacie otrzymujemy iloczyn kartezjański hiperpametrów.

W badaniu powinny być przebadane następujące metryki
**Accuracy, F-score (średnia), Wartość funkcji kosztu, Czas uczenia modelu (per epoka i końcowy)**

**3. Przedstaw podsumowanie badań z punktów 1. i 2. wraz z wnioskami. (1 pkt)**

**4. Dla znalezionego najlepszego zestawu hiperparametrów przedstaw (1 pkt)**:
- wykres zależności accuracy od numeru epoki 
- krzywą funkcji kosztu w zależności od numeru epoki
- macierz pomyłek (confusion matrix)
- wizualizację kilku przykładów na klasę, dla których model podejmowal złą decyzję (przeanalizuj dlaczego model mógł źle sklasyfikować)

**5\* Dla ambitnych (niepunktowane): Wykorzystaj następujące narzędzia w celu interpretacji decyzji sieci**:
- [What-If Tool](https://pair-code.github.io/what-if-tool/index.html#features) - nakładka do Tensorboarda. Na stronie znajduje się zakładka `Smile Detection`, zawierająca wizualizację dla problemu klasyfikacji.
- `tf-explain` ([link](https://github.com/sicara/tf-explain)) - dodatek do Tensorflowa, implementujący jedne z ostatnich metod w dziedzinie wyjaśnialnej sztucznej inteligencji (explainable AI lub XAI). Metody pozwalają zwizualizować, jakie obszary obrazu najbardziej przyczyniły się do podjęcia decyzji. Ich dokładny opis znajduje się w publikacjach, można je znaleźć po nazwach metod.

***Uwaga***
Dla każdego z badananych parametrów (1A, 1B) i hiperparametrów (2A, 2B, 2C) powinny być przebadane przynajmniej dwie wartości lub metody.
W przypadku punktu 2B należy przebadać przynajmniej dwie metody optymalizacji oraz dla każdej metody przynajmniej dwa współczynniki uczenia.
