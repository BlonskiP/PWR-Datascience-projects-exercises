# Konwolucyjna sieć neuronowa, Cześć II

## Wstęp

Poznanie możliwości sieci konwolucyjnej (CNN) w zadaniu klasyfikacji obrazów w bibliotece Tensorflow. Sieć powinna rozwiązywać problem klasyfikacji obrazów ze zbioru stylów architektonicznych.

## Lista zadań (Część II)

Wykorzystując model zrealizowany w części pierwszej dokonaj próby poprawienia jakości klasyfikacji.

1. Zaimplementuj co najmniej jedną z podanych możliwości (2 pkt)
- Data Augmentation – powiększenie zbioru uczącego przez zastosowanie pewnych transformacji obrazu jak skalowanie przesunięcie, wprowadzenie szumu itp.
- Skip Connections takie jak w sieci ResNet (https://arxiv.org/abs/1512.03385)
- Depthwise Separable Convolutions – takie jak w module incepcji (https://arxiv.org/abs/1610.02357)
- Steerable Convolutions (https://arxiv.org/abs/1612.08498)
- Inne zaproponowane usprawnienie np. (https://arxiv.org/pdf/1911.11907.pdf)

2. Zaimplmentuj co najmniej dwie wybrane metody regularyzacji (2 pkt)  
Przykładowe metody regularyzacji: Dropout, L1, L2 (SGD) / Weight decay (Adam), Batch normalization

**W zadaniach powinny być przebadane następujące metryki F-score (per epoka), Wartość funkcji kosztu (per epoka), Podsumowanie np. F-score (macro)** 
Można zastosować metryki z biblioteki sklearn https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

3. Dla znalezionego najlepszego modelu przedstaw dodatkowo (1 pkt)
- macierz pomyłek (confusion matrix)
- wizualizację kilku przykładów na klasę, dla których model podejmowal złą decyzję (przeanalizuj dlaczego model mógł źle sklasyfikować*)

\* Dla ambitnych (niepunktowane): Wykorzystaj następujące narzędzia w celu interpretacji decyzji sieci:
- [What-If Tool](https://pair-code.github.io/what-if-tool/index.html#features) - nakładka do Tensorboarda. Na stronie znajduje się zakładka `Smile Detection`, zawierająca wizualizację dla problemu klasyfikacji.
- `tf-explain` ([link](https://github.com/sicara/tf-explain)) - dodatek do Tensorflowa, implementujący jedne z ostatnich metod w dziedzinie wyjaśnialnej sztucznej inteligencji (explainable AI lub XAI). Metody pozwalają zwizualizować, jakie obszary obrazu najbardziej przyczyniły się do podjęcia decyzji. Ich dokładny opis znajduje się w publikacjach, można je znaleźć po nazwach metod.
