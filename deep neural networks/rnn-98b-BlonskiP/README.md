# Sieci rekurencyjne, Część I

## Wstęp:

 Ćwiczenie jest wprowadzeniem do zastosowania sieci rekurencyjnych w analizie języka naturalnego. W ramach zadania należy przebadać wyniki w zadaniu klasyfikacji tematycznej tekstów na zbiorze artykułów Reuters. Uwaga - zbiór jest niezbalansowany, więc dla łatwiejszej interpretacji wyników można ograniczyć analizę do 10 najczęstszych klas.
 
 Zbiór dostępny jest w module tf.keras.datasets
 
 https://keras.io/api/datasets/reuters/
 
 W pierwszej części zadania należy zaimplementować i przebadać działanie sieci rekurencyjnej na danych zakodowanych za pomocą uczonej razem z siecią warstwy word embedding.


## Lista zadań: 

 1. Zaimplementuj w Tensorflow warstwę word embedding. Warstwa ta powinna przekształcać dodatnie liczby całkowite reprezentujące indeks słowa w słowniku na wektory rzeczywiste. (1pkt)
 
 2. Zaimplementuj w Tensorflow sieć rekurencyjną RNN i LSTM. W implementacji dozwolone jest wykorzystanie operacji w tf.keras.backend, ale nie gotowych warstw. (2 pkt)
 
 Podpowiedź: rekurencję można zaimplementować wykorzystując operację tf.keras.backend.rnn lub tf.scan.

 3. Metoda wczytywania zbioru umożliwia ograniczenie słownika do n najczęściej występujących słów. Sprawdź wyniki achitetury o postaci: 
 - Embedding 
 - LSTM 
 - warstwa wyjściowa 
 
 dla róźnych wartości n. (1pkt)
 
 4. Porównaj wynik sieci LSTM z prostą siecią rekurencyjną. Oceń zysk z zastosowania komórki LSTM. W porównaniu przedstaw:
   - krzywą funkcji kosztu w zależności od epoki
   - krzywą accuracy w zależności od epoki dla zbioru testowego i uczącego
  (1pkt)
