# Sieci rekurencyjne, Część 2
## Wstęp:

 W drugiej części przebadamy publicznie dostępne metody pre-trenowanej reprezentacji tekstu oraz popularne rozwinięcie standardowego LSTM-a - dwukierunkowe przetwarzanie sekwencji.

## Lista zadań: 

Dodaj poniższe do zaimplementowanego wcześniej modelu sieci rekurencyjnej:

1. Opcję wykorzystania 3 różnych publicznie dostępnych, pretrenowanych modeli wektorowej reprezentacji tekstów, zastępujących bądź wczytywanych do warstwy word embedding. (2 pkt) 

Dla zainteresowanych dopuszczalne jest zastosowanie modeli opartych o architekturę transformer (w takim przypadku można podzielić dokumenty na krótsze sekwencje o wyznaczonej długości).

2. Dwukierunkowość w warstwie LSTM. Warstwa dwukierunkowa powinna przetwarzać sekwencje w kolejności standardowej oraz odwróconej i konkatenować otrzymane wyniki. (2 pkt)

3. Przebadaj zyski z zaimplementowanych usprawnień i opisz otrzymane wyniki w sprawozdaniu. (1pkt)
