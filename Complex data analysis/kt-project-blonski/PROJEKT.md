22.05.2021
Piotr Błoński  

<p align="center">
<b> Wykrywanie potencjalnych komórek Ki67 </b>
</p>

**Wstęp**  
Powodem wyboru tematu jest rozszerzenie badań które zostały wykonane w ramach pracy magisterskiej o nową metodę znajdowania punktów kluczowych.

**Cel**  
Celem jest wypróbowanie nowej metody zaproponowanej przez Dr inż. Maritna Tabakova do znajdowania potencjalnych interesujących punktów przed wykonaniem segmentacji obrazu medycznego i wyuczaniu modelu. 

**Odbiorca**  
Odbiorcami projektu są główni zainteresowani czy ja i Dr inż Martin Tabakov, być może także inni badacze zajmujący się obrazami medycznymi. 

**Wartość dodana**  
Wartością dodatnią będzie potwierdzenie że zaproponowana metoda oparata na elipsach będzie skuteczna w wykrywaniu punktów które mogą oznaczać komórki do zliaczania indeksu Ki67

**Zbiór danych**  
Pierwszym zbiorem danych będzie: SHIDC-B-Ki-67 dostępnym przez kontakt z https://shiraz-hidc.com/ 
W ramach tego projektu zostaną ręcznie oznaczone komórki za pomocą VGG Image Annotator (VIA) który umożliwia zaznaczanie regionów na obrazie za pomocą elips pod różnym kątem,
stworzy to dodatkowy zbiór danych z którego głównie będziemy korzystać w tym projekcie. 

**Realizacja i ewaluacja projektu**  
W ramach sprawdzenia metody zostanie stworzona metryka która określa ile komórek jest w stanie być oznaczonych tą metodą w oparciu o:
* oznaczenia punktów z  SHIDC-B-Ki-67
* ręczne oznaczenia elips z pomoca VIA
W ramach analizy zostaną poddane cechy lokalne elips którymi zostały zaznaczone komórki, ich rozkład po różnych kanałach kolorów.
W ramach ewaluacji oprócz metryk nastąpi porównanie wyników modelu Densenet uczonego na segmentacji opartych o Voronoi, gdzie grupą testową będzie segmentacja na punktach z orginalnego zbioru, a grupą testowaną segmentacja oparta na wykrytych punktach z pomocą elips. Aktualnie grupa testowa osiąga ok 0.92 fscore na zadaniu klasyfikacji segmentu.

**Referencje**  
Negahbani, F., Sabzi, R., Pakniyat Jahromi, B., Firouzabadi, D., Movahedi,F., Kohandel Shirazi, M., Majidi, S., Dehghanian, A.Pathonet introduced as adeep neural network backend for evaluation of ki-67 and tumor-infiltrating lymphocytesin breast cancer.Scientific Reports 11, 1 (Apr 2021)

Huang, G., Liu, Z., van der Maaten, L., Weinberger, K. Q.Denselyconnected convolutional networks, 2018.


