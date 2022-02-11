Základní NN sítě

Před samotným použitím sítě je nutné doinstalovat dva nástroje. Tento krok se provádí v Terminálu, kam se vypíšou příkazy:

pip install -U scikit-learn

pip install matplotlib

Následně je možné importovat všechny moduly a příslušné funkce. 


Co program basic_NN dělá?

Nejprve je pomocí funkce load_digits naimportován dataset MNIST s číslicemi. V dalším kroku je set rozdělen na trénovací a testovací data. 
Následně je uživatel dotazován na zadání volitelných parametrů jako je počet iterací a solver. Pomocí funkce MLPClassifier je použita trénovací část datasetu pro trénování.
Vložená funkce warning zachytává hlášku Continuous Integration a ignoruje ji. Tato hláška upozorňuje na nedostatečný počet iterací pro kvalitní trénování. 
Funkcí mlp.predict síť predikuje o jakou číslici by se mělo jednat. Následně se připravuje chybová matice, která je jako výstup vytisknuta a uvádí, kde došlo k záměně.


Co program basic_NN_02 dělá?

V této části bylo úkolem vyzkoušet zejména načtení další části datasetu MNIST, k čemuž slouží funkce fatch openml.
Stažení datasetu chvíli trvá (záleží na výkonosti počítače) proto jsou přidány informativní printy: Downloading dataset a Downloading complete.
V dalším kroku je set rozdělen na trénovací a testovací data. Pomocí funkce MLPClassifier je použita trénovací část datasetu pro trénování. Opět je vložena i funkce warning vysvětlená výše. Následně jsou vytisknuty skóre dosažené testováním a trénováním. 
V poslední části jsou vytvořeny podgrafy a zobrazeny váhy se stejným měřítkem.
