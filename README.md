# BirrAI
La produzione di birra artigianale ha una storia millenaria, ed è considerata uno dei processi più
antichi al mondo, paragonabile all'invenzione del pane. Nel corso dei secoli, le ricette e le 
conoscenze sono state tramandate, mentre la birra e i suoi processi 
di produzione sono stati oggetto di studio, portando a un continuo sviluppo e miglioramento.
In contemporanea, i progressi nell'intelligenza artificiale hanno aperto nuove prospettive e 
opportunità per l'innovazione nelle ricette di birra e per il monitoraggio e il miglioramento 
dei processi produttivi e della qualità del prodotto finale. 

## Descrizione
Questo progetto consiste in cinque cartelle contenenti diversi moduli per la normalizzazione del dataset di partenza e per la scrittura di algoritmi di diversa
natura per creare nuove ricette di birra.

## Struttura delle Cartelle
Il progetto è strutturato nel seguente modo:

1. `algoritmi`:
   - `rules_based.py`: questo modulo implementa l'applicazione di regole specifiche per combinare gli ingredienti, in base alle caratteristiche della categoria di birra desiderata.
   - `item-based-recommendation.py`: questo modulo, dato l'ID di una ricetta nel database, calcola le ricette più simili in base agli ingredienti e agli attributi, per poi combinarli e creare una nuova ricetta di birra.
   - `algoritmo_genetico.py`: questo modulo implementa un algoritmo genetico per generare nuove ricette basate su un dataset esistente di ricette.
   - `clustering.py`: questo modulo mira ad organizzare e generare ricette in base alle loro caratteristiche utilizzando l'algoritmo di clustering K-means, aprendo la possibilità di scoprire nuove combinazioni.
   - `random_forest.py`: questo modulo implementa un algoritmo random forest per predire il valore di IBU (Unità Internazionali di Amarezza) di una ricetta data.
   - `apriori_&_fp-growth.py`: questo modulo contiene gli algoritmi Apriori e FP-Growth per calcolare gli itemset frequenti e le relative regole di associazione.

2. `dataset`: questa cartella contiene i dataset originali utilizzati nel progetto.

3. `dataset_cleaned`: questa cartella contiene la versione pulita dei dataset dopo la normalizzazione.

4. `dataset_uniti`: questa cartella contiene il dataset ottenuto combinando i cinque dataset originali e il relativo modulo unione_dataset.py.

5. `normalizzazione_dataset`: questa cartella contiene i moduli per la normalizzazione dei dataset originali.
   Ogni modulo è specifico per un dataset e comprende le fasi di data cleaning, feature scaling, feature selection e data balancing.
