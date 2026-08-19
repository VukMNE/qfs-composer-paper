## Prompt templates

#### 1. Query-decomposition prompt template

```
Razbij dano uporabniško poizvedbo v več manjših, atomarnih podvprašanj, ...
- Vrni rezultat IZKLJUČNO kot veljaven JSON objekt z ENIM poljem:
  {{"podvprasanja": ["...", "...", "..."]}}

### Primer 1
Uporabniška poizvedba: "Generiraj povzetek o porastu primerov garij v Sloveniji in svetu."
Podvprašanja (JSON objekt):
{{"podvprasanja": [
  "Koliko primerov garij je bilo v Sloveniji lani?",
  "Koliko primerov garij so v Sloveniji obravnavali letos?",
  "Koliko ljudi z garjami je Svetovna zdravstvena organizacija ocenila leta 2017?",
  "Koliko ljudi z garjami je Svetovna zdravstvena organizacija ocenila lani?",
  "Kateri so glavni razlogi za porast garij, navedeni v besedilu?"
]}}

### Primer 2
Uporabniška poizvedba: "Generiraj povzetek o tem, kako se garje širijo in zakaj zdravljenje pogosto spodleti."
Podvprašanja (JSON objekt):
{{"podvprasanja": [
  "Kako se garje prenašajo med člani gospodinjstva?",
  "Katere navade ali napake pri terapiji prispevajo k neuspehu zdravljenja?",
  "Kdo je po besedilu najbolj izpostavljen tveganju za okužbo z garjami?",
  "Kateri simptom je tipičen za garje ponoči?"
]}}

### Primer 3
Uporabniška poizvedba: "Generiraj povzetek navodil za zdravljenje garij."
Podvprašanja (JSON objekt):
{{"podvprasanja": [
  "Kako je treba nanašati kremo na telo pri odraslih?",
  "Kako pogosto in v kakšnih intervalih je treba ponoviti nanos zdravila?",
  "Kaj je treba storiti z oblačili in posteljnino pred in po terapiji?",
  "Na katerih mestih na telesu odraslih so spremembe najpogosteje opazne?"
]}}

### Nova uporabniška poizvedba:
"{query}"

Podvprašanja (JSON objekt z enim poljem "podvprasanja"):
```

#### 2. QFS augmented prompt

```
[USER QUERY]
--------------------------
Besedilo: [SOURCE DOCUMENT] 
--------------------------
V nadaljevanju so odgovori na vprašanja, povezana z poizvedbo:\\
Vprašanje: [QUESTION]
Odgovor: [ANSWER]
Vprašanje: [QUESTION]
Odgovor: [ANSWER]
...

Na podlagi zgornjih odgovorov, na kratko in jedrnato povzemite bistvo besedila v povezavi z začetno poizvedbo.
```
