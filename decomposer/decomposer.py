import os, json, time, math, traceback
from openai import OpenAI
from tqdm import tqdm
import json, re, time





class QueryDecomposer:
    def __init__(self,  model: str = "gpt-4.1-mini"):
        self.client = OpenAI(
            api_key="********")
        self.model = model

    def _build_prompt(self, query: str) -> str:
        template = """
Razbij dano uporabniško poizvedbo v več manjših, atomarnih podvprašanj, ki jih lahko model za odgovarjanje na vprašanja (QA) neposredno odgovori na podlagi podanega besedila.
Zahteve:
- Vsako podvprašanje naj bo jasno, specifično in preverljivo v viru.
- Izogibaj se “da/ne” vprašanj; raje uporabi “kaj/kako/kdaj/kdo/kje/koliko/zakaj”.
- Ne odgovarjaj na vprašanja, samo jih zapiši.
- Podvprašanja naj se ne podvajajo in naj pokrivajo različne vidike prvotne poizvedbe.
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
"""
        # Use .format() to inject the query; all other braces are escaped as {{ }}
        return template.format(query=query)

    def decompose(self, query: str, max_attempts: int = 5, base_delay: float = 1.5):
        import json, time
        use_json_mode = True
        prompt = self._build_prompt(query)

        for attempt in range(1, max_attempts + 1):
            try:
                kwargs = dict(
                    model=self.model,
                    temperature=1.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content
                data = json.loads(text)

                subqs = data.get("podvprasanja")
                if not isinstance(subqs, list) or not all(isinstance(x, str) for x in subqs):
                    raise ValueError("JSON ne vsebuje pravilnega polja 'podvprasanja' (seznam nizov).")

                # Trim + dedupe
                cleaned, seen = [], set()
                for q in subqs:
                    q2 = q.strip()
                    if q2 and q2 not in seen:
                        cleaned.append(q2); seen.add(q2)
                if not cleaned:
                    raise ValueError("Polje 'podvprasanja' je prazno po čiščenju.")
                
                print(f"Decomposition succeeded in {attempt}. poskusu.")
                print("Decomposed Podvprašanja:", cleaned)
                return cleaned

            except TypeError as te:
                if "response_format" in str(te) and use_json_mode:
                    use_json_mode = False
                    continue
                raise
            except Exception as e:
                err = repr(e)
                if attempt < max_attempts and any(code in err for code in ["429", "500", "502", "503", "504"]):
                    time.sleep(base_delay * (2 ** (attempt - 1)))
                    continue
                raise
