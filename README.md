<h1 align="center">Multimodalny RAG dla dokumentacji technicznej maszyn</h1>

<p align="center">
  Lokalny system RAG do wyszukiwania informacji w dokumentacji technicznej PDF.
</p>

<p align="center">
  <b>Docling</b> · <b>Milvus</b> · <b>Ollama</b> · <b>LangChain</b> · <b>Gradio</b>
</p>

---

## 1. Opis projektu

Projekt jest lokalnym systemem RAG, czyli Retrieval-Augmented Generation, przygotowanym do pracy z dokumentacją techniczną maszyn, pojazdów, podzespołów i systemów górniczych.

System umożliwia zadawanie pytań w języku naturalnym do wielu plików PDF. Dokumenty są wcześniej przetwarzane, dzielone na chunki, wzbogacane o metadane, indeksowane w bazie wektorowej Milvus, a następnie wykorzystywane jako kontekst dla lokalnego modelu językowego uruchamianego przez Ollama.

Projekt obsługuje nie tylko zwykły tekst, ale również tabele, procedury, dane techniczne, ostrzeżenia, rysunki, schematy i opisy wizualne stron PDF. Dzięki temu nadaje się do pracy z dokumentacją techniczną, instrukcjami obsługi, DTR, katalogami części, opisami układów oraz procedurami serwisowymi.

---

## 2. Główne funkcje

- lokalne działanie bez wysyłania dokumentów do zewnętrznych usług,
- indeksowanie dokumentów PDF z folderu `doc_sources`,
- automatyczne rozpoznawanie typu treści: tekst, tabela, procedura, obraz, mixed,
- obsługa OCR przez Docling,
- tworzenie chunków tekstowych z metadanymi,
- tworzenie dodatkowych chunków `image_context` dla stron ze schematami i rysunkami,
- opisywanie wybranych stron PDF przez lokalny model vision z Ollama,
- zapis dokumentów w Milvus jako dense vector + sparse BM25,
- wyszukiwanie semantyczne i keyword/metadata,
- filtrowanie po typie maszyny i modelu,
- ranking wyników według intencji pytania,
- rozszerzanie kontekstu o sąsiednie strony/chunki,
- generowanie odpowiedzi przez lokalny model LLM,
- zwracanie źródeł odpowiedzi,
- interfejs webowy w Gradio,
- wizualizacja bazy wektorowej do pliku PNG.

---

## 3. Architektura działania

```text
PDF z doc_sources
        |
        v
Docling + OCR
        |
        v
Dzielenie dokumentów na chunki
        |
        v
Czyszczenie tekstu i poprawki OCR
        |
        v
Dodanie metadanych:
- nazwa pliku
- strona
- sekcja
- typ treści
- typ maszyny
- model maszyny
- namespace
        |
        v
Opcjonalne opisy stron wizualnych przez Ollama Vision
        |
        v
Milvus Vector Store
- dense vectors
- sparse BM25
- metadane
        |
        v
Pytanie użytkownika
        |
        v
Analiza intencji pytania
        |
        v
Wyszukiwanie:
- semantyczne
- keyword
- metadata
- filtry maszyny/modelu
        |
        v
Ranking i rozszerzenie kontekstu
        |
        v
Lokalny model LLM przez Ollama
        |
        v
Odpowiedź + źródła
```

---

## 4. Technologie

| Element | Rola w projekcie |
|---|---|
| Python 3.12 | główny język projektu |
| Docling | ekstrakcja treści z PDF, OCR i przygotowanie dokumentu |
| LangChain | integracja dokumentów, embeddingów i modelu LLM |
| Milvus | baza wektorowa dla chunków dokumentacji |
| BM25BuiltInFunction | wyszukiwanie sparse/keyword w Milvus |
| Ollama | lokalne modele LLM, vision i embedding |
| Gradio | interfejs użytkownika |
| Docker Compose | uruchamianie Milvus, MinIO i etcd |
| Matplotlib / UMAP / t-SNE / PCA | wizualizacja przestrzeni wektorowej |

---

## 5. Struktura projektu

```text
RAG-MULTIMODAL/
│
├── .venv/
│   └── lokalne środowisko wirtualne Python
│
├── doc_sources/
│   ├── 01.OPIS TECHNICZNY.pdf
│   ├── 02. INSTRUKCJA OBSŁUGI.pdf
│   ├── 02B. Instrukcja użytkowania cz. V (monitoring standardowy).pdf
│   ├── 10. Most 113.pdf
│   ├── 11. SILNIK QSB 4 5.pdf
│   ├── 17.HOLOWANIE.pdf
│   ├── 20. DCL katalizatory MINE-x opis - czyszczenie.pdf
│   └── 22a. DTR Sybet Proxima SWOI 1.7.pdf
│
├── hf_models/
│   └── lokalny tokenizer używany przy chunkowaniu
│
├── src/
│   ├── config.py
│   ├── index.py
│   ├── milvus_store.py
│   └── rag.py
│
├── video/
│   └── materiały demonstracyjne / nagrania projektu
│
├── volumes/
│   └── dane kontenerów Milvus, MinIO i etcd
│
├── .gitignore
├── .gitmodules
├── config.yaml
├── docker-compose.yml
├── images.png
├── kghm_logo.png
├── machine_assigments.txt
├── pyproject.toml
├── README.md
├── run_ui.py
└── vector_store_visualization.py
```

---

## 6. Najważniejsze pliki

### `config.yaml`

Główny plik konfiguracyjny projektu.

Zawiera ustawienia modeli, dokumentów, bazy Milvus oraz parametrów wyszukiwania.

Najważniejsze wartości:

```yaml
model:
  provider: ollama
  text_generation: gemma4:latest
  vision: gemma4:latest
  embeddings: nomic-embed-text
  tokenizer: hf_models/all-MiniLM-L6-v2
  ollama_base_url: http://localhost:11434
  num_ctx: 8192
  num_predict: 1400

document:
  doc_dir: doc_sources
  supported_file_types:
    - .pdf
  max_tokens: 512
  visual_model: gemma4:latest
  visual_max_pages: 30
  visual_min_score: 90
  visual_dpi: 120
  visual_timeout: 120
  visual_num_predict: 260

database:
  uri: http://localhost:19530
  name: rag_multimodal
  collection_name: collection_universal
  namespace: CaseDoneDemo
  index_batch_size: 128

retrieval:
  k: 8
```

---

### `src/config.py`

Plik odpowiada za ładowanie konfiguracji z `config.yaml`.

Dodatkowo ustawia opcje przetwarzania PDF dla Docling:

- włączony OCR,
- wyłączone zewnętrzne usługi,
- lokalne przetwarzanie dokumentów,
- ustawiona skala renderowania obrazów.

---

### `src/index.py`

Najważniejszy plik odpowiedzialny za indeksowanie dokumentów.

Wykonuje:

- odczyt plików PDF z `doc_sources`,
- konwersję dokumentów przez Docling,
- chunkowanie dokumentów,
- czyszczenie tekstu,
- poprawki typowych błędów OCR,
- wykrywanie numeru strony,
- wykrywanie sekcji,
- klasyfikację typu treści,
- oznaczanie tabel, procedur i obrazów,
- przypisywanie dokumentu do typu i modelu maszyny,
- wykrywanie pustych lub mało wartościowych chunków,
- wybór stron wizualnych,
- generowanie opisów wizualnych przez Ollama,
- tworzenie chunków `image_context`,
- wysyłanie dokumentów do Milvus.

---

### `src/milvus_store.py`

Warstwa komunikacji z bazą Milvus.

Odpowiada za:

- połączenie z Milvus,
- utworzenie bazy danych,
- utworzenie kolekcji,
- konfigurację vector store,
- obsługę embeddingów Ollama,
- zapis dokumentów,
- wyszukiwanie podobnych fragmentów,
- filtrowanie po namespace, typie maszyny i modelu.

Projekt używa dwóch typów reprezentacji:

```text
dense  - embedding semantyczny
sparse - BM25 / keyword search
```

---

### `src/rag.py`

Główna logika RAG.

Odpowiada za:

- analizę pytania użytkownika,
- rozpoznanie intencji,
- wyszukiwanie keyword,
- wyszukiwanie semantyczne,
- filtrowanie po maszynie i modelu,
- ranking chunków,
- rozszerzanie kontekstu o sąsiednie strony,
- składanie kontekstu dla modelu,
- generowanie odpowiedzi,
- obsługę odpowiedzi tabelarycznych,
- fallback, gdy model zwróci słabą odpowiedź,
- zwracanie źródeł.

Rozpoznawane intencje pytania:

```text
wants_table
wants_procedure
wants_safety
wants_visual
wants_parameters
wants_parts
wants_troubleshooting
```

Dzięki temu system inaczej traktuje pytania o tabele, inaczej o procedury, inaczej o schematy, a jeszcze inaczej o usterki lub parametry.

---

### `run_ui.py`

Interfejs webowy w Gradio.

Zawiera:

- logo KGHM,
- wybór typu maszyny,
- wybór modelu maszyny,
- okno rozmowy z chatbotem,
- przycisk wysyłania wiadomości,
- przycisk ponowienia odpowiedzi,
- animację oczekiwania,
- zwijane źródła odpowiedzi,
- motyw jasny i ciemny,
- własny CSS dopasowany do wyglądu projektu.

Domyślny adres aplikacji:

```text
http://127.0.0.1:7860
```

---

### `vector_store_visualization.py`

Skrypt do wizualizacji bazy wektorowej.

Po uruchomieniu pobiera wektory z Milvus, redukuje je do 2D i zapisuje obraz:

```text
vector_store_visualization.png
```

Obsługiwane metody redukcji wymiarów:

```text
UMAP
t-SNE
PCA
```

Przykłady kolorowania punktów:

```text
machine_model
machine_type
content_type
file_name
```

---

### `docker-compose.yml`

Plik uruchamia środowisko Milvus.

Zawiera kontenery:

```text
etcd
minio
milvus-standalone
```

Porty:

```text
19530 - Milvus
9091  - Milvus healthcheck
9000  - MinIO API
9001  - MinIO Console
```

---

### `pyproject.toml`

Plik opisujący projekt i zależności.

Najważniejsze biblioteki:

```text
docling
docling-core
langchain-core
langchain-docling
langchain-milvus
langchain-ollama
pymilvus
python-dotenv
pyyaml
requests
transformers
pypdfium2
gradio
```

Projekt wymaga Pythona:

```text
>=3.12,<4.0
```

---

## 7. Dokumenty i przypisanie do maszyn

System przypisuje każdy dokument do konkretnego typu i modelu maszyny. Dzięki temu użytkownik może filtrować odpowiedzi w interfejsie.

| Plik PDF | Typ maszyny | Model |
|---|---:|---|
| `01.OPIS TECHNICZNY.pdf` | LK1 | LK1 -109/L |
| `02. INSTRUKCJA OBSŁUGI.pdf` | SWK | SWK -177/L |
| `02B. Instrukcja użytkowania cz. V (monitoring standardowy).pdf` | SWK | SWK -188/L |
| `10. Most 113.pdf` | SWK | SWK -226/L |
| `11. SILNIK QSB 4 5.pdf` | SWS | SWS -068/L |
| `17.HOLOWANIE.pdf` | SWS | SWS -070/L |
| `20. DCL katalizatory MINE-x opis - czyszczenie.pdf` | LK1 | LK1 -109/L |
| `22a. DTR Sybet Proxima SWOI 1.7.pdf` | WOS | WOS -175/L |

Modele dostępne w interfejsie:

```text
LK1
└── LK1 -109/L

SWK
├── SWK -177/L
├── SWK -188/L
└── SWK -226/L

SWS
├── SWS -068/L
└── SWS -070/L

WOS
└── WOS -175/L
```

---

## 8. Instalacja projektu

### 8.1. Wejście do folderu projektu

```bash
cd ~/Projekty/rag-multimodal
```

---

### 8.2. Utworzenie środowiska wirtualnego

```bash
python3.12 -m venv .venv
```

---

### 8.3. Aktywacja środowiska

```bash
source .venv/bin/activate
```

---

### 8.4. Aktualizacja pip

```bash
python -m pip install --upgrade pip
```

---

### 8.5. Instalacja zależności

```bash
pip install -e .
```

Jeżeli projekt nie jest instalowany jako paczka, można też zainstalować zależności bezpośrednio z `pyproject.toml`.

---

## 9. Uruchomienie Ollama

Projekt korzysta z lokalnych modeli przez Ollama.

Wymagane modele według `config.yaml`:

```text
gemma4:latest
nomic-embed-text
```

Pobranie modeli:

```bash
ollama pull gemma4:latest
ollama pull nomic-embed-text
```

Sprawdzenie dostępnych modeli:

```bash
ollama list
```

Sprawdzenie, czy Ollama działa:

```bash
curl http://localhost:11434/api/tags
```

---

## 10. Uruchomienie Milvus

Milvus uruchamiany jest przez Docker Compose.

```bash
docker compose up -d
```

Sprawdzenie kontenerów:

```bash
docker ps
```

Sprawdzenie logów Milvus:

```bash
docker logs milvus-standalone
```

Sprawdzenie healthcheck:

```bash
curl http://localhost:9091/healthz
```

Zatrzymanie kontenerów:

```bash
docker compose down
```

Zatrzymanie kontenerów razem z usunięciem danych:

```bash
docker compose down -v
```

Uwaga: dane Milvus znajdują się w folderze `volumes/`.

---

## 11. Indeksowanie dokumentów

### 11.1. Standardowe indeksowanie PDF

```bash
python -m src.index doc_sources --drop
```

Opcja `--drop` usuwa starą kolekcję i tworzy indeks od nowa.

---

### 11.2. Indeksowanie z opisami obrazów i schematów

```bash
python -m src.index doc_sources --drop --describe-images
```

Ta komenda dodatkowo tworzy chunki `image_context` dla wybranych stron zawierających schematy, rysunki, widoki techniczne lub tabliczki.

---

### 11.3. Indeksowanie bez usuwania starej kolekcji

```bash
python -m src.index doc_sources
```

---

### 11.4. Indeksowanie tylko wybranych stron wizualnych

```bash
python -m src.index doc_sources --drop --describe-images --visual-pages "3,5,10-12"
```

---

### 11.5. Zmiana modelu vision przy indeksowaniu

```bash
python -m src.index doc_sources --drop --describe-images --visual-model gemma4:latest
```

---

## 12. Uruchomienie interfejsu

Po uruchomieniu Milvus i po indeksowaniu dokumentów można uruchomić aplikację:

```bash
python run_ui.py
```

Następnie wejść w przeglądarce:

```text
http://127.0.0.1:7860
```

---

## 13. Przykładowe pytania do systemu

```text
Jak wygląda procedura holowania?
```

```text
Co oznacza HAP?
```

```text
Podaj momenty dokręcania dla śrub M10 z dokumentu Most 113.
```

```text
Co ile należy wymieniać olej w moście napędowym serii 113?
```

```text
Opisz schemat układu paliwowego.
```

```text
Jak zresetować awarię systemu antykolizyjnego?
```

```text
Jakie są parametry ciśnienia w układzie?
```

```text
Gdzie znajdują się uchwyty holownicze?
```

---

## 14. Test zapytania z terminala

Można pominąć interfejs Gradio i zadać pytanie bezpośrednio z terminala.

```bash
python - <<'PY'
from src.rag import UniversalRAG

rag = UniversalRAG()

answer, sources = rag.ask_with_sources(
    "Co ile powinno się robić wymianę oleju w moście napędowym serii 113?",
    k=8,
    machine_type="SWK",
    machine_model="SWK -226/L",
)

print("ODPOWIEDŹ:")
print(answer)

print("\nŹRÓDŁA:")
for source in sources:
    print("-", source)
PY
```

---

## 15. Podejrzenie chunków z bazy Milvus

Przykładowa komenda do podejrzenia chunków z konkretnego typu treści:

```bash
python - <<'PY'
from pymilvus import connections, db, Collection
from src.config import config

uri = config.get("database", "uri", default="http://localhost:19530")
db_name = config.get("database", "name", default="rag_multimodal")
collection_name = config.get("database", "collection_name", default="collection_universal")
namespace = config.get("database", "namespace", default="CaseDoneDemo")

connections.connect(alias="default", uri=uri)
db.using_database(db_name)

collection = Collection(collection_name)
collection.load()

rows = collection.query(
    expr=f'namespace == "{namespace}" and content_type == "image_context"',
    output_fields=[
        "text",
        "file_name",
        "page_no",
        "chunk_id",
        "machine_type",
        "machine_model",
        "content_type",
    ],
    limit=5,
)

for row in rows:
    print("=" * 100)
    print("PLIK:", row.get("file_name"))
    print("STRONA:", row.get("page_no"))
    print("TYP:", row.get("content_type"))
    print("MASZYNA:", row.get("machine_type"), row.get("machine_model"))
    print()
    print((row.get("text") or "")[:3000])
PY
```

---

## 16. Wizualizacja bazy wektorowej

Uruchomienie domyślne:

```bash
python vector_store_visualization.py
```

Wynik:

```text
vector_store_visualization.png
```

Kolorowanie według modelu maszyny:

```bash
python vector_store_visualization.py --color-by machine_model
```

Kolorowanie według typu maszyny:

```bash
python vector_store_visualization.py --color-by machine_type
```

Kolorowanie według typu treści:

```bash
python vector_store_visualization.py --color-by content_type
```

Wizualizacja tylko dla SWK:

```bash
python vector_store_visualization.py --expr 'machine_type == "SWK"'
```

Wymuszenie metody t-SNE:

```bash
python vector_store_visualization.py --method tsne
```

Wymuszenie PCA:

```bash
python vector_store_visualization.py --method pca
```

---

## 17. Metadane chunków

Każdy chunk zapisany do Milvus zawiera zestaw metadanych.

Najważniejsze pola:

| Pole | Znaczenie |
|---|---|
| `source` | ścieżka źródłowa dokumentu |
| `file_name` | nazwa pliku PDF |
| `doc_id` | identyfikator dokumentu |
| `page_no` | numer strony |
| `chunk_id` | numer chunka |
| `namespace` | przestrzeń danych, domyślnie `CaseDoneDemo` |
| `machine_type` | typ maszyny, np. `SWK` |
| `machine_model` | model maszyny, np. `SWK -226/L` |
| `section` | wykryta sekcja dokumentu |
| `content_type` | typ treści |
| `has_image` | informacja, czy chunk dotyczy obrazu |
| `has_table` | informacja, czy chunk zawiera tabelę |
| `is_procedure` | informacja, czy chunk wygląda jak procedura |
| `is_noise` | oznaczenie szumu, np. logo, stopka, pusta strona |

Typy treści:

```text
text
table
image
procedure
mixed
image_context
```

---

## 18. Jak działa wyszukiwanie RAG

Po wpisaniu pytania system wykonuje kilka kroków.

### 18.1. Analiza pytania

System rozpoznaje, czy użytkownik pyta o:

```text
tabelę
procedurę
bezpieczeństwo
schemat / rysunek / obraz
parametry
części
usterkę
```

Przykład:

```text
Podaj momenty dokręcania dla śrub M10
```

System traktuje to jako pytanie tabelaryczne i parametryczne.

---

### 18.2. Wyszukiwanie keyword/metadata

System szuka dokładnych słów, numerów, jednostek, nazw dokumentów i oznaczeń.

Przykładowe elementy:

```text
M10
Nm
Most 113
HAP
DCL
MINE-X
QSB
SWOI 1.7
```

---

### 18.3. Wyszukiwanie semantyczne

Równolegle wykonywane jest wyszukiwanie podobieństwa wektorowego w Milvus.

Dzięki temu system potrafi znaleźć pasujące fragmenty nawet wtedy, gdy pytanie jest zadane innymi słowami niż treść dokumentu.

---

### 18.4. Ranking

Wyniki są punktowane według:

```text
zgodności z pytaniem
zgodności z nazwą dokumentu
obecności liczb i jednostek
typu treści
obecności tabeli
obecności procedury
obecności schematu
zgodności z filtrem maszyny
```

---

### 18.5. Rozszerzenie kontekstu

Dla najlepszych wyników system pobiera również sąsiednie strony lub chunki, szczególnie przy pytaniach o:

```text
procedury
tabele
schematy
rysunki
```

To zmniejsza ryzyko, że odpowiedź zostanie wygenerowana z wyrwanego fragmentu.

---

### 18.6. Generowanie odpowiedzi

Do modelu LLM trafia tylko wybrany kontekst z bazy.

Model ma odpowiadać:

```text
po polsku
technicznie
konkretnie
bez zgadywania
na podstawie kontekstu
z zachowaniem liczb i jednostek
```

Jeżeli informacja nie znajduje się w kontekście, system powinien odpowiedzieć, że nie znalazł jej w dostępnych fragmentach dokumentacji.

---

## 19. Tryb wizualny i chunki `image_context`

Podczas indeksowania z opcją:

```bash
--describe-images
```

system wybiera strony, które prawdopodobnie zawierają wartościowe informacje wizualne.

Brane są pod uwagę między innymi:

```text
rysunki
schematy
widoki silnika
tabliczki znamionowe
schematy przepływu
oznaczenia elementów
strzałki
diagramy
widoki techniczne
```

Następnie strona PDF jest renderowana jako obraz i wysyłana do lokalnego modelu vision w Ollama.

Model tworzy krótki opis techniczny strony, który zostaje zapisany jako osobny chunk:

```text
content_type = image_context
```

Dzięki temu system może odpowiadać na pytania typu:

```text
Co pokazuje rysunek?
Gdzie znajduje się dany element?
Jak przebiega przepływ medium?
Co oznaczają strzałki?
Jak wygląda schemat układu?
```

---

## 20. Interfejs użytkownika

Interfejs Gradio zawiera:

- logo projektu,
- listę typów maszyn,
- listę modeli zależną od typu maszyny,
- chatbot,
- pole wpisywania wiadomości,
- przycisk wysłania,
- przycisk ponowienia ostatniego pytania,
- animację oczekiwania,
- zwijane źródła odpowiedzi.

Filtr maszyny działa przed generowaniem odpowiedzi. Oznacza to, że po wybraniu np. `SWK -226/L` system powinien korzystać tylko z chunków przypisanych do tego modelu.

---

## 21. Typowy workflow pracy

### Pierwsze uruchomienie

```bash
cd ~/Projekty/rag-multimodal
source .venv/bin/activate
docker compose up -d
ollama pull gemma4:latest
ollama pull nomic-embed-text
python -m src.index doc_sources --drop --describe-images
python run_ui.py
```

---

### Kolejne uruchomienie następnego dnia

```bash
cd ~/Projekty/rag-multimodal
source .venv/bin/activate
docker compose up -d
python run_ui.py
```

---

### Przebudowa bazy po zmianach w dokumentach

```bash
cd ~/Projekty/rag-multimodal
source .venv/bin/activate
docker compose up -d
python -m src.index doc_sources --drop --describe-images
```

---

## 22. Przydatne komendy diagnostyczne

Sprawdzenie, czy Milvus działa:

```bash
curl http://localhost:9091/healthz
```

Sprawdzenie kontenerów:

```bash
docker ps
```

Sprawdzenie logów Milvus:

```bash
docker logs milvus-standalone
```

Sprawdzenie modeli Ollama:

```bash
ollama list
```

Sprawdzenie endpointu Ollama:

```bash
curl http://localhost:11434/api/tags
```

Sprawdzenie, czy aplikacja Gradio działa:

```text
http://127.0.0.1:7860
```

---

## 23. Najczęstsze problemy

### Problem: brak połączenia z Milvus

Możliwe przyczyny:

```text
kontenery Docker nie są uruchomione
Milvus jeszcze się startuje
port 19530 jest zajęty
kolekcja nie została utworzona
```

Rozwiązanie:

```bash
docker compose up -d
docker ps
docker logs milvus-standalone
```

---

### Problem: brak kolekcji `collection_universal`

Oznacza to, że dokumenty nie zostały jeszcze zaindeksowane.

Rozwiązanie:

```bash
python -m src.index doc_sources --drop --describe-images
```

---

### Problem: Ollama nie odpowiada

Sprawdź:

```bash
curl http://localhost:11434/api/tags
ollama list
```

Jeżeli model nie jest pobrany:

```bash
ollama pull gemma4:latest
ollama pull nomic-embed-text
```

---

### Problem: brak tokenizerów z `hf_models`

W `config.yaml` ustawiony jest lokalny tokenizer:

```yaml
tokenizer: hf_models/all-MiniLM-L6-v2
```

Jeżeli folder nie istnieje albo jest pusty, należy pobrać tokenizer lub zmienić ścieżkę w konfiguracji.

---

### Problem: odpowiedź nie zawiera źródeł

Możliwe sytuacje:

```text
system nie znalazł wystarczającego kontekstu
model odpowiedział, że nie ma informacji w dokumentacji
pytanie było zbyt ogólne
filtr maszyny/modelu ograniczył wyniki
```

Warto wtedy zmienić filtr maszyny albo zadać pytanie bardziej konkretnie.

---

### Problem: wyszukiwanie nie znajduje rysunków

Należy upewnić się, że indeksowanie było wykonane z opcją:

```bash
--describe-images
```

Bez tej opcji system korzysta głównie z tekstu i OCR, ale nie tworzy dodatkowych opisów wizualnych `image_context`.

---

## 24. Co powinno być commitowane

Do repozytorium powinny trafić:

```text
src/
run_ui.py
vector_store_visualization.py
config.yaml
docker-compose.yml
pyproject.toml
README.md
.gitignore
obrazy używane przez UI, np. logo
```

Nie powinny być commitowane:

```text
.venv/
volumes/
__pycache__/
.cache/
pliki tymczasowe
duże lokalne modele
logi
```

Folder `doc_sources/` zależy od polityki projektu. Jeżeli dokumentacja jest poufna lub firmowa, nie powinna trafiać do publicznego repozytorium.

---

## 25. Podsumowanie

Projekt jest lokalnym multimodalnym systemem RAG dla dokumentacji technicznej PDF.

Najważniejsze elementy projektu:

```text
Docling odpowiada za odczyt i OCR dokumentów.
Index.py przygotowuje chunki i metadane.
Ollama generuje embeddingi, opisy wizualne i odpowiedzi.
Milvus przechowuje dense/sparse vectors oraz metadane.
Rag.py odpowiada za logikę wyszukiwania i ranking.
Run_ui.py udostępnia gotowy interfejs Gradio.
Vector_store_visualization.py pozwala zobaczyć rozkład wektorów.
```

System został zaprojektowany tak, aby można było zadawać pytania techniczne do dokumentacji maszyn i otrzymywać konkretne odpowiedzi z podaniem źródeł.
