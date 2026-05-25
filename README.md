# PDF2Ontology

A framework for automatically extracting structured knowledge from technical device datasheets (PDF) and populating an OWL ontology (TBox-guided ABox). Given a TBox that defines device classes, properties, and dimension concepts, the framework identifies all devices and device-to-device relationships present in the datasheet, as well as dimension-related information such as height, width, depth, diameter, and mounting dimensions — producing RDF/Turtle output that conforms to the provided schema.

## Extraction Strategies

Two extraction pipelines are implemented, both downstream of a shared visual preprocessing step (see [Dolphin Preprocessing](#dolphin-preprocessing) below).

### Strategy A — Three-Stage Layered Extraction (`stage123_extraction_chain`)

A sequential, modular pipeline that processes different content modalities one at a time:

| Stage | Input | What it does |
|-------|-------|-------------|
| **Stage 1** — Text | Preprocessed text JSON | Reads plain text; discovers device prototypes and their relationships using TBox class/property definitions |
| **Stage 2** — Visual | Stage 1 output + image JSON | Enriches prototypes with visual attributes found in figures and diagrams |
| **Stage 3** — Table | Stage 2 output + table JSON | Reads specification tables; applies *instance fission* — one table row becomes one concrete variant instance |

The final output is a **two-layer knowledge graph**: a *prototype layer* (device subclasses, inter-device relations) and an *instance layer* (concrete ABox instances carrying dimension values with units and orientation tags).

**Best for:** documents with rich tables of variants; cases where you want explainable, stage-wise intermediate results for debugging or evaluation.

### Strategy B — One-Shot Extraction (`one_shot_extraction_chain`)

A single LangChain LCEL pipeline that feeds all PDF page images directly to the LLM in one conversation, instructing it to produce RDF/Turtle triples in one pass.

```
PDF → render images → LLM (TBox schema + all pages) → ABox .ttl → merge with TBox → merged ontology .ttl
```

**Best for:** shorter or less table-dense documents; rapid prototyping; cases where a simpler pipeline is preferred and intermediate JSON is not needed.

---

Both strategies are guided by the **TBox ontology** (`data/ontology/`), which defines the vocabulary the LLM may use. Within this scope, the framework extracts:

- All device entities (e.g., connectors, housings, contacts) and their type assignments
- Device-to-device relationships (e.g., `fitsInto`, `isCompatibleWith`, `hasPart`)
- Dimension information: height, width, depth, diameter, pitch, mounting dimensions — with numeric values, units, and orientation (vertical / horizontal / diameter / other)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended. Key libraries: `langchain-core`, `langchain-openai`, `openai`, `rdflib`, `owlready2`, `pymupdf`, `pydantic`, `fastapi`.

### 2. Set API keys

The extraction chains call OpenAI-compatible models (GPT-4o / GPT-5.x). Set your key as an environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Linux / macOS:**
```bash
export OPENAI_API_KEY="sk-..."
```

Or place it in a `.env` file in the project root (loaded via `python-dotenv`):
```
OPENAI_API_KEY=sk-...
```

### 3. Dolphin preprocessing (brief)

Both chains consume a `*_raw_analysis.json` file produced by the Dolphin document-layout vision model. See [Dolphin Preprocessing](#dolphin-preprocessing) for full deployment details. The short version: a remote RunPod server or a local GPU deployment of Dolphin v1.5 is used to perform layout analysis and assign a reading order to every element on each page; the client script at `src/services/dolphin_runpod_batch_service.py` sends the PDF there and saves the result.

---

## Demo

### Data layout

```
data/
├── ontology/                      # TBox input — put your OWL/RDF file here
│   ├── DeviceDimension_demo.rdf
│   └── DeviceDimension_test.rdf
├── test_materials/                # PDF inputs — put your datasheets here
│   ├── SS_03.pdf
│   ├── EH_01.pdf
│   └── KN_05.pdf
├── test_intermediate_results/     # Dolphin preprocessing outputs (per document)
│   └── {doc_name}/
│       ├── raw_analysis/          # {doc_name}_raw_analysis.json  ← key input
│       ├── text/                  # {doc_name}_text.json
│       ├── images/                # {doc_name}_images.json
│       └── tables/                # {doc_name}_tables.json
└── test_results/
    ├── stage123_extraction/       # Strategy A outputs
    │   └── {doc_name}_{timestamp}/
    │       ├── {doc_name}_stage1_skeleton.json
    │       ├── {doc_name}_stage2_patched.json
    │       ├── {doc_name}_stage3_fissioned.json
    │       └── {doc_name}_performance_metrics.json
    └── one_shot_extraction/       # Strategy B outputs
        └── {doc_name}_{timestamp}/
            ├── {doc_name}_extracted_abox.ttl
            ├── {doc_name}_merged_ontology.ttl
            └── {doc_name}_performance_metrics.json
```

To test with your own datasheet, place the PDF in `data/test_materials/` and run the Dolphin preprocessing step first to generate `{doc_name}_raw_analysis.json` under `data/test_intermediate_results/{doc_name}/raw_analysis/`.

### Running the chains

Both scripts read `OPENAI_API_KEY` from the environment (or a `.env` file in the project root). All paths have sensible defaults so the scripts run out of the box against the bundled sample data.

> **Input requirements differ between the two strategies.**
> Strategy A consumes the preprocessed JSONs produced by the Dolphin preprocessing step (`text/`, `images/`, `tables/` sub-folders) — run the Dolphin client first.
> Strategy B takes the **raw PDF directly** and renders each page at runtime via PyMuPDF — no Dolphin preprocessing required.

**Strategy A — Three-stage extraction:**

```bash
# Minimal — runs on the bundled SS_03 sample
python src/chains/stage123_extraction_chain.py

# Custom document
python src/chains/stage123_extraction_chain.py --doc EH_01

# Full control
python src/chains/stage123_extraction_chain.py \
    --doc       EH_01 \
    --ontology  data/ontology/DeviceDimension_test.rdf \
    --model     gpt-4o-mini \
    --input_dir data/test_intermediate_results/EH_01 \
    --output_dir data/test_results/my_run
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--doc` | `SS_03` | Document name; locates preprocessed JSONs under `data/test_intermediate_results/{doc}/` |
| `--ontology` | `data/ontology/DeviceDimension_demo.rdf` | TBox `.rdf` file that guides extraction |
| `--model` | `gpt-4o` | OpenAI model for all three stages |
| `--input_dir` | `data/test_intermediate_results/{doc}/` | Override directory for `text/`, `images/`, `tables/` sub-folders |
| `--output_dir` | auto-timestamped | Explicit output path; auto-creates `{doc}_{timestamp}/` under `data/test_results/stage123_extraction/` if omitted |

Output is written to `data/test_results/stage123_extraction/{doc}_{timestamp}/`.

**Strategy B — One-shot extraction:**

```bash
# Minimal — runs on the bundled SS_03 sample
python src/chains/one_shot_extraction_chain.py

# Custom PDF
python src/chains/one_shot_extraction_chain.py --pdf data/test_materials/EH_01.pdf

# Full control
python src/chains/one_shot_extraction_chain.py \
    --pdf        data/test_materials/EH_01.pdf \
    --ontology   data/ontology/DeviceDimension_test.rdf \
    --model      gpt-4o \
    --dpi        150 \
    --max_pages  10 \
    --output_dir data/test_results/my_run
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pdf` | `data/test_materials/SS_03.pdf` | Input PDF file |
| `--ontology` | `data/ontology/DeviceDimension_demo.rdf` | TBox `.rdf` file |
| `--model` | `gpt-5.2` | OpenAI model name |
| `--dpi` | `300` | PDF page rendering resolution — lower DPI reduces token usage and cost |
| `--max_pages` | all | Process only the first N pages; useful for quick tests |
| `--output_dir` | auto-timestamped | Explicit output path; auto-creates `{doc}_{timestamp}/` under `data/test_results/one_shot_extraction/` if omitted |
| `--no_validate` | off | Skip TBox class validation after merge |

Output is written to `data/test_results/one_shot_extraction/{doc}_{timestamp}/`.

**Dolphin preprocessing client** (run this before either chain):

```bash
python src/services/dolphin_runpod_batch_service.py \
    --pdf_path data/test_materials/SS_03.pdf \
    --output_dir data/test_intermediate_results/SS_03/raw_analysis \
    --dpi 300 \
    --api_url http://localhost:8080/analyze
```

---

## Dolphin Preprocessing

Both extraction chains are built on top of a `*_raw_analysis.json` file generated by a document-layout vision model. This file captures the structural interpretation of each PDF page — bounding boxes, content types, and reading order — before any LLM-based knowledge extraction takes place.

### Connecting to a remote Dolphin service

The recommended setup uses [RunPod](https://www.runpod.io/) to host a FastAPI server running the Dolphin model. The server code is in `src/services/remote/runpod_server.py`. Access is routed through an SSH tunnel:

```bash
# Forward the RunPod pod port to localhost:8080
ssh -L 8080:<pod-internal-host>:<port> <runpod-ssh-target>
```

Once the tunnel is up, the batch client (see above) sends pages to `http://localhost:8080/analyze` one by one and assembles the results into a single JSON file.

### Local deployment

If your machine has a capable GPU, Dolphin can be run locally. The official model and demo are available at the [Dolphin GitHub repository (v1.5)](https://github.com/ByteDance/Dolphin). The key adaptation points for this project are:

1. **Wrap the demo inference** in a FastAPI endpoint matching the `/analyze` interface expected by `dolphin_runpod_batch_service.py` (single image upload, JSON response per page).
2. **Align the output format** to what the preprocessors in `src/preprocessors/` expect (described below).

### Using an alternative vision model

Dolphin is not strictly required. Any vision model that performs **document layout analysis** and assigns a **reading order** to each detected element can serve as a drop-in replacement, as long as its output is converted to the following JSON format.

**Required `raw_analysis.json` schema:**

```json
{
  "pdf_filename": "SS_03.pdf",
  "total_pages": 12,
  "results_per_page": [
    {
      "page_number": 1,
      "elements": [
        {
          "label": "text",
          "text": "The connector housing accepts up to 8 contacts.",
          "reading_order": 3
        },
        {
          "label": "table",
          "text": "| Height | Width | ... |",
          "reading_order": 7
        },
        {
          "label": "figure",
          "text": "",
          "reading_order": 5
        }
      ]
    }
  ]
}
```

**Field reference:**

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Element type. Recognized values: `text`, `header`, `sec_1`, `list`, `table` / `tab`, `figure` / `fig`, `image` |
| `text` | string | Extracted text content of the element (empty for pure image elements) |
| `reading_order` | integer | Position of this element in the logical reading sequence on the page |

The three preprocessors (`text_preprocessor.py`, `image_preprocessor.py`, `table_preprocessor.py`) split this flat element list by label type and sort by `reading_order` before passing content to the LLM stages. As long as your vision model's output is converted to this schema, it can be integrated directly.
