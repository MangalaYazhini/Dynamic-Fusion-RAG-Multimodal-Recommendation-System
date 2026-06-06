# Agentic Multimodal RAG with Dynamic Cross-Modal Fusion for E-Commerce Product Recommendation

## Overview

This repository contains the implementation of **DynaFuse-RAG**, an Agentic Multimodal Retrieval-Augmented Generation (RAG) framework for intelligent e-commerce product recommendation.

The system combines visual and textual product information using dynamically computed cross-modal fusion weights, semantic vector retrieval, and Large Language Model (LLM) based grounded recommendation generation. It supports text-only, image-only, and multimodal user queries, and reduces hallucinations through a multi-layer retrieval-grounded pipeline.

The complete implementation is provided as a single Jupyter Notebook containing the full experimental pipeline: data ingestion, embedding generation, vector indexing, dynamic category taxonomy discovery, query understanding, agentic retrieval, recommendation generation, evaluation, and ablation studies.

---

## Research Objective

Traditional recommendation systems often rely on collaborative filtering, content-based filtering, or matrix factorization methods. These approaches typically struggle with:

- Cold-start scenarios
- Limited use of visual information
- Static modality fusion (fixed α = 0.5 for all queries)
- Hallucination in generative recommendations

This project addresses these limitations through three key innovations:

1. **Dynamic Cross-Modal Fusion** — The text/image fusion weight α is computed adaptively per query based on text specificity, visual salience, and LLM visual intent signals, replacing the conventional fixed α = 0.5.
2. **Dynamic Category Taxonomy Discovery** — Product categories are extracted directly from the populated SQLite database at runtime, eliminating hardcoded whitelists and enabling automatic scaling as the catalog changes.
3. **Multi-Layer Anti-Hallucination Pipeline** — Domain gating, similarity thresholding, category cross-validation, irrelevant product filtering, and grounded LLM generation collectively prevent fabricated recommendations across six checkpoints.

---

## Key Features

- Text-only, image-only, and multimodal (text + image) recommendation queries
- OpenCLIP-based multimodal embeddings (ViT-B-32-quickgelu, OpenAI weights)
- BLIP image caption generation with four-stage caption cleaning and quality gating
- Gemini 2.5 Flash powered structured intent extraction (8 intent signals per query)
- Dynamic cross-modal fusion weight computation (adaptive α per query)
- ChromaDB vector retrieval with HNSW index and cosine similarity
- SQLite structured metadata storage with dual Amazon schema support (2018 and 2023)
- Agentic retrieval refinement loop (up to 3 iterations with query reformulation)
- Taxonomy-aware category validation with synonym expansion (49 colloquialisms mapped)
- Out-of-domain rejection with 28 keyword triggers and 55 taxonomy-based blocklist terms
- Recommendation output caching to avoid redundant LLM calls across ablation runs
- Fuzzy product-name matching for evaluation accuracy (RapidFuzz)
- Statistical significance testing (paired t-test, bootstrap confidence intervals)
- Manual annotation validation (20 query-product pairs with Cohen's κ)

---

## System Architecture

The system follows a hybrid offline-online architecture.

### Offline Processing Pipeline

1. **Dataset Ingestion and Dirty-Data Filtering** — Amazon Electronics JSONL parsed with dual-format support (2018 and 2023 schemas); malformed lines handled via literal-evaluation fallback; products with empty, whitespace-only, or placeholder titles (13 predefined terms) and single-token titles under four characters are removed; fewer than 3% of entries are discarded in a representative run.
2. **SQLite Structured Metadata Storage** — Filtered products stored in a ten-column relational database (asin, title, description, features, brand, price, category, average rating, rating count, image URL); used for metadata lookup, taxonomy discovery, keyword baseline search, and post-generation grounding.
3. **Dynamic Category Taxonomy Discovery** — Category engine scans the populated SQLite database using a four-delimiter parser (>, |, /, ,) to extract full hierarchical paths (720 unique paths in a 15,000-product representative run), leaf categories (661 unique), individual terms, and parent taxonomy nodes (180 distinct parents after deduplication).
4. **Multimodal Product Embedding Construction** — Text and image embeddings generated using OpenCLIP ViT-B-32-quickgelu (OpenAI weights); 512-dimensional embeddings; product-side fusion uses a fixed symmetric weight (α = 0.5) as per Equation 1; products are priority-ordered by image availability then by rating count.
5. **ChromaDB Vector Index Construction** — Embeddings stored persistently in ChromaDB using the HNSW graph; cosine similarity used for retrieval (internally stored as cosine distance, 1 − cosine distance reported for interpretability); metadata payload stored alongside each vector to avoid a two-stage I/O lookup.

### Online Query Processing Pipeline

1. Query intake and modality detection
2. BLIP image caption generation (lazy-loaded `Salesforce/blip-image-captioning-base`); caption denoising: prefix removal, consecutive-word deduplication, minimum two unique words required
3. Structured intent extraction using Gemini 2.5 Flash (8 signals: Refined Query, Predicted Budget, Predicted Category, Semantic Preferences, Domain Classification Flag, Domain Note, Query Type, Visual Intent Score); bypassed for pure image-only queries without a usable BLIP caption (7 programmatic intent fields returned instead)
4. Dynamic fusion weight computation (α) based on text specificity, visual salience, and visual intent score
5. ChromaDB vector retrieval
6. Agentic retrieval refinement loop (up to 3 iterations; refines query and re-retrieves when similarity thresholds, category alignment, or budget compliance criteria are not met)
7. Grounded recommendation generation with anti-hallucination rules
8. Post-generation validation

---

## Dynamic Cross-Modal Fusion

The DynamicFusionEngine computes a query-adaptive fusion weight α for combining text and image embeddings:

**e_fused = ( α · e_text + (1 − α) · e_image ) / ‖ α · e_text + (1 − α) · e_image ‖**

α is determined by:

- **Query modality** — text-only (α = 1.0), image-only (α = 0.0), multimodal (base α = 0.5 adjusted dynamically)
- **Text specificity score** — measures how descriptive and specific the text query is
- **Visual salience score** — estimates how informative the image is relative to text
- **LLM visual intent score** — extracted from the Gemini intent response

This is applied only at query time. Product-side indexing uses a fixed symmetric fusion (α = 0.5).

---

## Dataset

The system is evaluated using the Amazon Electronics metadata dataset.

**Dataset source:** https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

Product fields used:

- Product title, description, and features
- Brand name
- Price
- Category hierarchy
- Average rating and review count
- Product image URL

Both Amazon metadata formats are supported:

- **2018 schema** — `asin`, nested categories `[[...]]`, `imUrl`, `brand`, `feature` (singular)
- **2023 schema** — `parent_asin`, flat categories `[...]`, `images` dicts, `store`, `features` (plural)

Up to **15,000 products** are indexed per run.

---

## Technologies Used

### Deep Learning Models

- OpenCLIP (`open_clip_torch`)
  - Model: `ViT-B-32-quickgelu` with OpenAI pretrained weights
  - Embedding dimension: 512
- BLIP Image Captioning (`transformers`)
  - Model: `Salesforce/blip-image-captioning-base`
  - Lazy-loaded on first use

### Retrieval Components

- ChromaDB — persistent vector store
- HNSW (Hierarchical Navigable Small World) — approximate nearest-neighbor index
- Cosine similarity retrieval

### Database

- SQLite — structured metadata storage

### Large Language Model

- Gemini 2.5 Flash (`gemini-2.5-flash`) via `google-genai` SDK

### Supporting Libraries

- `sentence-transformers` — SBERT encoder used in the text-only retrieval baseline (Baseline 3)
- `rapidfuzz` — fuzzy product-name matching for evaluation accuracy
- `scipy` — statistical significance testing (paired t-test, bootstrap confidence intervals)
- `torch`, `numpy`, `Pillow`, `tqdm`, `ftfy`, `regex`

### Programming Language

- Python 3.10+

### Execution Environment

- Google Colab (primary)
- Jupyter Notebook (compatible)

---

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of top-K retrieved results that are relevant |
| Recall@K | Fraction of all relevant items found in top-K |
| NDCG@K | Normalized Discounted Cumulative Gain — rewards relevant items ranked higher |
| MRR | Mean Reciprocal Rank — average reciprocal position of first relevant result |
| Hit Rate@K | Fraction of queries with at least one relevant result in top-K |
| Hallucination Rate | Fraction of responses flagged as out-of-domain or irrelevant |
| Domain Rejection Accuracy | Correct rejection rate for out-of-domain queries |
| Avg Fusion Weight (α) | Mean dynamic α across queries for fusion analysis |
| Irrelevant Product Rate | Fraction of retrieved products filtered as irrelevant |

### Benchmark

A self-constructed benchmark of **144 queries** is used, covering:

- In-domain text queries (with and without budget)
- Out-of-domain queries (15 total, for hallucination and domain rejection testing)
- Image-only queries (visual retrieval)
- Multimodal queries (text + image, for dynamic fusion evaluation)
- Ambiguous and hard in-domain queries (agentic loop stress testing)
- Attribute-specific queries (category filter testing)

### Baselines

Four baselines are implemented and compared against the full system:

| Baseline | Description |
|----------|-------------|
| Baseline 1 — BM25 / Keyword Search | SQLite title keyword search, sorted by rating count |
| Baseline 2 — CLIP-Only | CLIP vector retrieval with no agentic loop, no LLM, no category filtering; fixed α = 0.5 |
| Baseline 3 — Text-Only | SBERT encoder projected into CLIP space via learned least-squares alignment matrix |
| Baseline 4 — Vanilla RAG | CLIP retrieval followed by keyword-overlap re-ranking; no domain gating or agentic loop; fixed α = 0.5 |

### Ablation Studies

Four ablation variants quantify the contribution of key components:

| Ablation System | Component Removed / Modified |
|-----------------|------------------------------|
| No-BLIP | BLIP image captioning disabled |
| Fixed α = 0.3 | Image-heavy fixed fusion weight |
| Fixed α = 0.5 | Balanced fixed fusion weight (conventional baseline) |
| Fixed α = 0.7 | Text-heavy fixed fusion weight |

### Statistical Validation

- Paired t-test across per-query scores between system variants
- Bootstrap confidence intervals (95%) computed on evaluation metrics
- Manual annotation of 20 query-product pairs by a single evaluator blind to automated scores, with Cohen's κ for agreement measurement

---

## Repository Structure

```text
.
├── Dynamic_Fusion_RAG_Multimodal_Recommendation_System.ipynb
└── README.md
```

---

## Prerequisites

Before running the notebook, ensure the following are in place:

1. **Google Colab** (recommended) or a local Jupyter environment with Python 3.10+
2. **Google Drive** mounted at `/content/drive` with the Amazon Electronics dataset accessible at the configured path
3. **Google API Key** configured via Colab `userdata` (key name: `GOOGLE_API_KEY`) for Gemini 2.5 Flash access
4. **Amazon Electronics JSONL dataset** available at the path specified in `SystemConfig.META_PATH`
5. Required Python libraries (installed by the notebook's setup cells):


---

## Running the Notebook

### On Google Colab (Recommended)

1. Upload the notebook to Google Colab.
2. Mount Google Drive and ensure the dataset is available at the configured path.
3. Set the `GOOGLE_API_KEY` secret in Colab (`Tools → Secrets`).
4. Execute all cells sequentially from top to bottom.

### On Local Jupyter

```bash
jupyter notebook Dynamic_Fusion_RAG_Multimodal_Recommendation_System.ipynb
```

Update `SystemConfig.META_PATH`, `SystemConfig.SQLITE_DIR`, and `SystemConfig.CHROMA_DIR` to point to your local data directories before running.

Execute cells sequentially. The notebook contains the complete workflow from dataset ingestion to evaluation and ablation analysis.

> **Note:** Several cells use `google.colab` imports (`userdata`, `drive`). These cells include fallback handling for non-Colab environments, but Google Drive-dependent paths must be updated manually when running locally.

---

## Example Query Types

### Text Query

```text
Best wireless headphones under Rs.3000 with noise cancellation
```

### Image Query

Provide an image URL and the system generates recommendations for similar electronics products using BLIP captioning and CLIP visual embeddings.

### Multimodal Query

```text
Wireless earbuds similar to this with good bass
```

(Supplied alongside a product image URL)

---

## Publication Status

The associated research manuscript is currently under review and has not yet been formally published.

This repository contains the implementation used for the research study.

---

## Authors

**Arjun Paramarthalingam**
Department of Computer Science and Engineering
University College of Engineering Villupuram
Tamil Nadu, India

**Mangala Yazhini E. S.**
Department of Computer Science and Engineering
University College of Engineering Villupuram
Tamil Nadu, India

---

## License

This repository is shared for academic and research purposes.
All rights remain with the authors.
