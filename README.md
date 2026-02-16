# 🧠 SkimLit — Medical Abstract Sentence Classification (NLP)

SkimLit is an NLP project that classifies each sentence in a medical abstract into its rhetorical role (e.g., **BACKGROUND**, **METHODS**, **RESULTS**, **CONCLUSIONS**).  
The goal is to make reading biomedical literature faster by automatically **segmenting abstracts into structured sections**.

This repo replicates ideas from the **PubMed RCT** benchmark and progressively improves models from a strong baseline to a **token + character + positional (tribrid) architecture**.

---

## 🔥 What This Project Demonstrates

- End-to-end NLP pipeline: parsing raw PubMed RCT text → structured samples → training datasets
- Multiple model iterations with measurable comparisons (baseline → deep models)
- Efficient input pipelines using **tf.data** (`batch`, `prefetch(AUTOTUNE)`)
- Embeddings at different granularities:
  - **Token embeddings** (TextVectorization + Embedding)
  - **Pretrained sentence embeddings** (Universal Sentence Encoder)
  - **Character embeddings** (TextVectorization + Conv1D / BiLSTM)
- Feature engineering for structured text:
  - **Line number** and **total lines** positional features
- Regularization & generalization:
  - Dropout, label smoothing, checkpointing
- Model saving/loading (SavedModel format)

---

## 🎯 Task

Given an abstract broken into sentences:

> “Emotional eating is associated with overeating…”  
> “We conducted a randomized trial…”  
> “Participants improved…”  

Predict a label per sentence such as:

- **BACKGROUND**
- **OBJECTIVE**
- **METHODS**
- **RESULTS**
- **CONCLUSIONS**

This enables automatic abstract structuring and faster comprehension.

---

## 📚 Dataset

Uses the **PubMed 20k RCT** dataset from Franck Dernoncourt et al.  
Dataset: `PubMed_20k_RCT_numbers_replaced_with_at_sign`

Each sentence is labeled with its section role and grouped by abstract.

Repository source (dataset):  
- https://github.com/Franck-Dernoncourt/pubmed-rct

---

## 🧪 Modeling Approach

The notebook runs a sequence of experiments:

### Model 0 — Baseline (Strong Starting Point)
- **TF-IDF + Multinomial Naive Bayes**
- Quick sanity check baseline for sentence classification

### Model 1 — Token Embeddings + Conv1D
- TextVectorization → trainable Embedding → Conv1D classifier

### Model 2 — Pretrained Embeddings (Feature Extraction)
- TensorFlow Hub **Universal Sentence Encoder**
- Dense classifier on top (faster convergence, strong semantics)

### Model 3 — Character Embeddings
- Character-level vectorizer + Embedding + Conv1D
- More robust to domain-specific tokens, abbreviations, formatting

### Model 4 — Hybrid Token + Character Model
- Combines pretrained token embeddings + char BiLSTM
- Concatenates representations before classification

### Model 5 — Tribrid Model (Best)
- **Token embeddings + Character embeddings + Positional features**
- Adds:
  - One-hot **line_number**
  - One-hot **total_lines**
- Includes **label smoothing** + dropout for generalization
- Saves best model as `skimlit_tribrid_model`

---

## ⚙️ Training Configuration

Typical configuration used in the notebook:

- Batch size: `32`
- Optimizer: `Adam`
- Loss: `CategoricalCrossentropy` (with optional label smoothing)
- Input pipeline: `tf.data.Dataset.batch(32).prefetch(AUTOTUNE)`
- Training strategy: quick iterations (often training on 10% steps for fast experimentation)

---

## 📈 Results

The notebook computes **precision / recall / F1-score / accuracy** on the validation set and compares all models in a single table + bar chart.

- Baseline provides a solid reference point
- Deep models improve performance by learning richer text representations
- Best performance comes from combining:
  - semantics (token embeddings)
  - robustness (char embeddings)
  - structure (positional features)

---
 
## 🧩 Future Improvements

- Add full test-set evaluation and report final metrics in the README  
- Add a confusion matrix + most-confused classes  
- Perform error analysis: inspect the “most wrong” predictions (high-confidence mistakes)  
- Add a simple inference demo: paste a raw abstract → output labeled sentences  
- Export for deployment (TFLite / TensorFlow Serving / FastAPI)  

---

## 📌 Reference Papers

- Dataset paper (PubMed RCT): https://arxiv.org/abs/1710.06071  
- Architecture inspiration (token + char + positional features): https://arxiv.org/abs/1612.05251  


