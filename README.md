# Medical Q&A Assistant: RAG over the Merck Manual

A retrieval-augmented generation (RAG) prototype that answers clinical questions from the *Merck Manual of Diagnosis & Therapy* (19th edition, about 4,100 pages). The answers come from a local Mistral-7B model, and every answer is shown with the PDF pages it was based on.

> **Prototype only.** This is a coursework project. It must not be used to make real clinical decisions: a small model can give wrong answers even when the retrieved text is correct, and the manual edition used is from 2011-2012.

## Business Problem

Healthcare professionals have to search very large volumes of medical literature under time pressure. The goal is a tool that answers questions such as *"What is the protocol for managing sepsis in a critical care unit?"* from a trusted reference, quickly and with a source that can be checked.

## How It Works

```
PDF -> clean pages -> split into chunks -> embed chunks -> vector database (Chroma)
                                                              |
question -> embed question -> 3 nearest chunks -> prompt (context + question) -> Mistral-7B -> answer + sources
```

| Step | What is used |
|---|---|
| PDF loading | PyMuPDF via LangChain |
| Cleaning | removes the licence footer on each page, the table of contents and the alphabetical index |
| Chunking | up to 512 **tokens** (`cl100k_base`) with 20 tokens of overlap |
| Embeddings | `thenlper/gte-large` (1,024 dimensions) |
| Vector database | Chroma, saved to `medical_db/` |
| LLM | Mistral-7B-Instruct v0.2, 6-bit GGUF, run locally with `llama-cpp-python` |
| Evaluation | LLM-as-a-judge for groundedness and relevance (the same Mistral model) |

The notebook compares three ways of answering the same five questions: the plain LLM, the LLM with a prompt that sets a role and a length limit, and RAG. It then tries different generation settings (`temperature`, `top_p`, `top_k`, `max_tokens`) and scores the RAG answers.

## Repository Contents

```
.
├── Full_Code_NLP_RAG_Project_Notebook.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The manual PDF is **not** included. Put your own copy next to the notebook as `medical_diagnosis_manual.pdf`, or change `PDF_PATH` in the settings cell.

## Running the Notebook

1. Open the notebook in **Google Colab with a GPU runtime** (or a machine with an NVIDIA GPU). Set `USE_GPU = False` on a CPU-only runtime; generation will be much slower.
2. Upload the PDF (or mount Google Drive and set `PDF_PATH`).
3. Run the two install cells, restart the runtime, then run the rest from the top.
4. The first run downloads the Mistral model (about 6 GB) and the embedding model, and builds the vector database. Later runs reuse the saved `medical_db/`.

## Notes on the Data

- Every page of the supplied PDF carries a personal licence notice (email address and code). The notebook strips it before chunking so it is not stored in the database or shown in outputs. **Do not commit the PDF, `medical_db/` or notebook outputs from the raw data to a public repository.** The `.gitignore` already excludes the PDF and the database.
- The PDF is a conversion of an e-book, so its table of contents and index are in the text too. They are removed because they match almost any query.

## Findings From the First Run (before the fixes in this version)

These come from the original run on Colab. This version of the notebook has **not** been re-run with the real model, so scores and answers may differ.

- The RAG answers were more specific than the plain LLM's and drew on the manual (for example the antibiotic regimens for sepsis).
- Several answers **ended mid-sentence** because `max_tokens` was only 128. They are now marked when cut off, and the default is 256.
- One answer (leg fracture) told the reader to apply "elastic bandages to layers 2 and 4 of the injured limb", which does not make sense. This shows that a small model can garble correct retrieved text.
- The LLM judge gave relevance scores of 5, 4, 4, 4 and 5. Groundedness was calculated but never displayed. The relevance prompt asked the judge to rate the *context* instead of the answer. Both are fixed.
- The embeddings of the first two chunks were identical, because both were the licence-only pages. Cleaning removes them.

## What Changed From the Original Notebook

- **Cleaning the PDF:** licence footer, table of contents, index and empty pages removed; metadata reduced to the page number and file name.
- **Prompt format:** all prompts now use the `[INST] ... [/INST]` format Mistral-Instruct expects (the original used it only for the judge).
- **Answers show their sources** (PDF pages) and mark answers cut off by the token limit.
- **Errors are no longer returned as answers**; the unused `k` parameter now works.
- **Vector database** is built once and reused. Re-running the original cell stored every chunk twice.
- **Evaluation:** the judge now rates the answer, scores are parsed into a table, and both groundedness and relevance are shown for all five questions. The answer is generated once and then judged, not regenerated.
- **An off-topic question and distance scores** were added to show how the system behaves when the manual has no answer.
- **Cleanup:** one install cell for each group of packages, one import cell, no Colab-only drive mount (now optional), deprecated LangChain classes replaced, a code cell that held plain text (a syntax error) turned into text, repeated per-question cells replaced by loops, and the "Fine-tuning" section renamed (it only changes generation settings).
- **Insights section** rewritten: the original claimed RAG "significantly outperforms" the plain LLM, but no head-to-head measurement was made.

## Limitations

- The source is the 19th edition (2011-2012). Guidance, for example for sepsis, has changed since, so answers may be out of date.
- A 7B model, even with RAG, can add, drop or mix up details. A rating from the same model is not a validation.
- The retriever returns the nearest chunks even for off-topic questions. A distance threshold is a recommended next step.
- Only five questions are evaluated, with no clinician-checked reference answers.
