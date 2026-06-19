# Cross-Linguistic Lyrics Complexity & Multifractal Sentiment Analysis

## Overview

This repository contains the full implementation and computational pipeline associated with the published paper:

**Cross-Linguistic Complexity, Language-Specific Sentiment: Multifractal Structure and Emotional Valence in Popular Music Lyrics Across Three Languages**

Published in *Computers* (MDPI).

The study investigates how linguistic complexity and emotional valence interact in multilingual song lyrics using sentiment analysis and Multifractal Detrended Fluctuation Analysis (MFDFA).

---

## DOI

📌 https://doi.org/10.3390/computers15050315

---

## Research Objectives

* Quantify emotional valence in lyrics across multiple languages
* Measure linguistic complexity using multifractal analysis
* Compare structural properties across languages
* Investigate relationships between sentiment and complexity
* Identify cross-linguistic patterns in musical text data

---

## Methodology

The computational pipeline includes:

1. Lyrics collection and preprocessing
2. Language detection and filtering
3. Sentiment analysis (language-specific models)
4. Feature extraction
5. Multifractal Detrended Fluctuation Analysis (MFDFA)
6. Statistical comparison across languages
7. Visualization and interpretation

---

## Repository Structure

The repository follows the existing modular script-based architecture.

Each Python script generates outputs in a dedicated folder with the same name:

```text id="repo1"
project/
│
├── *.py                  # Analysis scripts
├── */                   # Output directories (auto-generated)
│
├── requirements.txt
└── README.md
```

---

## Reproducibility

Install dependencies:

```bash id="repo2"
pip install -r requirements.txt
```

Run scripts in the order of the analysis pipeline.

Each script produces:

* Processed datasets
* Statistical outputs
* Figures and visualizations
* Intermediate analysis results

---

## Key Outputs

* Sentiment distribution across languages
* Multifractal spectrum analysis
* Hurst exponent comparisons
* Cross-linguistic statistical tests
* Sentiment–complexity correlation analysis

---

## Data Availability

Due to copyright restrictions on song lyrics, raw textual data is not publicly distributed.

However, all extracted features and processed datasets required for full reproducibility are included.

---

## Citation

If you use this work, please cite:

```bibtex id="repo3"
@Article{computers15050315,
AUTHOR = {Khanipour, Fateme and Shahbazi, Zeinab and Behnamian, Sara and Fogh, Fatemeh and Blood, Nathan},
TITLE = {Cross-Linguistic Complexity and Language-Specific Sentiment: Multifractal Structure and Emotional Valence in Popular Music Lyrics Across Three Languages},
JOURNAL = {Computers},
VOLUME = {15},
YEAR = {2026},
NUMBER = {5},
ARTICLE-NUMBER = {315},
URL = {https://www.mdpi.com/2073-431X/15/5/315},
ISSN = {2073-431X},
ABSTRACT = {We investigate the linguistic complexity and emotional valence of popular song lyrics across English (n=1491), Spanish (n=307), and German (n=225), using an analytical corpus of 2023 tracks drawn from 2113 deduplicated tracks on Spotify’s weekly Top 200 charts (2019–2021). Transformer-based sentiment analysis is combined with complexity-science tools to characterize both the affective content and the structural organization of commercially successful lyrics. A multilingual BERT model reveals a mild negative skew across all three languages (63.7% negative overall); the 1.003-point English–German gap observed under the English-centric VADER lexicon collapses to 0.127 points under BERT, indicating that earlier cross-linguistic sentiment differences are largely measurement artifacts. Word frequency distributions follow Zipf’s law in all three languages (R2>0.96), with English steepest (α=1.409) and German shallowest (α=1.181). Detrended fluctuation analysis indicates persistent long-range correlations (H≈0.66–0.76; none of the 50 shuffled surrogates exceeded the observed values), and multifractal singularity spectra are statistically indistinguishable across languages once corpus size is controlled (all pairwise Mann–Whitney p>0.13). Streaming counts within the Top 200 are concentrated (German Gini =0.556) but, given the truncated single-snapshot sample, are reported as within-chart descriptors rather than population-level scaling.},
DOI = {10.3390/computers15050315}
}

```
📄 Published Paper: https://doi.org/10.3390/computers15050315
---


## License

This project is released under the MIT License.

---

## Contact

For questions or collaboration inquiries, please open an issue in this repository.  
