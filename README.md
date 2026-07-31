# \# A Reproducible Protocol for Evaluating Large Language Models in Mathematical Reasoning

# 

# \[!\[License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)]()

# \[!\[Data: CC BY 4.0](https://img.shields.io/badge/Data-CC--BY--4.0-blue.svg)]()

# \[!\[Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg)]()

# 

# \---

# 

# \## Overview

# 

# This repository accompanies the research article:

# 

# > \*\*A Reproducible Protocol for Evaluating Large Language Models in Mathematical Reasoning Using Authentic Educational Assessment Data\*\*

# 

# The repository contains the complete implementation of the experimental protocol, including:

# 

# \- experimental software;

# \- analysis pipeline;

# \- human validation interface;

# \- experimental outputs;

# \- metadata;

# \- supplementary material;

# \- documentation required to reproduce every analysis reported in the manuscript.

# 

# Unlike many benchmark-oriented studies, this work focuses on the \*\*evaluation protocol itself\*\* rather than on a particular LLM.

# 

# The comparison between GPT models presented in the article should therefore be interpreted as a \*\*proof-of-concept application\*\* demonstrating the protocol.

# 

# \---

# 

# \# Repository contents

# 

# ```

# .

# ├── code/

# │   ├── experiment/

# │   ├── analysis/

# │   └── validation/

# │

# ├── data/

# │   ├── source\_documents/

# │   ├── raw\_outputs/

# │   └── processed\_outputs/

# │

# ├── docs/

# │

# ├── metadata/

# │

# ├── supplementary/

# │

# ├── validation/

# │

# ├── results/

# │

# └── licenses/

# ```

# 

# \---

# 

# \# Scientific contribution

# 

# The repository implements an end-to-end protocol that allows researchers to evaluate mathematical reasoning capabilities of LLMs using authentic examination problems.

# 

# The protocol includes:

# 

# 1\. corpus preparation

# 2\. prompt generation

# 3\. automatic execution

# 4\. response auditing

# 5\. human validation

# 6\. statistical analysis

# 7\. figure generation

# 8\. manuscript tables

# 

# \---

# 

# \# Experimental workflow

# 

# ```

# Official EBAU examinations

# &#x20;         │

# &#x20;         ▼

# PDF extraction

# &#x20;         │

# &#x20;         ▼

# Structured JSON corpus

# &#x20;         │

# &#x20;         ▼

# Experimental execution

# &#x20;         │

# &#x20;         ▼

# Raw model outputs

# &#x20;         │

# &#x20;         ▼

# Automatic auditing

# &#x20;         │

# &#x20;         ▼

# Master dataset

# &#x20;         │

# &#x20;         ▼

# Statistical analysis

# &#x20;         │

# &#x20;         ▼

# Tables and figures

# &#x20;         │

# &#x20;         ▼

# Scientific manuscript

# ```

# 

# \---

# 

# \# Repository structure

# 

# \## code/

# 

# Contains all Python source code.

# 

# \### experiment/

# 

# Scripts used to query the LLM API and generate experimental results.

# 

# \### analysis/

# 

# Complete statistical pipeline.

# 

# Including:

# 

# \- data auditing

# \- master dataset generation

# \- descriptive statistics

# \- inferential statistics

# \- publication figures

# \- publication tables

# 

# \### validation/

# 

# Human validation interface and agreement analysis.

# 

# \---

# 

# \# data/

# 

# Contains:

# 

# \- original educational material

# \- experimental outputs

# \- processed datasets

# 

# The repository distinguishes between:

# 

# \- source documents;

# \- raw experimental outputs;

# \- processed datasets.

# 

# \---

# 

# \# metadata/

# 

# Contains reproducibility information.

# 

# Including:

# 

# \- inventory

# \- checksums

# \- preparation report

# \- security report

# \- dataset description

# 

# \---

# 

# \# supplementary/

# 

# Additional material supporting the manuscript.

# 

# Including:

# 

# \- prompts

# \- evaluation rubric

# \- supplementary tables

# \- methodological notes

# 

# \---

# 

# \# Reproducibility

# 

# The repository has been designed to maximize reproducibility.

# 

# The following elements are included:

# 

# \- source code

# \- experimental configuration

# \- prompts

# \- statistical pipeline

# \- human validation tools

# \- documentation

# \- inventories

# \- integrity hashes

# 

# \---

# 

# \# Software requirements

# 

# Python 3.12+

# 

# Required packages:

# 

# \- pandas

# \- numpy

# \- scipy

# \- matplotlib

# \- openpyxl

# \- tabulate

# \- python-docx

# 

# See:

# 

# requirements.txt

# 

# \---

# 

# \# Running the protocol

# 

# \## 1

# 

# Configure

# 

# ```

# OPENAI\_API\_KEY

# ```

# 

# as an environment variable.

# 

# \## 2

# 

# Run

# 

# ```

# gpt\_POO\_paper\_v8\_4\_detector\_ebau.py

# ```

# 

# to generate experimental outputs.

# 

# \## 3

# 

# Execute

# 

# ```

# run\_pipeline.py

# ```

# 

# or execute the pipeline scripts individually.

# 

# The pipeline automatically produces:

# 

# \- audited records

# \- master dataset

# \- descriptive statistics

# \- inferential statistics

# \- publication figures

# \- publication tables

# 

# \---

# 

# \# Human validation

# 

# A dedicated human validation interface is included.

# 

# The repository also contains:

# 

# \- validation forms

# \- validation outputs

# \- agreement calculations

# 

# \---

# 

# \# Data availability

# 

# Large experimental datasets are archived in Zenodo.

# 

# GitHub contains:

# 

# \- source code

# \- documentation

# \- sample data

# 

# The complete datasets are available through the Zenodo repository associated with the article.

# 

# DOI:

# 

# (To be added after publication.)

# 

# \---

# 

# \# Licenses

# 

# \## Code

# 

# MIT License

# 

# \## Original research outputs

# 

# Creative Commons Attribution 4.0

# 

# \## Third-party educational documents

# 

# Original examination documents remain the property of the corresponding educational authorities.

# 

# They are included exclusively for scientific reproducibility.

# 

# \---

# 

# \# Security

# 

# No API keys should be stored in this repository.

# 

# Authentication must be performed exclusively through

# 

# ```

# OPENAI\_API\_KEY

# ```

# 

# environment variables.

# 

# \---

# 

# \# Citation

# 

# If you use this repository, please cite both:

# 

# \- the accompanying article;

# \- the Zenodo dataset.

# 

# Citation information will be updated once the DOI has been assigned.

# 

# \---

# 

# \# Contact

# 

# Corresponding author:

# 

# Tomás Hombreiro

# 

# \---

# 

# \# Acknowledgements

# 

# The authors acknowledge the educational administrations that publicly provide the examination materials used in this research.

# 

# \---

# 

# \# Disclaimer

# 

# LLM services evolve over time.

# 

# Exact response replication may therefore not be possible.

# 

# The repository nevertheless allows full computational reproduction of the analytical workflow reported in the article.



