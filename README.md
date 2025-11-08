# Exploring EEG-Image Classification and EEG-Based Caption Retrieval

## Project Overview

This project explores the fundamental question: **Can we decode what someone is seeing from their brain activity alone?** 

We investigate whether electroencephalography (EEG) signals recorded while participants view images can be meaningfully aligned with multimodal representations—specifically, embedding spaces that jointly represent both visual and linguistic information. The project uses a dataset of EEG recordings captured while participants viewed images from 20 different object categories, each with associated natural language captions.

## Main Tasks

### Task 1: Image Category Classification

**Objective:** Build an EEG encoder to predict which of 20 object categories an image belongs to based solely on EEG signals recorded while a subject viewed that image.

- Implement a multi-head learning framework with a shared backbone architecture
- Train subject-specific prediction heads for classification
- Evaluate classification accuracy across subjects and sessions

### Task 2: Caption Retrieval via Multimodal Alignment

**Objective:** Align EEG signals with language representations in a shared embedding space, enabling caption retrieval directly from brain activity.

#### Task 2A: Image-Caption Retrieval with CLIP
- Use pretrained CLIP model to perform image-to-caption retrieval
- Establish baseline performance for retrieval tasks
- Evaluate with metrics: Recall@K, BERTScore, CLIPScore, and Mean Average Precision (MAP)

#### Task 2B: EEG-Caption Retrieval
- Train EEG encoder to map brain signals into CLIP's text embedding space
- Experiment with knowledge distillation, contrastive learning, and parameter-efficient fine-tuning
- Evaluate EEG-to-caption retrieval performance

## Dataset

- **EEG Recordings:** 13 subjects, 5 sessions each, low-speed paradigm (1 second image presentation)
- **Images:** 10,000 images from PASCAL VOC and ImageNet (20 categories, 500 images each)
- **Captions:** Natural language descriptions for each image
- **Location:** `/ocean/projects/cis250019p/gandotra/11785-gp-eeg` (read-only)

## Setup

See `environment.yml` for conda environment setup. Requires:
- Python 3.10
- PyTorch with CUDA support
- Transformers (Hugging Face)
- Additional dependencies listed in `environment.yml`

## Project Structure

```
exploring-eeg/
├── 1-instructions/                    # Project instructions and documentation
│   ├── CLIP_Fine_Tuning.ipynb        # Starter code for CLIP fine-tuning strategies
│   ├── EEG_write-up.pdf              # Project write-up document
│   └── instructions.md               # Complete assignment instructions
│
├── 2-docs-reference/                 # Reference papers and materials
│   ├── 1-original-paper.pdf          # Original EEG dataset paper
│   ├── 2-destilation.pdf             # Knowledge distillation reference
│   ├── 3-lora.pdf                    # LoRA paper
│   ├── 4-visual-models.pdf           # Visual models reference
│   ├── 5-multimodal-learning.pdf     # Multimodal learning reference
│   └── EEGDataset-main/              # Reference implementation code
│       ├── paradigm-1_RSVP/          # RSVP paradigm code
│       ├── paradigm-2_low-speed/     # Low-speed paradigm code
│       ├── preprocess/                # Preprocessing scripts
│       └── technical_validate/       # Baseline model implementations
│
├── 3-midterm-report/                 # Midterm deliverables
│   └── group-9.pdf                   # Midterm report
│
├── 4-eeg-classifiers/                # Task 1: EEG Classification
│   ├── data_extraction/              # Data loading utilities
│   │   ├── dataloader.py             # PyTorch dataloader
│   │   ├── dataset_builder.py        # Dataset construction
│   │   └── path_setup.py             # Path configuration
│   ├── EER_BL_Model.ipynb            # EEG encoder baseline model
│   ├── results_per_class_accuracy.tex # Per-class accuracy results
│   └── results_summary.tex           # Results summary
│
├── 5-caption-retrieval/              # Task 2: Caption Retrieval
│   └── task2a.ipynb                 # Task 2A: Image-Caption Retrieval with CLIP
│
├── environment.yml                   # Conda environment configuration
└── README.md                          # This file
```

## Course Information

**Course:** 11-685: Introduction to Deep Learning (Fall 2025)  
**Institution:** Carnegie Mellon University

