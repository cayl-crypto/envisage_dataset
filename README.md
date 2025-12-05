# ENVISAGE Dataset

Welcome to the **ENVISAGE Dataset** repository. This dataset has been collected and curated to support research on **image captioning for visually impaired individuals**. It contains over **53,000 images**, each annotated with high‑quality captions emphasizing spatial and contextual information.

---

## 📦 Dataset Downloads (Zenodo)
Below are the Zenodo links where the ENVISAGE dataset components can be downloaded:

> **🔗 Please insert your Zenodo links below.**

- Dataset Part 1: [Zenodo Link 1](#)
- Dataset Part 2: [Zenodo Link 2](#)
- Dataset Part 3: [Zenodo Link 3](#)
- Additional Files: [Zenodo Link 4](#)

You can add or remove links depending on how many parts your dataset includes.

---

## 📄 Dataset Description
The ENVISAGE dataset was collected by visually impaired contributors and annotated by trained annotators following strict guidelines:

- **53,452 images** in total
- Captions focus on spatial relationships, context, and object interactions to better support assistive technologies
- No punctuation in captions to simplify RNN/Transformer preprocessing
- Multi‑label tags available per image using MSCOCO supercategories
- Dataset split into training (70%), validation (10%), and testing (20%)
- **10 stratified splits** prepared; split_9 selected as the benchmark split

---

## 📁 Repository Structure
```
ENVISAGE_dataset/
│
├── README.md
├── annotations/           # Caption annotations and tags
├── images/                # (Optional) Symlink or sample images
├── scripts/               # Example loaders or utilities
└── examples/              # Notebook or script examples
```

---

## 🧪 Benchmarking
Our experiments include multiple captioning architectures:

- CNN‑RNN (Show and Tell)
- Attention‑based models (Show, Attend and Tell)
- Transformer‑based captioning
- GPT‑based captioning (CLIPCap, BLIP)
- Vision‑language evaluation using COCOEvalCap metrics

More benchmarking details will be provided in this repository.

---

## 📚 Citation
If you use the ENVISAGE dataset in academic work, please cite:

```
[Your citation text for Zenodo or journal paper]
```

---

## 📨 Contact
For questions or collaboration opportunities:
- **Maintainer:** [Your Name]
- **Email:** [Your Email]

---

Let me know if you want:
- automatic insertion of your Zenodo links,
- a polished project description,
- example dataset loaders/code,
- or GitHub Actions for releasing the dataset.

