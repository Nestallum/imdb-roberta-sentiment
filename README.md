# IMDb Sentiment Classification with RoBERTa

Fine-tuning [RoBERTa-base](https://huggingface.co/roberta-base) on the IMDb movie reviews dataset for binary sentiment classification (positive / negative).

---

## Results

🚧 *Training in progress — results will be reported here once available.*

---

## Project Structure

🚧 *To be documented as the codebase grows.*

---

## Stack

- Python 3.11
- PyTorch (nightly, CUDA 12.8+ for Blackwell GPUs)
- Hugging Face Transformers & Datasets
- TensorBoard for experiment tracking

---

## Dataset

[IMDb](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., 2011) — 50,000 movie reviews split evenly into 25k train / 25k test, balanced across positive and negative classes. Reviews with neutral ratings (5–6 / 10) are excluded from the dataset by construction.

---

## License

MIT — see [LICENSE](LICENSE).