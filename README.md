# IMDb Sentiment Classification with RoBERTa

Fine-tuning [RoBERTa-base](https://huggingface.co/roberta-base) on the IMDb movie reviews dataset for binary sentiment classification (positive / negative).

## Project Status

🚧 Work in progress.

## Goals

- Achieve >94% test accuracy on IMDb (50k reviews, balanced split).
- Apply professional ML engineering practices: reproducibility, configuration management, clean code, comprehensive logging.
- Document the full training pipeline and results.

## Stack

- Python 3.11
- PyTorch (nightly, CUDA 12.8+ for Blackwell GPUs)
- Hugging Face Transformers & Datasets
- TensorBoard for experiment tracking

## Dataset

IMDb (Maas et al., 2011): 50,000 movie reviews split evenly into 25k train / 25k test, balanced across positive and negative classes. Reviews with neutral ratings (5–6 / 10) are excluded from the dataset by construction.

## License

MIT — see [LICENSE](LICENSE).