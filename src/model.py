"""RoBERTa-based sequence classifier for the IMDb sentiment task."""

from transformers import RobertaForSequenceClassification


def build_model(model_name: str, num_labels: int) -> RobertaForSequenceClassification:
    """Instantiate a RoBERTa model with a classification head.

    The backbone is loaded from pre-trained weights. The classification head
    (the final Linear layer projecting to `num_labels`) is randomly initialized
    and will be learned during fine-tuning.
    """
    return RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )