import os
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate

def get_metric(metric_name: str, device: str):
    if metric_name == "bleu":
        return BLEUScore().to(device)
    elif metric_name == "cer":
        return CharErrorRate().to(device)
    elif metric_name == "wer":
        return WordErrorRate().to(device)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")