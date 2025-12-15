import os
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
from gemini_score import GeminiScore

def get_metric(metric_name: str, device: str):
    if metric_name == "bleu":
        return BLEUScore().to(device)
    elif metric_name == "cer":
        return CharErrorRate().to(device)
    elif metric_name == "wer":
        return WordErrorRate().to(device)
    elif metric_name == "gemini":
        return GeminiScore()
    else:
        raise ValueError(f"Unknown metric: {metric_name}")