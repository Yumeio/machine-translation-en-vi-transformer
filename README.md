# Machine Translation En-Vi Transformer

English-Vietnamese machine translation project using the Transformer model.

[Tiếng Việt](./README_vi.md)

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

## Training

To start training the model:

```bash
python train.py
```

- Models will be saved in the `weights/` directory.
- Configuration file is located at `config.py`.
- Loss chart will be saved at `loss_plot.png`.

## Inference

To run translation testing:

```bash
python infer.py
```

- The script will automatically load the latest weights or specified weights in `infer.py`.
- Currently, the script is translating a sample English sentence into Vietnamese.

## Evaluation (BLEU Score)

```bash
python eval.py
```

Results:
- **BLEU Score**: [24.05]

## References

1.  **Attention Is All You Need**: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  **IWSLT15 English-Vietnamese Dataset**: [https://nlp.stanford.edu/projects/nmt/](https://nlp.stanford.edu/projects/nmt/)
