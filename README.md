# Machine Translation En-Vi Transformer

Dự án dịch máy Anh-Việt sử dụng mô hình Transformer.

## Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Huấn luyện (Training)

Để bắt đầu huấn luyện mô hình:

```bash
python train.py
```

- Mô hình sẽ được lưu trong thư mục `weights/`.
- File cấu hình nằm ở `config.py`.
- Biểu đồ Loss sẽ được lưu tại `loss_plot.png`.

## Dự đoán (Inference)

Để chạy thử nghiệm dịch:

```bash
python infer.py
```

- Script sẽ tự động load weights mới nhất hoặc weights được chỉ định trong `infer.py`.
- Hiện tại script đang chạy dịch thử một câu mẫu tiếng Anh sang tiếng Việt.

## Chạy lấy điểm BLEU

```bash
python eval.py
```

Kết quả:
- **BLEU Score**: [24.05]

## References

1.  **Attention Is All You Need**: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  **IWSLT15 English-Vietnamese Dataset**: [https://nlp.stanford.edu/projects/nmt/](https://nlp.stanford.edu/projects/nmt/)