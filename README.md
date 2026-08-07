# OCR Historical Chinese – PP-OCRv5 Baseline

Project thử nghiệm OCR trên dữ liệu văn bản Hán cổ, với baseline đầu tiên **B0** sử dụng mô hình pretrained `PP-OCRv5_server_rec` trên tập **MTH1000**.

Hiện project chứa ba nguồn dữ liệu raw:

- `MTHv2`: gồm `MTH1000`, `MTH1200` và `TKH`.
- `document`: dữ liệu document OCR ở định dạng LMDB.
- `handwriting`: dữ liệu handwriting từ HWDB / ICDAR 2013 ở định dạng LMDB.

Baseline B0 hiện sử dụng **MTH1000**.

---

## 1. Cấu trúc project

```text
final_project/
├── data/
│   ├── processed/
│   │   └── MTH1000_B0/
│   │       ├── images/
│   │       │   ├── test/
│   │       │   ├── train/
│   │       │   └── val/
│   │       ├── skipped.json
│   │       ├── splits.json
│   │       ├── test.tsv
│   │       ├── train.tsv
│   │       ├── train_characters.txt
│   │       └── val.tsv
│   │
│   └── raw/
│       ├── document/
│       │   └── document/
│       │       ├── document_test/
│       │       │   ├── data.mdb
│       │       │   └── lock.mdb
│       │       ├── document_train/
│       │       │   ├── data.mdb
│       │       │   └── lock.mdb
│       │       └── document_val/
│       │           ├── data.mdb
│       │           └── lock.mdb
│       │
│       ├── handwriting/
│       │   └── hwdb_ic13/
│       │       ├── handwriting_hwdb_train/
│       │       │   ├── data.mdb
│       │       │   └── lock.mdb
│       │       ├── handwriting_hwdb_val/
│       │       │   ├── data.mdb
│       │       │   └── lock.mdb
│       │       └── handwriting_ic13_test/
│       │           ├── data.mdb
│       │           └── lock.mdb
│       │
│       └── mthv2/
│           ├── MTH1000/
│           │   ├── img/
│           │   ├── label_char/
│           │   ├── label_table/
│           │   └── label_textline/
│           ├── MTH1200/
│           │   ├── img/
│           │   ├── label_char/
│           │   ├── label_table/
│           │   └── label_textline/
│           └── TKH/
│               ├── img/
│               ├── label_char/
│               ├── label_table/
│               └── label_textline/
│
├── outputs/
│   ├── B0_PP-OCRv5_server_rec/
│   │   ├── metrics.json
│   │   └── predictions.tsv
│   └── B0_smoke_test/
│       ├── metrics.json
│       └── predictions.tsv
│
├── scripts/
│   ├── data/
│   │   └── prepare_mth1000_b0.py
│   └── eval/
│       └── evaluate_paddleocr_b0.py
│
├── .gitignore
└── README.md
```

> Các lệnh bên dưới giả sử terminal đang đứng tại thư mục:
>
> ```bash
> ~/final_project
> ```

---

## 2. Môi trường

Kích hoạt virtual environment:

```bash
cd ~/final_project
source .venv/bin/activate
```

Cài các dependency cần thiết nếu chưa có:

```bash
pip install paddleocr paddlepaddle-gpu opencv-python pillow numpy tqdm rapidfuzz
```

Nếu chạy CPU, cài phiên bản PaddlePaddle phù hợp với môi trường CPU thay cho `paddlepaddle-gpu`.

---

# Baseline B0 – PP-OCRv5 trên MTH1000

Baseline B0 gồm hai bước:

1. Chuẩn bị MTH1000 và crop từng text line.
2. Chạy `PP-OCRv5_server_rec` pretrained trên test set và tính metric.

Baseline này không fine-tune model.

---

## 3. Chuẩn bị MTH1000

Script:

```text
scripts/data/prepare_mth1000_b0.py
```

Dataset đầu vào:

```text
data/raw/mthv2/MTH1000
```

Script sử dụng:

```text
data/raw/mthv2/MTH1000/
├── img/
└── label_textline/
```

Chạy:

```bash
python scripts/data/prepare_mth1000_b0.py \
  --root data/raw/mthv2/MTH1000 \
  --output data/processed/MTH1000_B0
```

### Thiết lập mặc định

```text
seed       = 2026
train      = 800 pages
validation = 100 pages
test       = 100 pages
padding    = 2 px
min-side   = 4 px
```

Việc split được thực hiện theo **page**, không split ngẫu nhiên từng text line.

Điều này giúp tránh việc các dòng thuộc cùng một trang xuất hiện ở nhiều split khác nhau.

### Ghi đè crop cũ

Nếu muốn tạo lại toàn bộ crop:

```bash
python scripts/data/prepare_mth1000_b0.py \
  --root data/raw/mthv2/MTH1000 \
  --output data/processed/MTH1000_B0 \
  --overwrite
```

---

## 4. Output của bước prepare

Sau khi hoàn tất:

```text
data/processed/MTH1000_B0/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── train.tsv
├── val.tsv
├── test.tsv
├── splits.json
├── skipped.json
└── train_characters.txt
```

Ý nghĩa:

- `images/train/`: text-line crop của training set.
- `images/val/`: text-line crop của validation set.
- `images/test/`: text-line crop của test set.
- `train.tsv`: manifest training.
- `val.tsv`: manifest validation.
- `test.tsv`: manifest test.
- `splits.json`: danh sách page thuộc từng split.
- `skipped.json`: các sample bị bỏ qua khi parse hoặc crop.
- `train_characters.txt`: các ký tự xuất hiện trong training split.

Các text line dọc được xoay để phù hợp hơn với recognizer dạng text-line ngang.

Kiểm tra nhanh:

```bash
ls -lah data/processed/MTH1000_B0
```

Đếm số crop:

```bash
find data/processed/MTH1000_B0/images/train -type f | wc -l
find data/processed/MTH1000_B0/images/val -type f | wc -l
find data/processed/MTH1000_B0/images/test -type f | wc -l
```

---

## 5. Smoke test

Trước khi chạy toàn bộ test set, nên kiểm tra pipeline bằng một số lượng nhỏ sample.

Ví dụ 100 samples:

```bash
python scripts/eval/evaluate_paddleocr_b0.py \
  --dataset-root data/processed/MTH1000_B0 \
  --manifest test.tsv \
  --output-dir outputs/B0_smoke_test \
  --model-name PP-OCRv5_server_rec \
  --device gpu:0 \
  --batch-size 8 \
  --limit 100
```

Nếu chạy thành công, output sẽ nằm tại:

```text
outputs/B0_smoke_test/
├── metrics.json
└── predictions.tsv
```

---

## 6. Chạy baseline B0 đầy đủ

Script:

```text
scripts/eval/evaluate_paddleocr_b0.py
```

Chạy toàn bộ test set bằng GPU:

```bash
python scripts/eval/evaluate_paddleocr_b0.py \
  --dataset-root data/processed/MTH1000_B0 \
  --manifest test.tsv \
  --output-dir outputs/B0_PP-OCRv5_server_rec \
  --model-name PP-OCRv5_server_rec \
  --device gpu:0 \
  --batch-size 8
```

### Chạy bằng CPU

```bash
python scripts/eval/evaluate_paddleocr_b0.py \
  --dataset-root data/processed/MTH1000_B0 \
  --manifest test.tsv \
  --output-dir outputs/B0_PP-OCRv5_server_rec \
  --model-name PP-OCRv5_server_rec \
  --device cpu \
  --batch-size 8
```

---

## 7. Cấu hình baseline

```text
Experiment        : B0
Recognizer        : PP-OCRv5_server_rec
Weights           : Official pretrained
Fine-tuning       : No
Custom dictionary : No
Augmentation      : No
Post-processing   : No
Normalization     : NFC
Device            : GPU / CPU
```

Đây là baseline pretrained nhằm đo khả năng nhận diện ban đầu của PP-OCRv5 trên MTH1000 trước khi thực hiện fine-tuning, augmentation hoặc post-correction.

---

## 8. Metric đánh giá

Baseline ghi các metric vào:

```text
outputs/B0_PP-OCRv5_server_rec/metrics.json
```

Các metric chính:

### Character Error Rate – CER

```text
CER = total edit distance / total number of ground-truth characters
```

CER càng thấp càng tốt.

### Exact-match accuracy

Tỷ lệ text line mà prediction giống hoàn toàn ground truth.

### Mean confidence

Confidence trung bình do recognizer trả về.

### Speed

Pipeline ghi thêm:

```text
elapsed_seconds
lines_per_second
milliseconds_per_line
```

để đánh giá tốc độ inference.

---

## 9. Predictions

Chi tiết prediction nằm tại:

```text
outputs/B0_PP-OCRv5_server_rec/predictions.tsv
```

Các trường:

```text
image_path
page_id
line_number
ground_truth
prediction
confidence
edit_distance
gt_length
pred_length
sample_cer
exact_match
```

Các sample được sort theo lỗi từ cao xuống thấp dựa trên:

1. `sample_cer`
2. `edit_distance`

Do đó có thể dùng trực tiếp file này để phân tích lỗi OCR.

Xem nhanh:

```bash
head -20 outputs/B0_PP-OCRv5_server_rec/predictions.tsv
```

Xem metric:

```bash
cat outputs/B0_PP-OCRv5_server_rec/metrics.json
```

---

## 10. Lệnh chạy nhanh

### Prepare dataset

```bash
python scripts/data/prepare_mth1000_b0.py \
  --root data/raw/mthv2/MTH1000 \
  --output data/processed/MTH1000_B0
```

### Smoke test

```bash
python scripts/eval/evaluate_paddleocr_b0.py \
  --dataset-root data/processed/MTH1000_B0 \
  --manifest test.tsv \
  --output-dir outputs/B0_smoke_test \
  --device gpu:0 \
  --batch-size 8 \
  --limit 100
```

### Full evaluation

```bash
python scripts/eval/evaluate_paddleocr_b0.py \
  --dataset-root data/processed/MTH1000_B0 \
  --manifest test.tsv \
  --output-dir outputs/B0_PP-OCRv5_server_rec \
  --model-name PP-OCRv5_server_rec \
  --device gpu:0 \
  --batch-size 8
```

---

## 11. Raw datasets

### MTHv2

```text
data/raw/mthv2/
├── MTH1000/
├── MTH1200/
└── TKH/
```

Hiện B0 sử dụng:

```text
data/raw/mthv2/MTH1000
```

`MTH1200` và `TKH` được giữ lại cho các experiment tiếp theo.

### Document

```text
data/raw/document/document/
├── document_train/
├── document_val/
└── document_test/
```

Các split đang ở định dạng LMDB.

### Handwriting

```text
data/raw/handwriting/hwdb_ic13/
├── handwriting_hwdb_train/
├── handwriting_hwdb_val/
└── handwriting_ic13_test/
```

Các split đang ở định dạng LMDB.

---

## 12. Git

Dataset, processed data, model output, virtual environment và archive không nên commit lên repository.

Các file này được loại trừ thông qua `.gitignore`.

Source code và tài liệu chính nên được commit:

```text
README.md
.gitignore
scripts/
```
