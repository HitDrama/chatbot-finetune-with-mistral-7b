# chatbot-finetune-with-mistral-7b

- This project features a fine-tuned Mistral-7B model (developed by Mistral AI) specifically optimized for chatbot applications. The model was trained on a curated subset (train[:500]) of the tatsu-lab/alpaca dataset, using efficient 4-bit quantization and LoRA (Low-Rank Adaptation) techniques to reduce resource usage while maintaining high response quality.

- The repository provides all key configuration details, training scripts, and a downloadable checkpoint for quick deployment or further customization.

---

## Demo

<img src="https://github.com/HitDrama/chatbot-finetune-with-mistral-7b/blob/main/static/2025-06-27%2011-13-39.gif" style="width:100%; max-width:1000px;" />
<!-- Thay thế bằng file gif demo thực tế của bạn -->

---

## Model & Training Configuration

- **Base Model:** [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/) (from Mistral AI)
- **Quantization:** 4-bit (nf4, double quantization, float16 compute)
- **LoRA:** r=8, alpha=16, dropout=0.05, target_modules=[q_proj, v_proj]
- **Dataset:** `tatsu-lab/alpaca`, split: `train[:500]`
- **Training Args:**
  - Epochs: 1
  - Per device batch size: 1
  - Gradient accumulation steps: 2
  - Optimizer: paged_adamw_8bit
  - Warmup steps: 5
  - Logging steps: 5
  - Save strategy: epoch
  - Learning rate: 2e-4
  - FP16 training: Yes
  - Gradient norm clip: 0.3
  - Gradient checkpointing: Enabled
- **Model Download:** [Google Drive link](https://drive.google.com/drive/folders/151wCbeNOz7JgO48rG9MOFMvDfrNe3oe4?usp=sharing)

---

## Configuration Details

### Quantization (`BitsAndBytesConfig`)
- **4-bit quantization:** Giúp tiết kiệm bộ nhớ, tăng tốc độ train/inference.
- **nf4:** Dạng nén trọng số hiệu quả hơn so với int4 truyền thống.
- **float16 compute:** Giảm sử dụng VRAM, phù hợp GPU consumer.
- **Double quantization:** Nâng cao hiệu suất lưu trữ và inference.

### LoRA (`LoraConfig`)
- **r=8, alpha=16:** Cấu hình phổ biến, cân bằng giữa chất lượng và tài nguyên.
- **target_modules=[q_proj, v_proj]:** Chỉ fine-tune phần trọng số chính của attention.
- **dropout=0.05:** Giúp mô hình tránh overfitting, tăng tính tổng quát.

### TrainingArguments
- **Batch size nhỏ + gradient accumulation:** Cho phép train trên máy VRAM thấp.
- **Optimizer paged_adamw_8bit:** Tối ưu tốc độ và tài nguyên bộ nhớ.
- **FP16 training:** Tối ưu hiệu suất và tiết kiệm RAM.
- **Epochs = 1:** Huấn luyện thử nghiệm với tập dữ liệu nhỏ.

### Dataset
- **tatsu-lab/alpaca, split="train[:500]":** Dữ liệu huấn luyện gồm 500 mẫu ngắn, phù hợp thử nghiệm, benchmark.

---

## Model Download

- Model checkpoint và các file liên quan: [Tải tại đây](https://drive.google.com/drive/folders/151wCbeNOz7JgO48rG9MOFMvDfrNe3oe4?usp=sharing)

---

## Author

- **Dev:** Đặng Tô Nhân

---


