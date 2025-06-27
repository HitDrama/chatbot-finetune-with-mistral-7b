# from flask import Flask, request, render_template
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# app = Flask(__name__)

# # Tải mô hình và tokenizer
# model_path = "./qwen2-standalone"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#   model_path,
#   torch_dtype=torch.float32, # float32 an toàn hơn trên CPU
#   device_map="cpu" # Dùng CPU
# )

# @app.route("/", methods=["GET", "POST"])
# def index():
#   response = ""
#   prompt = request.form.get("prompt", "")
  
#   if request.method == "POST" and prompt.strip():
#     # Mã hóa prompt thành token
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids

#     # Sinh phản hồi dài (gần như không giới hạn)
#     output_ids = model.generate(
#       input_ids,
#       do_sample=True,
#       top_p=0.9,
#       temperature=0.8,
#       max_new_tokens=256, # Có thể tăng lên nếu máy đủ mạnh
#       eos_token_id=tokenizer.eos_token_id # Dừng khi gặp token kết thúc
#     )

#     # Giải mã output thành văn bản
#     full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     # Nếu chỉ muốn phần sinh ra, cắt bỏ phần prompt đầu:
#     if full_text.startswith(prompt):
#       response = full_text[len(prompt):].strip()
#     else:
#       response = full_text # fallback nếu không trùng

#   return render_template("index.html", prompt=prompt, response=response)

# if __name__ == "__main__":
#   app.run(debug=True)



from flask import Flask, request, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig

app = Flask(__name__)

# Tải mô hình và tokenizer
model_path = "./fine_tuned_model_merged"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
  load_in_4bit=True, # load mô hình dạng 4 bit
  bnb_4bit_quant_type="nf4", # dung dịch sang dạn n5f cho nén trọng số
  bnb_4bit_compute_dtype=torch.float16, # tiets kiệm Vram
  bnb_4bit_use_double_quant=True, # cải thiện hiệu suất
)

model = AutoModelForCausalLM.from_pretrained(
  model_path,
  quantization_config=bnb_config,
  device_map="auto",
  trust_remote_code=True,
  low_cpu_mem_usage=True,
)

# Tạo pipeline với batch nhỏ và tối ưu
text_generator = pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
  max_length=128, # Giảm max_length để tăng tốc
  truncation=True,
  return_full_text=False # Chỉ trả về phần sinh ra
)

@app.route("/", methods=["GET", "POST"])
def index():
  response = ""
  if request.method == "POST":
    prompt = request.form["prompt"]
    response = text_generator(prompt, do_sample=True, top_p=0.9, temperature=0.7)[0]["generated_text"]
  return render_template("index.html", prompt=request.form.get("prompt", ""), response=response)

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=5000)
  
  
#   pip install accelerate tp_plan device_map bitsandbytes