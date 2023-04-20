import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/Users/cayden/Documents/doc/openai/code/MiniGPT-4/models/pytorch_model.bin"

# tokenizer = AutoTokenizer.from_pretrained(model_path,repo_name="TurkuNLP", repo_type="HuggingFace")
# model = AutoModelForCausalLM.from_pretrained(model_path,repo_name="TurkuNLP", repo_type="HuggingFace")

# 使用远程 执行后会进行下载
model = AutoModelForCausalLM.from_pretrained("TurkuNLP/gpt3-finnish-small")
tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/gpt3-finnish-small")

# 使用模型生成新的文本
prompt = "The quick brown fox"
inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(inputs, max_length=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# 生成文本
generated_text = model.generate(
    input_ids=tokenizer.encode("你好，我是MiniGPT-4，", return_tensors="pt"),
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# 打印生成的文本
print(tokenizer.decode(generated_text[0], skip_special_tokens=True))
