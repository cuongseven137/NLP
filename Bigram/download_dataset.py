import os
from datasets import load_dataset

# 1. Khai báo Token để Hugging Face cấp quyền tải tốc độ cao
hf_token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
print("Đang tải dataset từ Hugging Face bằng token...")
# Tải dataset (sử dụng tham số token)
ds = load_dataset("tmnam20/Vietnamese-Book-Corpus", split="train", token=hf_token)

print("Tải xong! Đang lưu về ổ cứng cục bộ...")
# 2. Lưu toàn bộ dataset vào một thư mục có tên 'local_vietnamese_corpus' nằm cùng thư mục chứa code
ds.save_to_disk("local_vietnamese_corpus")

print("Hoàn tất.")
