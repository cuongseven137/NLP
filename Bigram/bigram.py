import random
import re
import math
from collections import defaultdict, Counter
from datasets import load_from_disk

# ==========================================
# 1 & 2. TẢI DỮ LIỆU VÀ HUẤN LUYỆN DẦN DẦN (STREAMING)
# ==========================================
print("Load dataset cục bộ...")
ds_local = load_from_disk("local_vietnamese_corpus")

# Khởi tạo bộ đếm
unigram_counts = Counter()
bigram_counts = Counter()

# Số lượng văn bản bạn muốn huấn luyện 
LIMIT = 500 

# Trộn ngẫu nhiên dataset (sử dụng seed ngẫu nhiên mỗi lần chạy)
# và cắt đúng n văn bản đầu tiên sau khi trộn
print(f"Đang chọn ngẫu nhiên {LIMIT} văn bản...")
random_seed = random.randint(1, 10000)
ds_random = ds_local.shuffle(seed=random_seed).select(range(LIMIT))
print(f"Bắt đầu huấn luyện dần dần trên {LIMIT} văn bản...")

for i, item in enumerate(ds_random): 
    text = item['text']
    if not text: 
        continue
        
    # In tiến độ để theo dõi, dùng \r để ghi đè dòng, không làm rác màn hình terminal
    print(f"\rĐang xử lý văn bản thứ {i+1}/{LIMIT}...", end="", flush=True)
    
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for s in sentences:
        s_clean = re.sub(r'[^\w\s]', '', s).strip().lower()
        tokens_raw = s_clean.split()
        
        if len(tokens_raw) >= 2:
            tokens = ["<s>"] + tokens_raw + ["</s>"]
            
            # 1. Cập nhật Unigram
            unigram_counts.update(tokens)
            
            # 2. Tạo danh sách các cặp Bigram cho câu này và cập nhật
            bigrams = [(tokens[j], tokens[j+1]) for j in range(len(tokens)-1)]
            bigram_counts.update(bigrams)


# Tính lại kích thước tập từ vựng V (bỏ <s> và </s>)
V = len([w for w in unigram_counts.keys() if w not in ['<s>', '</s>']])

def get_smoothed_prob(w1, w2):
    count_w1_w2 = bigram_counts[(w1, w2)]
    count_w1 = unigram_counts[w1]
    return (count_w1_w2 + 1) / (count_w1 + V)

# ==========================================
# 3. TÍNH LOG XÁC SUẤT CỦA CÂU
# ==========================================
def calculate_log_prob(sentence, verbose=False):
    tokens = ["<s>"] + sentence.lower().split() + ["</s>"]
    log_prob = 0.0
    
    if verbose:
        print(f"\nChi tiết tính Log xác suất cho câu: '{sentence}'")
    
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        p = get_smoothed_prob(w1, w2)
        log_p = math.log(p) 
        
        if verbose:
            print(f"P({w2} | {w1}) = {p:.6f} -> Log = {log_p:.4f}")
            
        log_prob += log_p
        
    return log_prob

test_sentence = "hôm nay trời đẹp lắm"
log_prob = calculate_log_prob(test_sentence, verbose=True)
try:
    standard_prob = math.exp(log_prob)
except OverflowError:
    standard_prob = 0.0
print(f"-> xác suất = {standard_prob:.4e}")

# ==========================================
# 4. SINH CÂU VỚI TOP-K SAMPLING
# ==========================================
transition_probs = defaultdict(list)
# Tính xác suất chuyển đổi dựa trên smoothed probability cho các cặp ĐÃ XUẤT HIỆN
for (w1, w2), count in bigram_counts.items():
    prob = count / unigram_counts[w1] # Lưu ý: khi sinh câu ta dùng MLE thuần túy cho mượt, không cần +1 của smoothing
    transition_probs[w1].append((w2, prob))

def generate_sentence_top_k(max_length=20, top_k=3):
    """Sinh câu sử dụng Top-K để tránh chọn các từ quá vô lý"""
    current_word = "<s>"
    sentence = []
    
    while len(sentence) < max_length:
        if current_word not in transition_probs:
            break
            
        # Sắp xếp các từ tiếp theo theo xác suất giảm dần
        next_words_probs = sorted(transition_probs[current_word], key=lambda x: x[1], reverse=True)
        
        # Chỉ lấy Top K từ có khả năng cao nhất
        top_choices = next_words_probs[:top_k]
        
        next_words = [w for w, p in top_choices]
        probs = [p for w, p in top_choices]
        
        # Chọn ngẫu nhiên trong Top K
        next_word = random.choices(next_words, weights=probs)[0]
        
        if next_word == "</s>":
            break
            
        sentence.append(next_word)
        current_word = next_word
        
    return " ".join(sentence)

print("\n--- Sinh 5 câu ngẫu nhiên (sử dụng Top-3 Sampling) ---")
for i in range(5):
    gen_sent = generate_sentence_top_k()
    
    # Tính log xác suất
    log_p = calculate_log_prob(gen_sent)
    
    try:
        standard_p = math.exp(log_p)
    except OverflowError:
        standard_p = 0.0
        
    print(f"\nCâu {i+1}: {gen_sent}")
    # print(f" -> Xác suất Log: {log_p:.4f}")
    print(f" -> Xác suất: {standard_p:.4e}")
    