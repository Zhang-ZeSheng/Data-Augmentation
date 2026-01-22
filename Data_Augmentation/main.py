import json
import requests
import time
import os

# === 配置区 ===
DEEPSEEK_API_KEY = "**********************" #输入你的API
BASE_URL = "https://api.deepseek.com/v1/chat/completions"
INPUT_PATH = "./ACL/Data_Augmentation/test.json"  # 替换为你的输入路径
OUTPUT_PATH = "./ACL/Data_Augmentation/result.json"

def augment_via_deepseek(item):
    """
    增强数据, 试模仿 structure 和 dep_dfs_seq 的风格
    """
    prompt = f"""
    You are an expert NLP data annotator. Rewrite the following ABSA data for data augmentation.
    
    Requirements:
    1. Keep 'category' and 'sentiment' exactly the same: {item['category']}, {item['sentiment']}.
    2. Rewrite 'sentence' with different wording but similar meaning.
    3. Provide 'tokens': the new sentence split into words.
    4. Provide 'dep': the dependency head index for each token (root is -1).
    5. Provide 'structure': follow the parenthetical format (root (quad (aspect...))).
    6. Provide 'dep_dfs_seq': follow the specific tree-based string format.
    
    Input:
    Sentence: {item['sentence']}
    Aspect: {item['aspect']}
    Opinion: {item['opinion']}
    Original Structure: {item['structure']}
    Original Dep_DFS: {item['dep_dfs_seq']}

    Return ONLY a JSON object:
    {{
        "sentence": "...",
        "tokens": [...],
        "aspect": [...],
        "opinion": [...],
        "dep": [...],
        "structure": "...",
        "dep_dfs_seq": "..."
    }}
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
        content = json.loads(response.json()['choices'][0]['message']['content'])
        
        # 组装结果
        new_item = {}
        new_item["sentence"] = content["sentence"]
        new_item["aspect"] = content["aspect"]
        new_item["opinion"] = content["opinion"]
        
        # 自动计算 Span (基于 tokens)
        tk_low = [t.lower() for t in content["tokens"]]
        def get_spans(targets):
            res = []
            for t in targets:
                parts = t.lower().split()
                try:
                    start = tk_low.index(parts[0])
                    res.append([start, start + len(parts)])
                except: res.append([0, 1])
            return res
            
        new_item["aspectSpan"] = get_spans(content["aspect"])
        new_item["opinionSpan"] = get_spans(content["opinion"])
        new_item["sentiment"] = item["sentiment"]
        new_item["category"] = item["category"]
        new_item["dep"] = content["dep"]
        new_item["structure"] = content["structure"]
        new_item["dep_dfs_seq"] = content["dep_dfs_seq"]
        
        return new_item
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"File not found: {INPUT_PATH}")
        return

    print("🚀 正在以 JSONL 模式处理数据...")
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
                print(f"正在处理第 {i+1} 条...")
                
                augmented = augment_via_deepseek(item)
                if augmented:
                    # 以 JSONL 格式写入：每一行一个 JSON
                    f_out.write(json.dumps(augmented, ensure_ascii=False) + "\n")
                
                time.sleep(0.5) # 频率限制
            except Exception as e:
                print(f"第 {i+1} 行跳过，原因: {e}")

    print(f"✅ 任务完成！结果已存入: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
