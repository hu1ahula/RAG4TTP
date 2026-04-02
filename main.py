#!/usr/bin/env python3
import json
import re
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

def extract_mitre_techniques(text):
    # 匹配 MITRE ATT&CK 技术/子技术 ID，例如 T1059 或 T1059.001
    pattern = re.compile(r'T\d{4}(?:\.\d{3})?')
    matches = set(pattern.findall(text))
    # 去重后排序，保证输出稳定，便于评估与复现
    return sorted(matches)

def load_dataset(file_path):
    """从 JSON 文件加载测试集。"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_results(data, name):
    """将包含预测结果的数据保存到 results 目录。"""
    # 若目录不存在则自动创建
    os.makedirs('./results', exist_ok=True)
    
    output_path = f'./results/{name}_results.json'
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {output_path}")

def run_inference(data, base_url, model_name):
    """调用 OpenAI 兼容接口进行推理并回填 predicted 字段。"""
    print(f"Running inference on {len(data)} examples using model: {model_name}")
    
    # 使用可配置的 base_url，便于接入本地 vLLM 或其他兼容服务
    client = OpenAI(
        base_url=base_url,
        api_key="dummy-key"
    )
    
    for i, example in enumerate(tqdm(data)):
        instruction = example["instruction"]
        input_text = example["input"]
        
        # 按训练/推理约定拼接成单轮用户输入
        prompt = f"# Instruction:\n{instruction}\n# Input:\n{input_text}\n# Response:\n"   

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=151,
            temperature=0.7
        )
        
        # 提取模型输出文本
        response_text = response.choices[0].message.content

        # 从生成文本中抽取 MITRE 技术 ID
        predicted_techniques = extract_mitre_techniques(response_text)
        
        # 将预测结果写回当前样本
        data[i]["predicted"] = predicted_techniques
    
    return data

def main():
    parser = argparse.ArgumentParser(description="使用 OpenAI 兼容 API 对数据集做推理并抽取 MITRE 技术 ID")
    parser.add_argument("--name", required=True, help="Name of the dataset file (without .json extension)")
    parser.add_argument("--base_url", default="http://localhost:9003/v1", help="Base URL of the vLLM hosted model with OpenAI-compatible API")
    parser.add_argument("--model", required=True, help="Name of the model to use for inference")
    args = parser.parse_args()
    
    # 约定测试集路径：datasets/TechniqueRAG-Datasets/test/{name}.json
    dataset_path = Path(f"datasets/TechniqueRAG-Datasets/test/{args.name}.json")
    
    if not dataset_path.exists():
        print(f"Error: Dataset file {dataset_path} does not exist.")
        return
    
    # 1) 读取数据
    print(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)
    
    # 2) 执行推理并写入 predicted
    data = run_inference(data, args.base_url, args.model)
    
    # 3) 保存结果文件
    save_results(data, args.name)
    
    print("Done!")

if __name__ == "__main__":
    main()
