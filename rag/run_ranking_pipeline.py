import argparse
import json
import pandas as pd
from tqdm import tqdm
import re
import time
import copy
from openai import OpenAI
import glob
from libs import resources as res, rank
from libs.pygaggle.data.relevance import RelevanceExample
from libs.pygaggle.model import StepEvaluator
import nltk

class OpenaiClient:
    """对 OpenAI 兼容接口的轻量封装，用于 RankGPT 重排调用。"""
    def __init__(self, api_key=None, base_url="https://api.deepseek.com"):
        if api_key is None:
            raise ValueError("Please provide an API Key.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, *args, **kwargs):
        # 带重试的聊天调用：网络抖动或服务端错误时循环重试
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=90)
                return completion.choices[0].message.content
            except Exception as e:
                print(f"An exception occurred: {e}")
                if "This model's maximum context length is" in str(e):
                    return 'ERROR::reduce_length'
                time.sleep(1)

def get_prefix_prompt(query, num):
    """
    构造 RankGPT 会话前缀消息。
    - system: 角色设定与排序规则
    - user: 告知将提供 num 条 passage，要求按 query 重排
    - assistant: 占位回复，进入逐条 passage 交互
    """
    return [
        {
            'role': 'system',
            'content': (
                "You are RankGPT, an expert security analyst specializing in ranking passages by their relevance to the query. "
                "Your goal is to focus on core MITRE ATT&CK techniques (e.g., process injection, data exfiltration, lateral movement), "
                "and sub-techniques (e.g., Windows Command Shell, web services, masquerading), ensuring balanced attention across attack phases (initial access, execution, persistence, etc.).\n\n"
                
                "Key Instructions for Ranking:\n"
                "- First break down the steps involved in the attack (query)"
                "- Try to match each step to the techniques (passages)"
                "- Queries may contain multiple implied techniques-subtechniques, so be careful and try to break the query to find multiple implicit techniques.\n"
                "- Avoid overemphasizing one technique at the expense of others (e.g., obfuscation vs. discovery).\n"
                "- Provide the attack steps breakdown and reasoning before delivering the final rankings by explaining how each passage connects to the key attack phases or techniques. If a passage doesn't match the query at all don't provide the reasoning for it.\n"
            )
        },
        {
            'role': 'user',
            'content': (
                f"I will provide you with {num} passages, each indicated by number identifier [].\n"
                f"Rank the passages (techniques/sub-techniques) based on their relevance to the query: {query}. "
                "Explain your reasoning before providing the final ranking."
            )
        },
        {
            'role': 'assistant',
            'content': "Please provide the passages."
        }
    ]

def get_post_prompt(query, num):
    """构造最后一条指令，要求输出标准化排序格式 `[x] > [y] > ...`。"""
    return (
        f"Search Query: {query}. Rank the {num} passages based on their relevance to the search query, "
        "in descending order using identifiers. Provide your reasoning first, explaining the key techniques or phases identified in each passage, "
        "then output the final ranking in this format: [] > []. While reasoning for only matching passage is required, make sure to include all passages in the final ranking order."
        "Ensure format like this only [x] > [y] > [z] > [a] > [b] > [c] in final rankings, no other format."
    )

def create_permutation_instruction(item=None, rank_start=0, rank_end=100):
    """
    将一个 query 的候选文档切片打包成 RankGPT 输入消息列表。
    item 结构核心字段：
    - item['query']: 查询文本
    - item['hits']: 候选 passage 列表（每条至少含 content）
    """
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])
    max_length = 300
    messages = get_prefix_prompt(query, num)
    rank_num = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank_num += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # 控制每条 passage 长度，降低上下文超限风险
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank_num}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank_num}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})
    return messages

def run_llm(messages, api_key, model_name):
    """执行一次 RankGPT 调用，返回模型原始文本输出。"""
    client = OpenaiClient(api_key=api_key)
    response = client.chat(model=model_name, messages=messages, temperature=0.0)
    return response

def clean_response(response: str):
    """
    从模型输出中提取排序序列。
    目标格式示例：`[3] > [1] > [2]`
    返回值为纯数字空格串，例如：`3 1 2`
    """
    sequence_pattern = re.compile(r'\[\d+\](?:\s*>\s*\[\d+\])+')
    match = sequence_pattern.search(response)
    if match:
        extracted_sequence = match.group(0)
        cleaned_sequence = re.sub(r'[^\d]', ' ', extracted_sequence)
        cleaned_sequence = ' '.join(cleaned_sequence.split())
        return cleaned_sequence
    return ''

def remove_duplicate(response):
    """按出现顺序去重，避免模型重复输出某个编号。"""
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    """
    根据模型返回的排序编号，重排 item['hits'] 的指定切片区间。
    - 非法编号会被过滤
    - 未出现的编号会按原顺序补到末尾，保证结果完整
    """
    response = clean_response(permutation)
    if not response:
        return item
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = list(range(len(cut_range)))
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item

def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo', api_key=None):
    """单次重排：构造提示词 -> 调用 LLM -> 应用重排结果。"""
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end)
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item

def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10, model_name='gpt-3.5-turbo', api_key=None):
    """
    滑窗重排策略：
    从 [rank_end-window_size, rank_end) 开始向前滑动，
    每个窗口做一次 permutation_pipeline，逐步优化前 rank_end 范围内排序。
    """
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name, api_key=api_key)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item

def extract_technique_ids(data):
    """从 hits 内容中提取 MITRE 技术 ID，形成最终排名列表。"""
    pattern = r'T\d{4}(?:\.\d{3})?'
    technique_ids = []
    for entry in data:
        match = re.search(pattern, entry['content'])
        if match:
            technique_ids.append(match.group())
    return technique_ids

def get_dict_from_examples(examples, df_new):
    """
    将 pygaggle 的 RelevanceExample 结果转换成易读字典结构。
    输出字段包括 query、true_labels、pred_labels_rankings、docs。
    """
    readable_list=[]
    for i,j in zip(examples, df_new["tech_id"].tolist()):
        query=i.query.text
        true_labels = j
        docs=[doc.__dict__ for doc in i.documents]
        pred_labels_ranking = [k.__dict__["metadata"]["docid"] for k in i.documents]
        readable_list.append({"query":query, "true_labels": true_labels,"pred_labels_rankings": pred_labels_ranking, "docs":docs})
    return readable_list

def main():
    """主入口：BM25 召回 + RankGPT 重排 + 结果落盘。"""
    parser = argparse.ArgumentParser(description="Run BM25 and RankGPT ranking pipeline.")
    parser.add_argument("--input_file", required=True, help="Path to the input TSV file with queries.")
    parser.add_argument("--output_file", required=True, help="Path to save the output JSON file.")
    parser.add_argument("--api_key", required=True, help="DeepSeek API key.")
    parser.add_argument("--bm25_cache_file", required=True, help="Path to cache BM25 results.")
    parser.add_argument("--model_name", default="deepseek-chat", help="Model name for RankGPT.")
    parser.add_argument("--bm25_top_k", type=int, default=100, help="Number of documents to retrieve with BM25.")
    parser.add_argument("--rankgpt_rank_end", type=int, default=40, help="Rank end for RankGPT sliding window.")
    parser.add_argument("--rankgpt_window_size", type=int, default=40, help="Window size for RankGPT sliding window.")
    parser.add_argument("--rankgpt_step", type=int, default=20, help="Step for RankGPT sliding window.")
    args = parser.parse_args()
    
    # NLTK 分词资源（某些环境首次运行需要下载）
    nltk.download('punkt')

    print("Loading corpus and summaries...")
    # 1) 读取 MITRE 语料并按技术聚合，作为 BM25 检索库
    dataset = res.load_mitre_kb()
    corpus_df = pd.DataFrame([{
        'tech_id': tech_id,
        'text': name + ' ' + ' '.join(g['text'].values),
        'tech_name': name,
    } for (tech_id, name), g in dataset.groupby(['tech_id', 'tech_name'])])

    # 2) 读取每个技术的摘要文本，供 RankGPT 重排时作为 passage 内容
    json_files = glob.glob("corpus_summaries/*.json")
    all_summaries = {}
    for i in json_files:
        with open(i) as f:
            data_response = json.load(f)
            all_summaries[i.split("/")[-1].split(".json")[0]] = data_response["choices"][0]["message"]["content"]
    
    print(f"Loading queries from {args.input_file}...")
    # 输入 TSV 需要包含 query 与 tech_id 两列
    queries_df = pd.read_csv(args.input_file, sep='\t')
    queries_df['tech_id'] = queries_df['tech_id'].apply(eval)

    texts, _ = rank.get_texts(corpus_df)
    queries = rank.get_queries(queries_df)

    examples = [RelevanceExample(query, texts) for query in queries]

    def stage1_runner(examples):
        # Stage-1：用 BM25 做初始召回
        bm25_reranker = rank.construct_bm25()
        bm25_eval = StepEvaluator(bm25_reranker, [], n_hits=args.bm25_top_k)
        return bm25_eval.evaluate(examples)

    print("Running BM25...")
    bm25_examples, _ = rank.load_cache_or_run(args.bm25_cache_file, stage1_runner, examples=examples)
    
    bm25_results = get_dict_from_examples(bm25_examples, queries_df)

    print("Preparing data for RankGPT...")
    # 3) 组装 RankGPT 输入 item：query + hits(+true_labels)
    all_items = []
    for item in bm25_results:
        query = item['query']
        hits = []
        for doc in item["docs"]:
            doc_id = doc['metadata']['docid']
            if doc_id in all_summaries:
                 hits.append({"content": f"{doc_id}: {all_summaries[doc_id]}"})
        
        rankgpt_item = {"query": query, "hits": hits}
        if 'true_labels' in item:
            rankgpt_item['true_labels'] = item['true_labels']

        all_items.append(rankgpt_item)
        
    print("Running RankGPT...")
    # 4) Stage-2：对 BM25 前若干候选执行滑窗重排
    all_rankgpt_rankings_final = []
    for item in tqdm(all_items):
        new_item = sliding_windows(item, 
                                   rank_start=0, 
                                   rank_end=min(len(item['hits']), args.rankgpt_rank_end), 
                                   window_size=args.rankgpt_window_size, 
                                   step=args.rankgpt_step, 
                                   model_name=args.model_name, 
                                   api_key=args.api_key)
        
        if 'true_labels' in item:
            new_item["true_labels"]=item["true_labels"]
            
        # 从重排后的 hits 中提取技术 ID 顺序，作为最终预测排序
        new_item["pred_labels_rankings"]=extract_technique_ids(new_item["hits"])
        all_rankgpt_rankings_final.append(new_item)

    print(f"Saving results to {args.output_file}...")
    # 5) 输出 JSON：每个 query 一条记录，含最终技术排名
    with open(args.output_file, 'w') as f:
        json.dump(all_rankgpt_rankings_final, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main() 
