import json
import re
import os
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent import futures
from datetime import datetime
from dataclasses import dataclass

from openai import AzureOpenAI, OpenAI

# 配置命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="IntelEx Pipeline for MITRE Technique Extraction and Validation")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with sentences to process")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for results")
    parser.add_argument("--kb", type=str, help="Knowledge base JSON file with MITRE techniques",
                        default="./assets/mitre_kb.json")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for validation")
    parser.add_argument("--api", type=str, choices=["azure", "openai", "local"], default="local",
                        help="API to use (azure, openai, or local)")
    parser.add_argument("--threads", type=int, default=10, help="Number of concurrent threads")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for LLM calls")
    parser.add_argument("--max-tokens", type=int, default=151, help="Max tokens for LLM response")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--continue-from", action="store_true", help="Continue from existing output file")
    parser.add_argument("--local-url", type=str, default="http://localhost:9002/v1", help="URL for local LLM server")
    return parser.parse_args()

# 配置日志（控制台 + 文件）
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging with file and console output"""
    log_filename = f"intelex_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Map string log level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    level = level_map.get(log_level, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """流水线运行参数。"""
    input_file: str
    output_file: str 
    kb_file: str
    model: str
    api_type: str
    threads: int
    temperature: float
    max_tokens: int
    continue_from: bool
    local_url: str

class LLMClient:
    """统一封装 Azure/OpenAI/本地兼容接口。"""
    
    def __init__(self, api_type: str, model: str, config: PipelineConfig, logger: logging.Logger):
        self.api_type = api_type
        self.model = model
        self.config = config
        self.logger = logger
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> Union[AzureOpenAI, OpenAI]:
        """根据 api_type 初始化不同客户端。"""
        if self.api_type == "azure":
            api_key = os.getenv("AZURE_OPENAI_KEY")
            endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
            
            if not api_key or not endpoint:
                raise ValueError("AZURE_OPENAI_KEY and AZURE_OPENAI_API_ENDPOINT environment variables must be set")
                
            self.logger.info(f"Initializing Azure OpenAI client with endpoint: {endpoint}")
            return AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )
        elif self.api_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set")
                
            self.logger.info("Initializing OpenAI client")
            return OpenAI(api_key=api_key)
        elif self.api_type == "local":
            self.logger.info(f"Initializing local LLM client with URL: {self.config.local_url}")
            return OpenAI(
                base_url=self.config.local_url,
                api_key="not-needed"
            )
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")
    
    def validate_technique(self, text: str, technique: str, description: str) -> bool:
        """让 LLM 判定某个 technique 是否在文本中出现。"""
        start_time = time.time()
        
        background = """Background:
You are a helpful assistant for a cybersecurity analyst.
You will be given a cyber threat intelligence (CTI) report.
"""
        task = f"""Task:
The user will provide another possible technique and description. Your task is to determine whether the technique exists in
the report.
Description: {description}
"""

        guidelines = """Guidelines:
Below is the instruction to finish the task:
- You need to verify whether the technique exists in the report.
- If the technique exists in the report, you need to output YES and the reason.
- If the technique does not exist in the report, you need to output NO and the reason.
- The output needs to be in JSON format. The output format is as follows:
{"if_exist": "YES/NO", "reason": "REASON in 20 words"}"""

        technique_text = f"Technique: {technique}"
        prompt = f"{background}\n{task}\n{guidelines}\n{technique_text}\nText: {text}\n"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            response_content = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            self.logger.debug(f"LLM call for technique {technique} completed in {elapsed_time:.2f}s")
            
            # 这里用简单字符串规则解析结果，鲁棒性一般但速度快
            if "YES" in response_content and "NO" not in response_content:
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error in LLM call for technique {technique}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise

class MitreTechniqueExtractor:
    """MITRE 技术候选抽取 + 基于 KB 描述的二次验证。"""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.mitre_kb = self._load_knowledge_base()
        self.llm_client = LLMClient(
            api_type=config.api_type,
            model=config.model,
            config=config,
            logger=logger
        )
        
    def _load_knowledge_base(self) -> Dict:
        """加载 MITRE 知识库 JSON。"""
        try:
            with open(self.config.kb_file, "r") as fp:
                kb = json.load(fp)
                self.logger.info(f"Loaded MITRE KB with {len(kb)} techniques")
                return kb
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {str(e)}")
            raise
            
    def extract_techniques(self, text: str) -> List[str]:
        """用正则抽取文本中的潜在技术 ID。"""
        mitre_pattern = r'T\d{4}(?:\.\d{3})?'
        matches = re.findall(mitre_pattern, text)
        unique_matches = list(set(matches))
        self.logger.debug(f"Extracted {len(unique_matches)} unique MITRE techniques from text")
        return unique_matches
    
    def validate_technique(self, text: str, technique: str) -> Optional[str]:
        """结合 KB 描述与 LLM 判定该技术是否真实存在。"""
        try:
            if technique not in self.mitre_kb:
                self.logger.warning(f"Technique {technique} not found in knowledge base")
                return None
                
            # 使用该技术在 KB 中的描述，辅助 LLM 做语义匹配判断
            description = self.mitre_kb[technique]['description']
            is_valid = self.llm_client.validate_technique(text, technique, description)
            
            if is_valid:
                self.logger.info(f"Technique {technique} validated as present")
                return technique
                
            self.logger.debug(f"Technique {technique} not found in text")
            return None
            
        except Exception as e:
            self.logger.error(f"Error validating technique {technique}: {str(e)}")
            return None
            
    def process_technique_item(self, args: Tuple[str, str]) -> Optional[str]:
        """并发 worker：处理单个 technique 的验证任务。"""
        text, technique = args
        return self.validate_technique(text, technique)

class IntelExPipeline:
    """
    IntelEx 主流程控制器：读入 -> 生成候选 -> 并发验证 -> 输出。

    该类对外最关键的方法是 `run()`，其职责是：
    1) 读取输入样本并转换为内部统一结构；
    2) 合并候选技术（上游预测 + instruction 正则抽取）；
    3) 并发调用验证器过滤伪阳性；
    4) 以 JSONL 方式落盘，支持中断后续跑。
    """
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        """
        初始化流水线实例。

        :param config: 运行配置（输入输出路径、模型、并发数等）
        :param logger: 日志对象
        :return: None
        """
        self.config = config
        self.logger = logger
        self.extractor = MitreTechniqueExtractor(config, logger)
        
    def _load_input_data(self) -> List[Dict]:
        """
        读取输入 JSON，并转换成流水线内部使用的统一结构。

        输入文件中每个样本通常包含：
        - instruction: 包含提示词/上下文（可能含 MITRE ID）
        - input: 原始 CTI 句子
        - gold: 标注技术列表
        - predicted: 上游模型给出的候选技术（可为空）

        转换后每条样本结构：
        - sentence: 原始待判定文本（来自 input）
        - gold: 标注真值
        - gpt4_techs: 来自 predicted 的候选技术
        - rag_techs: 从 instruction 正则抽取得到的候选技术

        :return: List[Dict]，样例：
        [
          {
            "sentence": "The malware uses PowerShell to fetch payloads.",
            "gold": ["T1059.001", "T1105"],
            "gpt4_techs": ["T1059.001"],
            "rag_techs": ["T1059.001", "T1105", "T1027"]
          }
        ]
        """
        try:
            with open(self.config.input_file, "r") as f:
                sentences = json.load(f)
            self.logger.info(f"Loaded {len(sentences)} sentences for processing")
            
            raw_sentences = []
            for sentence in sentences:
                if sentence['predicted'] is None:
                    sentence['predicted'] = []
                    
                obj = {
                    "sentence": sentence['input'],
                    "gold": sentence['gold'],
                    # gpt4_techs: 已有预测候选（来自上游）
                    "gpt4_techs": sentence['predicted'],
                    # rag_techs: 从 instruction 中正则抽取到的候选
                    "rag_techs": self.extractor.extract_techniques(text=sentence['instruction']),
                }
                raw_sentences.append(obj)
                
            return raw_sentences
            
        except Exception as e:
            self.logger.error(f"Error loading input file: {str(e)}")
            raise
            
    def _load_processed_data(self) -> List[str]:
        """
        从输出 JSONL 中读取已经处理过的 sentence，用于 `--continue-from` 断点续跑。

        读取逻辑：
        - 若未开启 continue_from，或输出文件不存在，直接返回空列表；
        - 否则逐行解析 JSONL，提取每行的 sentence 字段。

        :return: 已处理 sentence 列表，样例：
        [
          "The malware uses PowerShell to fetch payloads.",
          "Attackers modified registry keys for persistence."
        ]
        """
        processed = []
        
        if not self.config.continue_from or not os.path.exists(self.config.output_file):
            return processed
            
        try:
            with open(self.config.output_file, "r") as f:
                for line in f:
                    try:
                        processed.append(json.loads(line)['sentence'])
                    except json.JSONDecodeError:
                        self.logger.warning(f"Could not parse line in output file: {line}")
                        
            self.logger.info(f"Loaded {len(processed)} already processed sentences")
            return processed
            
        except Exception as e:
            self.logger.error(f"Error loading processed data: {str(e)}")
            return []
            
    def run(self) -> None:
        """
        执行整条 IntelEx 流水线并将结果写入 JSONL。

        处理步骤：
        1) 加载并规范化输入样本；
        2) 读取已处理样本（可选，断点续跑）；
        3) 对每条样本合并候选技术并并发验证；
        4) 将每条结果追加写入输出文件。

        注意：本函数返回值为 None，结果通过文件副作用产出。

        输出文件每行样例（JSONL）：
        {"sentence":"The malware uses PowerShell to fetch payloads.","gold":["T1059.001","T1105"],"predicted":["T1059.001","T1105"]}
        """
        start_time = time.time()
        self.logger.info("Starting IntelEx Pipeline")
        
        try:
            # Load input data
            raw_sentences = self._load_input_data()
            
            # Load already processed data if continuing
            processed = self._load_processed_data()
            
            # 逐句处理；句内 technique 验证会并发执行
            processed_count = 0
            with futures.ThreadPoolExecutor(max_workers=self.config.threads) as executor:
                self.logger.info(f"Started ThreadPoolExecutor with {self.config.threads} workers")
                
                for sentence in raw_sentences:
                    if sentence['sentence'] in processed:
                        self.logger.debug(f"Skipping already processed sentence")
                        continue
                        
                    # 合并两路候选：上游预测 + instruction 正则抽取
                    techs = list(set(sentence['gpt4_techs']).union(set(sentence['rag_techs'])))
                    self.logger.info(f"Processing sentence with {len(techs)} unique techniques")
                    
                    # 构造并发任务参数（同一句子，不同 technique）
                    args_list = [(sentence['sentence'], tech) for tech in techs]
                    
                    # 并发验证每个候选技术
                    results = list(executor.map(self.extractor.process_technique_item, args_list))
                    
                    # 过滤掉验证失败/异常项，仅保留确认存在的技术
                    valid_techs = [tech for tech in results if tech is not None]
                    self.logger.info(f"Found {len(valid_techs)} valid techniques out of {len(techs)} candidates")
                    
                    # Prepare output object
                    obj = {
                        "sentence": sentence['sentence'],
                        "gold": sentence['gold'],
                        "predicted": valid_techs
                    }
                    
                    # 以 JSONL 逐行追加，便于中断恢复
                    try:
                        with open(self.config.output_file, "a") as f:
                            f.write(json.dumps(obj) + "\n")
                        processed_count += 1
                        self.logger.debug(f"Written results for sentence {processed_count}")
                    except Exception as e:
                        self.logger.error(f"Error writing results to output file: {str(e)}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Pipeline completed. Processed {processed_count} sentences in {elapsed_time:.2f}s")
            self.logger.info(f"Results written to {self.config.output_file}")
            
        except Exception as e:
            self.logger.error(f"Error running pipeline: {str(e)}")
            raise

def main():
    """程序入口。"""
    args = parse_args()
    logger = setup_logging(args.log_level)
    
    # 汇总参数为配置对象，便于在各模块间传递
    config = PipelineConfig(
        input_file=args.input,
        output_file=args.output,
        kb_file=args.kb,
        model=args.model,
        api_type=args.api,
        threads=args.threads,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        continue_from=args.continue_from,
        local_url=args.local_url
    )
    
    # 初始化并执行
    pipeline = IntelExPipeline(config, logger)
    pipeline.run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Get logger directly since we might be outside the normal flow
        logger = logging.getLogger(__name__)
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        exit(1)
