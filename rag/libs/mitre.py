import os
import json
import pandas as pd


# 默认路径来自历史实验环境，当前仓库主流程一般不直接依赖这里
default_data_dir = '/home/local/QCRI/ukumarasinghe/projects/CTI2AttackMetrix/data/mitre'


def list_tactics(data_dir=default_data_dir):
    """
    列出 MITRE tactic 文件路径

    :param data_dir: Directory of scraped data
    :return: `list` of tactics
    """
    # 注意：变量名 `scapred_path` 是历史拼写，保持兼容
    scapred_path = os.path.join(data_dir, 'scraped')
    return [os.path.join(scapred_path, f) for f in os.listdir(scapred_path) if '.json' in f and 'TA' in f]


def list_techniques(data_dir=default_data_dir):
    """
    列出 MITRE technique 文件路径

    :param data_dir: Directory of scraped data
    :return: `list` of techniques
    """
    scapred_path = os.path.join(data_dir, 'scraped')
    return [os.path.join(scapred_path, f) for f in os.listdir(scapred_path) if '.json' in f and not 'TA' in f]


def load_technique_map(data_dir=default_data_dir, full_name=False):
    """
    加载 MITRE 技术 ID 到技术名称的映射

    :param data_dir: Directory of scraped data
    :param full_name: Include the parent technique name in sub-techniques
    :return: `dict` with contents of MITRE techinque mapping
    """    
    techniques = list_techniques(data_dir)

    tech_map = {}
    for tech_file in techniques:
        tech = load_technique_file(tech_file)
        tech_map[tech['ID']] = tech['Name']

    # 手工补充个别旧版本数据里缺失的映射项
    tech_map['T1521'] = 'Encrypted Channel'
    tech_map['T1533'] = 'Data from Local System'
    tech_map['T1218'] = 'System Binary Proxy Execution'
    tech_map['T1053.001'] = 'At'
    
    if full_name:
        # 子技术显示为“父技术: 子技术”的全名格式
        for k, v in tech_map.items():
            tech_map[k] = v if len(k) == 5 else f"{tech_map[k[:5]]}: {v}"

    return tech_map

    
def load_technique(tech_id, data_dir=default_data_dir):
    """
    加载单个 MITRE technique 详情

    :param tech_id: ID of the MITRE technique
    :param data_dir: Directory of scraped data
    :return: `dict` with contents of MITRE resource
    """    
    return load_technique_file(os.path.join(data_dir, f'{tech_id}.json'), 'r')


def load_technique_file(file_path):
    """
    从 JSON 文件读取 technique 详情

    :param filepath: Path to file containing MITRE technique
    :return: `dict` with contents of MITRE resource
    """
    with open(file_path, 'r') as f:
        technique = json.load(f)

    return technique


def load_sources(data_dir=default_data_dir):
    """
    加载 MITRE 引用来源表（CSV）

    :param data_dir: Directory of scraped data
    :return: `pd.DataFrame` containing MITRE references
    """    
    return pd.read_csv(os.path.join(data_dir, 'meta_references.csv'))
