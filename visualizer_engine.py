# visualizer_engine.py
# -*- coding: utf-8 -*-
"""
可视化分析引擎：读取分析结果 Excel，对使用场景、满意点、不满意点、改进建议、对比车型
进行统计和简单的文本聚类，输出适合前端 ECharts 渲染的 JSON 数据。
"""

import pandas as pd
from collections import Counter
import difflib
import re


def cluster_texts(texts, threshold=0.6):
    """
    用相似度对文本进行简单聚类。
    返回字典 {类代表文本: 出现次数}
    """
    clusters = []  # 每个元素是一个列表，存储属于同一类的所有文本
    for text in texts:
        text = text.strip()
        if not text:
            continue
        best_ratio = 0
        best_idx = -1
        for i, cluster in enumerate(clusters):
            ratio = difflib.SequenceMatcher(None, cluster[0], text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
        if best_ratio >= threshold and best_idx != -1:
            clusters[best_idx].append(text)
        else:
            clusters.append([text])

    result = {}
    for cluster in clusters:
        # 选择出现次数最多的文本作为类名
        counter = Counter(cluster)
        most_common = counter.most_common(1)[0][0]
        result[most_common] = len(cluster)
    return result


def generate_visualization(excel_path):
    """
    读取分析结果 Excel，返回用于前端的图表数据。
    """
    # 读取各个 Sheet
    df_scenes = pd.read_excel(excel_path, sheet_name='使用场景')
    df_satis = pd.read_excel(excel_path, sheet_name='满意点')
    df_dissatis = pd.read_excel(excel_path, sheet_name='不满意点')
    df_suggest = pd.read_excel(excel_path, sheet_name='改进建议')
    df_compare = pd.read_excel(excel_path, sheet_name='对比车型')

    # 1. 使用场景：聚类后统计
    scenes_raw = df_scenes['使用场景'].dropna().astype(str).tolist()
    if scenes_raw:
        scenes_clustered = cluster_texts(scenes_raw, threshold=0.55)
        # 按次数排序，取前15
        scenes_sorted = sorted(scenes_clustered.items(), key=lambda x: x[1], reverse=True)[:15]
        scenes_data = [{"name": k, "value": v} for k, v in scenes_sorted]
    else:
        scenes_data = []

    # 2. 满意领域统计
    satis_domain = df_satis['满意领域'].dropna().astype(str).tolist()
    satis_count = Counter(satis_domain).most_common(15)
    satisfactions_data = [{"name": k, "value": v} for k, v in satis_count]

    # 3. 不满意领域统计
    dis_domain = df_dissatis['不满意领域'].dropna().astype(str).tolist()
    dis_count = Counter(dis_domain).most_common(15)
    dissatisfactions_data = [{"name": k, "value": v} for k, v in dis_count]

    # 4. 改进建议：改进领域统计
    suggest_domain = df_suggest['改进领域'].dropna().astype(str).tolist()
    suggest_domain_count = Counter(suggest_domain).most_common(15)
    suggestions_data = [{"name": k, "value": v} for k, v in suggest_domain_count]

    # 5. 对比车型统计
    compare_models = df_compare['对比车型名称'].dropna().astype(str).tolist()
    compare_count = Counter(compare_models).most_common(15)
    comparisons_data = [{"name": k, "value": v} for k, v in compare_count]

    return {
        "scenes": scenes_data,
        "satisfactions": satisfactions_data,
        "dissatisfactions": dissatisfactions_data,
        "suggestions": suggestions_data,
        "comparisons": comparisons_data
    }


if __name__ == "__main__":
    # 测试
    test_excel = "outputs/理想i6_8183_分析结果.xlsx"
    import os
    if os.path.exists(test_excel):
        data = generate_visualization(test_excel)
        import json
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print("测试文件不存在")