# -*- coding: utf-8 -*-
"""
L2分析_联合提取V3.py
功能：对L1分析结果（最满意API分析_V2/分类结果明细.xlsx），按车型分别进行L2级细粒度分析。
       采用联合提取提示词，一次性获得 L1类别、L2具体方面、情感、评分、证据、目标车型。
优化：引入标准三级词库，优先匹配标准L2，后处理相似度合并与频率过滤。
      过滤规则基于L2提及次数占L1总提及次数的比例（≥2%保留）。
输入：合并分析输出的明细Excel（应包含“车型名称”、“类别”、“相关段落”、“是否本车”等列）
      标准三级词库文件：20. 标准三级词库/汽车三级词库_优化版.xlsx
输出：
  - 按车型分文件夹，每个车型下每个类别子文件夹内：统计表、明细表、图表
  - 两个汇总Excel：按车型汇总、按类别汇总
  - 新增评分统计表（整体平均评分、评分分布等）
"""

import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import traceback
import random
from difflib import SequenceMatcher

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 配置 ==========
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("❌ 错误: 未设置环境变量 DEEPSEEK_API_KEY")
    exit(1)

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
    timeout=30
)

MODEL = "deepseek-chat"
MAX_WORKERS = 3
RETRY_TIMES = 3
RETRY_DELAY = 1
REQUEST_DELAY = 0.2

# 输入文件（根据实际路径修改）
INPUT_FILE = "最满意API分析_V2/分类结果明细.xlsx"   # L1分析结果
STANDARD_LEXICON_FILE = "20. 标准三级词库/汽车三级词库_优化版V2.xlsx"  # 标准三级词库
OUTPUT_ROOT = "L2分析_联合提取"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

ANALYZE_BY_MODEL = True
SPECIFIC_CATEGORIES = []
SPECIFIC_MODELS = []
CHECKPOINT_FILE = os.path.join(OUTPUT_ROOT, "completed_units.txt")
BLACKLIST_PHRASES = []

# ========== 联合提取提示词（L1+L2+评分+目标） ==========
SYSTEM_PROMPT = """你是一个汽车评论细粒度分析助手。你的任务是从用户评论中提取用户提到的具体方面（L2 级别），并为每个具体方面判断其所属的 L1 类别、情感倾向、强度评分以及目标车型。

L1 类别列表（请从以下列表中选择一个作为 category）：
- 品牌
- 安全
- 操控
- 动力
- 质量
- 外观
- 舒适
- 使用成本（包含油耗、保养、电费等）
- 购车价格
- 配置
- 空间
- 内饰
- 服务
- 权益
- 智能座舱
- 智能驾驶

对于每个提取的具体方面，你需要：
1. 确定该方面所属的 L1 类别（从上述列表中选择最匹配的一项）。
2. 判断用户对该方面的整体情感，并给出一个强度评分（-2 到 2 之间，精确到 0.5）。
3. 从评论中直接摘录最能支持该情感的原文句子作为证据。
4. 判断该方面描述是针对当前讨论的车型（本车）还是其他车型。

评分定义（强度等级）：
- -2 分：很不满意（强烈负面评价）
- -1 分：不满意（负面评价，程度中等）
- -0.5 分：比较不满意（轻度负面）
- 0 分：中立（无明显倾向，或优缺点平衡）
- 0.5 分：比较满意（轻度正面）
- 1 分：满意（正面评价，程度中等）
- 2 分：很满意（强烈正面评价）

返回格式必须是 JSON 数组，数组中的每个元素是一个对象，包含：
- "category": L1 类别名称（从上述列表中选择）
- "aspect": 具体方面的名称（使用简洁的短语，例如“加速性能”、“内饰材质”、“座椅舒适度”、“后备箱空间”等）
- "sentiment": 情感，只能是 "满意"、"中立" 或 "不满意"（与评分对齐，但作为文字标签）
- "score": 强度评分，必须是 -2, -1, -0.5, 0, 0.5, 1, 2 之一
- "evidence": 支持该情感的原文句子（字符串，直接摘录，保持完整）
- "target": 目标车型，只能是 "this"（本车）或 "other"（其他车）。如果无法明确判断，默认为 "this"

要求：
1. 提取的方面必须属于上述 L1 类别下的具体点，例如：
   - 动力 → “加速性能”、“推背感”、“中后段加速”
   - 操控 → “转向精准”、“悬架支撑”、“车身稳定性”
   - 空间 → “后排腿部空间”、“后备箱容积”、“头部空间”
   - 内饰 → “座椅材质”、“中控屏幕”、“氛围灯”
   - 智能座舱 → “语音响应速度”、“车机流畅度”、“OTA 升级”
   - 等等
2. 如果一个方面在评论中出现多次但情感一致，只需输出一条记录（综合判断情感和评分）；如果同一方面在不同句子中情感不同，应拆分为多条记录。
3. 方面名称应具体、可归类，避免使用 L1 大类名称（如“外观”、“空间”等）。
4. 如果评论中没有提到任何可提取的具体方面，返回空数组 []。
5. 只输出 JSON，不要有任何其他文字（如解释、问候等）。

示例：
用户评论：“这车加速特别快，推背感很强，座椅很舒服，但后排空间有点小，另外车机偶尔会卡顿。”
输出：
[
  {
    "category": "动力",
    "aspect": "加速性能",
    "sentiment": "满意",
    "score": 2,
    "evidence": "这车加速特别快，推背感很强",
    "target": "this"
  },
  {
    "category": "舒适",
    "aspect": "座椅舒适度",
    "sentiment": "满意",
    "score": 1,
    "evidence": "座椅很舒服",
    "target": "this"
  },
  {
    "category": "空间",
    "aspect": "后排空间",
    "sentiment": "不满意",
    "score": -1,
    "evidence": "后排空间有点小",
    "target": "this"
  },
  {
    "category": "智能座舱",
    "aspect": "车机流畅度",
    "sentiment": "不满意",
    "score": -0.5,
    "evidence": "车机偶尔会卡顿",
    "target": "this"
  }
]"""

# ========== 新增：标准L2引导与过滤参数 ==========
SIMILARITY_THRESHOLD = 0.8
MIN_RATIO = 0.02   # 2%
MERGE_LOW_FREQ_TO_OTHER = True
OTHER_NAME = "其他"

# ========== 读取标准三级词库 ==========
print("正在读取标准三级词库...")
if not os.path.exists(STANDARD_LEXICON_FILE):
    print(f"❌ 错误：标准词库文件不存在：{STANDARD_LEXICON_FILE}")
    exit(1)
std_df = pd.read_excel(STANDARD_LEXICON_FILE)
if "一级分类" not in std_df.columns or "二级分类" not in std_df.columns:
    print("错误：标准词库中缺少“一级分类”或“二级分类”列")
    exit(1)

standard_l2 = {}
for _, row in std_df.iterrows():
    l1 = str(row["一级分类"]).strip()
    l2 = str(row["二级分类"]).strip()
    if l1 and l2:
        if l1 not in standard_l2:
            standard_l2[l1] = set()
        standard_l2[l1].add(l2)
for l1 in standard_l2:
    standard_l2[l1] = sorted(standard_l2[l1])
print(f"已加载标准L2，共涉及 {len(standard_l2)} 个一级分类")

# ========== 读取明细数据 ==========
print("正在读取分类明细文件...")
if not os.path.exists(INPUT_FILE):
    print(f"❌ 错误：输入文件不存在：{INPUT_FILE}")
    exit(1)
df = pd.read_excel(INPUT_FILE)
print(f"总记录数: {len(df)}")

if "是否本车" in df.columns:
    df = df[df["是否本车"] == "this"].copy()
    print(f"保留本车记录数: {len(df)}")
else:
    print("警告：明细文件中没有“是否本车”列，将使用全部记录。")

required_cols = ["车型名称", "类别", "相关段落"]
for col in required_cols:
    if col not in df.columns:
        print(f"错误：明细文件中缺少必要列 '{col}'")
        exit(1)

all_models = df["车型名称"].unique().tolist()
all_categories = df["类别"].unique().tolist()
print(f"所有出现的车型: {all_models}")
print(f"所有出现的L1类别: {all_categories}")

if SPECIFIC_MODELS:
    models_to_analyze = [m for m in SPECIFIC_MODELS if m in all_models]
    print(f"将分析指定的车型: {models_to_analyze}")
else:
    models_to_analyze = all_models

if SPECIFIC_CATEGORIES:
    categories_to_analyze = [c for c in SPECIFIC_CATEGORIES if c in all_categories]
    print(f"将分析指定的类别: {categories_to_analyze}")
else:
    categories_to_analyze = all_categories

completed_units = set()
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                completed_units.add(line)
    print(f"已加载 {len(completed_units)} 个已完成组合，将跳过它们。")

# ========== API调用函数（联合提取） ==========
def extract_l1l2_details(paragraph, category_hint):
    """
    对单个段落调用API提取L1+L2详细信息。
    输入：paragraph (str), category_hint (str) 作为上下文提示（实际API会自己判断category）
    返回列表，每个元素为 {"category":..., "aspect":..., "sentiment":..., "score":..., "evidence":..., "target":...}
    """
    if not paragraph or not isinstance(paragraph, str):
        return []
    for phrase in BLACKLIST_PHRASES:
        if phrase in paragraph:
            print(f"段落包含黑名单短语，已跳过: {paragraph[:50]}...")
            return []

    if len(paragraph) > 500:
        paragraph = paragraph[:500] + "……"

    user_prompt = f"请分析以下关于汽车的评论段落。\n段落：{paragraph}"

    for attempt in range(RETRY_TIMES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800,
                timeout=30
            )
            content = response.choices[0].message.content.strip()
            # 清理可能的代码块
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                # 验证必要字段
                validated = []
                for item in data:
                    if all(k in item for k in ("category","aspect","sentiment","score","evidence","target")):
                        # 可选：检查score是否在允许范围内
                        if item["score"] not in [-2,-1,-0.5,0,0.5,1,2]:
                            continue
                        validated.append(item)
                    else:
                        print(f"返回项缺少字段: {item}")
                return validated
            else:
                print(f"返回格式不是列表: {content[:100]}")
                return []
        except json.JSONDecodeError as e:
            print(f"JSON解析失败 (尝试 {attempt+1}/{RETRY_TIMES}): {e}")
            if attempt < RETRY_TIMES - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"API调用出错 (尝试 {attempt+1}/{RETRY_TIMES}): {e}\n{traceback.format_exc()}")
            if attempt < RETRY_TIMES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"段落失败: {paragraph[:30]}...")
                return []
    return []

# ========== 将aspect映射到标准L2（基于相似度） ==========
def map_to_standard(aspect, std_list, threshold=0.8):
    if not std_list:
        return aspect
    best_match = aspect
    best_score = 0.0
    for std_word in std_list:
        score = SequenceMatcher(None, aspect, std_word).ratio()
        if score > best_score:
            best_score = score
            best_match = std_word
    if best_score >= threshold:
        return best_match
    else:
        return aspect

# ========== 合并与过滤函数（基于比例） ==========
def merge_and_filter_aspects(results, category, min_ratio=0.02, similarity_threshold=0.8):
    """
    对提取的L2结果列表进行：
      1. 映射到标准L2（基于相似度）
      2. 基于相似度的全局合并
      3. 基于提及比例（占该L1类别总提及次数）的过滤
    返回更新后的结果列表和统计DataFrame。
    """
    if not results:
        return [], pd.DataFrame()

    df_raw = pd.DataFrame(results)
    total_mentions = len(df_raw)  # 该L1类别下所有L2提及总数（原始记录数）
    std_list = standard_l2.get(category, [])

    # 步骤1：映射到标准词
    df_raw["aspect_mapped"] = df_raw["aspect"].apply(
        lambda x: map_to_standard(x, std_list, SIMILARITY_THRESHOLD)
    )

    # 步骤2：基于相似度的全局合并
    unique_names = df_raw["aspect_mapped"].unique()
    groups = []
    used = set()
    for i, name1 in enumerate(unique_names):
        if name1 in used:
            continue
        group = [name1]
        used.add(name1)
        for j, name2 in enumerate(unique_names[i+1:], start=i+1):
            if name2 in used:
                continue
            sim = SequenceMatcher(None, name1, name2).ratio()
            if sim >= similarity_threshold:
                group.append(name2)
                used.add(name2)
        groups.append(group)

    merge_map = {}
    for group in groups:
        counts = {name: (df_raw["aspect_mapped"] == name).sum() for name in group}
        rep = max(counts.items(), key=lambda x: x[1])[0]
        for name in group:
            merge_map[name] = rep

    df_raw["aspect_final"] = df_raw["aspect_mapped"].map(merge_map)

    # 步骤3：基于占比的过滤
    final_counts = df_raw["aspect_final"].value_counts()
    ratios = final_counts / total_mentions
    keep_names = set(ratios[ratios >= min_ratio].index)

    if MERGE_LOW_FREQ_TO_OTHER:
        df_raw["aspect_final"] = df_raw["aspect_final"].apply(
            lambda x: x if x in keep_names else OTHER_NAME
        )
    else:
        df_raw = df_raw[df_raw["aspect_final"].isin(keep_names)]

    # 构建最终结果列表
    new_results = []
    for _, row in df_raw.iterrows():
        new_item = row.to_dict()
        new_item["aspect"] = row["aspect_final"]
        new_item["original_aspect"] = row["aspect"]  # 保留原始名称供追溯
        new_results.append(new_item)

    # 统计
    final_stats = df_raw.groupby(["aspect_final", "sentiment"]).size().unstack(fill_value=0)
    final_stats["总提及"] = final_stats.sum(axis=1)
    final_stats = final_stats.sort_values("总提及", ascending=False)

    return new_results, final_stats

# ========== 根据模式选择分析主体 ==========
if ANALYZE_BY_MODEL:
    analysis_units = []
    for model in models_to_analyze:
        model_df = df[df["车型名称"] == model]
        for cat in categories_to_analyze:
            unit_key = f"{model}||{cat}"
            if unit_key in completed_units:
                print(f"跳过已完成的组合: {model} - {cat}")
                continue
            cat_df = model_df[model_df["类别"] == cat]
            paragraphs = cat_df["相关段落"].dropna().tolist()
            paragraphs = list(set(paragraphs))
            if paragraphs:
                analysis_units.append((model, cat, paragraphs))
    print(f"待分析的 (车型, 类别) 组合数: {len(analysis_units)}")
else:
    analysis_units = []
    for cat in categories_to_analyze:
        cat_df = df[df["类别"] == cat]
        paragraphs = cat_df["相关段落"].dropna().tolist()
        paragraphs = list(set(paragraphs))
        if paragraphs:
            analysis_units.append(("全部车型", cat, paragraphs))
    print(f"待分析的类别数（合并车型）: {len(analysis_units)}")

all_l2_results = []  # 存储所有提取结果（含评分）
checkpoint_fp = open(CHECKPOINT_FILE, "a", encoding="utf-8")
total_pbar = tqdm(total=len(analysis_units), desc="整体进度", unit="组合")

for model, cat, paragraphs in analysis_units:
    unit_key = f"{model}||{cat}"
    print(f"\n{'='*50}")
    print(f"开始处理: 车型 = {model}, 类别 = {cat}")
    print(f"共有 {len(paragraphs)} 条去重后的相关段落")

    if ANALYZE_BY_MODEL:
        unit_dir = os.path.join(OUTPUT_ROOT, model, cat)
    else:
        unit_dir = os.path.join(OUTPUT_ROOT, "全部车型", cat)
    os.makedirs(unit_dir, exist_ok=True)

    l1l2_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_para = {}
        for para in paragraphs:
            time.sleep(random.uniform(0.05, 0.15))
            future = executor.submit(extract_l1l2_details, para, cat)
            future_to_para[future] = para

        for future in tqdm(concurrent.futures.as_completed(future_to_para),
                           total=len(future_to_para),
                           desc=f"{model}-{cat} 处理进度"):
            para = future_to_para[future]
            try:
                res = future.result(timeout=60)
                l1l2_results.extend(res)
            except concurrent.futures.TimeoutError:
                print(f"\n段落处理超时: {para[:50]}...")
            except Exception as e:
                print(f"\n段落处理异常: {para[:50]}... 错误: {e}\n{traceback.format_exc()}")

    print(f"提取到原始 L2 记录数: {len(l1l2_results)}")
    if not l1l2_results:
        print(f"该组合未提取到任何L2，跳过统计")
        checkpoint_fp.write(unit_key + "\n")
        checkpoint_fp.flush()
        total_pbar.update(1)
        continue

    # ========== 应用标准映射与过滤（基于比例） ==========
    merged_results, final_stats = merge_and_filter_aspects(
        l1l2_results,
        category=cat,
        min_ratio=MIN_RATIO,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    print(f"合并过滤后得到 {len(merged_results)} 条记录，涉及 {len(final_stats)} 个子维度")

    # 转换为DataFrame，添加车型信息
    l2_df = pd.DataFrame(merged_results)
    l2_df["车型"] = model
    # 可选：验证category是否与输入一致，不一致时可记录或修正
    # 这里简单保留API返回的category
    all_l2_results.append(l2_df)

    # 保存明细（含证据和原始aspect、评分、目标）
    l2_df.to_excel(os.path.join(unit_dir, f"{cat}_L2明细.xlsx"), index=False)

    # 准备统计表（包含评分信息）
    sentiment_cols = ["满意", "中立", "不满意"]
    for col in sentiment_cols:
        if col not in final_stats.columns:
            final_stats[col] = 0

    final_stats["满意率"] = (final_stats["满意"] / final_stats["总提及"] * 100).round(1)
    final_stats["中立率"] = (final_stats["中立"] / final_stats["总提及"] * 100).round(1)
    final_stats["不满意率"] = (final_stats["不满意"] / final_stats["总提及"] * 100).round(1)
    total_mentions_cat = final_stats["总提及"].sum()
    final_stats["提及占比"] = (final_stats["总提及"] / total_mentions_cat * 100).round(1)
    final_stats["提及率(占L1段落)"] = (final_stats["总提及"] / len(paragraphs) * 100).round(1)

    # 新增：平均评分
    # 计算每个aspect的平均score
    avg_score = l2_df.groupby("aspect_final")["score"].mean().round(2)
    final_stats["平均评分"] = avg_score.reindex(final_stats.index, fill_value=0)
    # 评分分布（可加一列）
    # 这里简单添加满意评分占比（score>0）和负面评分占比（score<0）
    positive_count = l2_df[l2_df["score"] > 0].groupby("aspect_final").size()
    negative_count = l2_df[l2_df["score"] < 0].groupby("aspect_final").size()
    final_stats["正面评分占比(%)"] = (positive_count.reindex(final_stats.index, fill_value=0) / final_stats["总提及"] * 100).round(1)
    final_stats["负面评分占比(%)"] = (negative_count.reindex(final_stats.index, fill_value=0) / final_stats["总提及"] * 100).round(1)

    # 保存统计表
    final_stats.to_excel(os.path.join(unit_dir, f"{cat}_L2维度统计.xlsx"))

    # ========== 可视化（增加平均评分图） ==========
    # 1. 提及次数TOP10柱状图
    top10 = final_stats.head(10).copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bars1 = ax1.barh(top10.index, top10["总提及"], color="steelblue")
    ax1.set_xlabel("提及次数")
    ax1.set_title(f"{model} - {cat} L2子维度提及次数 TOP10")
    ax1.invert_yaxis()
    for bar, val in zip(bars1, top10["总提及"]):
        ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, str(val), va="center")
    bars2 = ax2.barh(top10.index, top10["提及占比"], color="coral")
    ax2.set_xlabel("提及占比 (%)")
    ax2.set_title(f"{model} - {cat} L2子维度提及占比 TOP10")
    ax2.invert_yaxis()
    for bar, val in zip(bars2, top10["提及占比"]):
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f"{val}%", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(unit_dir, f"{cat}_提及TOP10.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. 情感分布堆叠图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    top10_sent = top10[sentiment_cols]
    top10_sent.plot(kind="bar", stacked=True, ax=ax1, color=["#4CAF50", "#FFC107", "#F44336"])
    ax1.set_xlabel("子维度")
    ax1.set_ylabel("提及次数")
    ax1.set_title(f"{model} - {cat} L2情感分布（绝对频数）")
    ax1.legend(title="情感")
    ax1.tick_params(axis="x", rotation=45)
    top10_pct = top10_sent.div(top10["总提及"], axis=0) * 100
    top10_pct.plot(kind="bar", stacked=True, ax=ax2, color=["#4CAF50", "#FFC107", "#F44336"])
    ax2.set_xlabel("子维度")
    ax2.set_ylabel("占比 (%)")
    ax2.set_title(f"{model} - {cat} L2情感分布（百分比）")
    ax2.legend(title="情感")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(unit_dir, f"{cat}_情感分布.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. 新增：平均评分柱状图
    if not top10.empty and "平均评分" in top10.columns:
        plt.figure(figsize=(10, 6))
        top10_avg = top10.sort_values("平均评分", ascending=False)
        bars = plt.barh(top10_avg.index, top10_avg["平均评分"], color="skyblue")
        plt.xlabel("平均评分")
        plt.title(f"{model} - {cat} L2子维度平均评分")
        for bar, val in zip(bars, top10_avg["平均评分"]):
            plt.text(val + 0.05, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(unit_dir, f"{cat}_平均评分.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"组合 {model}-{cat} 分析完成，结果保存至 {unit_dir}")

    checkpoint_fp.write(unit_key + "\n")
    checkpoint_fp.flush()
    total_pbar.update(1)

total_pbar.close()
checkpoint_fp.close()

# ========== 汇总所有L2结果 ==========
if all_l2_results:
    all_l2_df = pd.concat(all_l2_results, ignore_index=True)

    # 按车型汇总
    with pd.ExcelWriter(os.path.join(OUTPUT_ROOT, "L2_分析结果汇总_按车型.xlsx"), engine="openpyxl") as writer:
        for model in models_to_analyze:
            model_df = all_l2_df[all_l2_df["车型"] == model]
            if model_df.empty:
                continue
            model_stats = model_df.groupby(["category", "aspect", "sentiment"]).size().unstack(fill_value=0)
            model_stats["总提及"] = model_stats.sum(axis=1)
            # 添加平均评分
            avg_score_model = model_df.groupby(["category", "aspect"])["score"].mean().round(2)
            model_stats["平均评分"] = avg_score_model.reindex(model_stats.index, fill_value=0)
            model_stats = model_stats.reset_index()
            model_stats = model_stats.sort_values(["category", "总提及"], ascending=[True, False])
            sheet_name = model[:31]
            model_stats.to_excel(writer, sheet_name=sheet_name, index=False)

    # 按类别汇总
    with pd.ExcelWriter(os.path.join(OUTPUT_ROOT, "L2_分析结果汇总_按类别.xlsx"), engine="openpyxl") as writer:
        for cat in categories_to_analyze:
            cat_df = all_l2_df[all_l2_df["category"] == cat]
            if cat_df.empty:
                continue
            cat_stats = cat_df.groupby(["车型", "aspect", "sentiment"]).size().unstack(fill_value=0)
            cat_stats["总提及"] = cat_stats.sum(axis=1)
            avg_score_cat = cat_df.groupby(["车型", "aspect"])["score"].mean().round(2)
            cat_stats["平均评分"] = avg_score_cat.reindex(cat_stats.index, fill_value=0)
            cat_stats = cat_stats.reset_index()
            cat_stats = cat_stats.sort_values(["车型", "总提及"], ascending=[True, False])
            sheet_name = cat[:31]
            cat_stats.to_excel(writer, sheet_name=sheet_name, index=False)

    # 保存全量明细（含评分）
    all_l2_df.to_excel(os.path.join(OUTPUT_ROOT, "L2_全量明细.xlsx"), index=False)

    # 额外：整体评分分布统计
    score_dist = all_l2_df["score"].value_counts().sort_index()
    score_dist.to_excel(os.path.join(OUTPUT_ROOT, "评分分布.xlsx"), header=["频数"])

    print(f"\n✅ 汇总文件已生成：")
    print(f"   - {os.path.join(OUTPUT_ROOT, 'L2_分析结果汇总_按车型.xlsx')}")
    print(f"   - {os.path.join(OUTPUT_ROOT, 'L2_分析结果汇总_按类别.xlsx')}")
    print(f"   - {os.path.join(OUTPUT_ROOT, 'L2_全量明细.xlsx')}")
    print(f"   - {os.path.join(OUTPUT_ROOT, '评分分布.xlsx')}")
else:
    print("未提取到任何L2结果，未生成汇总文件。")

print("\n所有L2联合提取分析完成！")