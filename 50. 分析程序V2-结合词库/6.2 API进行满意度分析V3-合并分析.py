import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 配置 ==========
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("❌ 错误: 未设置环境变量 DEEPSEEK_API_KEY")
    exit(1)

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

MODEL = "deepseek-chat"
MAX_WORKERS = 10
RETRY_TIMES = 3
RETRY_DELAY = 1
REQUEST_DELAY = 0.1

# 类别列表
CATEGORIES = [
    "品牌", "安全", "操控", "动力", "质量", "外观", "舒适",
    "使用成本", "购车价格", "配置", "空间", "内饰", "服务",
    "权益", "智能座舱", "智能驾驶"
]

ALLOWED_SCORES = {-2, -1, -0.5, 0, 0.5, 1, 2}

# ========== 2. 系统提示词（与原程序完全相同）==========
system_prompt = """
你是一个汽车评论分析助手。你的任务是从用户评论中提取出用户提到的产品/服务类别，判断用户对该类别的情感倾向，并给出一个强度评分（-2到2之间的值，精确到0.5），同时提供支持该情感的原文句子。还需要判断该类别描述是针对当前讨论的车型（本车）还是其他车型，以及该体验类型是购车前就能感受到的还是需要长期使用后才能体会到的，并给出简要的分类理由。

类别列表：
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

评分定义（强度等级）：
- -2 分：很不满意（强烈负面评价）
- -1 分：不满意（负面评价，程度中等）
- -0.5 分：比较不满意（轻度负面）
- 0 分：中立（无明显倾向，或优缺点平衡）
- 0.5 分：比较满意（轻度正面）
- 1 分：满意（正面评价，程度中等）
- 2 分：很满意（强烈正面评价）

体验类型定义：
- "pre-purchase"：指用户在购车前可以通过静态体验、试驾、展厅查看等方式直接感受到的方面。例如：外观、内饰、空间、配置、品牌、购车价格、试驾时的动力/操控初步感受等。
- "long-term"：指需要用户在实际使用一段时间后才能有深刻体验的方面。例如：油耗/使用成本、质量可靠性、售后服务、智能驾驶/座舱的日常便利性、长期舒适性、动力平顺性/可靠性、操控的稳定性等。
如果难以明确判断，默认为 "long-term"。

请分析评论，找出评论中提到的类别。对于每个提到的类别，执行以下操作：
1. 判断用户对该类别的整体情感，并选择最匹配的评分（-2 到 2 之间，可以取 0.5 的倍数）。
2. 从评论中直接摘录最能支持该情感的原文句子作为证据，要求句子完整且直接体现情感。
3. 添加字段 "target"，值为 "this"（本车）或 "other"（其他车）。如果无法明确判断，默认为 "this"。
4. 添加字段 "experience_type"，值为 "pre-purchase" 或 "long-term"。
5. 添加字段 "experience_reason"，简要说明为什么将该方面归类为 "pre-purchase" 或 "long-term"。

返回格式必须是 JSON 数组，数组中的每个元素是一个对象，包含：
- "category": 类别名称（从上面列表中选择）
- "sentiment": 情感，只能是 "满意"、"中立" 或 "不满意"（与评分对齐，但作为文字标签）
- "score": 强度评分，必须是 -2, -1, -0.5, 0, 0.5, 1, 2 之一
- "evidence": 支持该情感的原文句子（字符串）
- "target": 目标车型，只能是 "this" 或 "other"
- "experience_type": 体验类型，只能是 "pre-purchase" 或 "long-term"
- "experience_reason": 体验类型分类的理由（字符串）

如果评论中没有提到任何类别，返回空数组 []。

注意：
- 同一个类别可能被多次提及，但只需在结果中包含一次（综合判断情感和评分）。
- 只输出 JSON，不要有任何其他文字（如解释、问候等）。
- 请确保 category 准确匹配列表中的名称，score 必须是允许的值之一，evidence 必须原样引用。
"""

# ========== 3. 输出目录 ==========
OUTPUT_DIR = "最满意API分析_V2"          # 可自行修改为 "满意度分析_V2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_EXCEL = "../10.汽车口碑数据/merged_autohome_reviews_V2.xlsx"
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "满意度分析结果.xlsx")   # 修改了文件名
DETAIL_EXCEL = os.path.join(OUTPUT_DIR, "分类结果明细.xlsx")
SHEET_NAME = "merged_autohome_reviews"

# ========== 4. 读取数据 + 合并“最满意”与“最不满意” ==========
print("正在读取数据...")
df = pd.read_excel(INPUT_EXCEL, sheet_name=SHEET_NAME)
print(f"原始总记录数: {len(df)}")

# "最满意", "最不满意", "空间评论", "驾驶感受评论", "续航评论",
#         "外观评论", "内饰评论", "性价比评论", "智能化评论",
#         "油耗评论", "配置评论"



# 合并两列：将“最满意”和“最不满意”等列拼接成一个完整的评论文本
def merge_satisfaction(row):
    parts = []
    if pd.notna(row["最满意"]) and str(row["最满意"]).strip():
        parts.append(f"【满意部分】\n{row['最满意']}")
    if pd.notna(row["最不满意"]) and str(row["最不满意"]).strip():
        parts.append(f"【不满意部分】\n{row['最不满意']}")
    if pd.notna(row["空间评论"]) and str(row["空间评论"]).strip():
        parts.append(f"【空间评论】\n{row['最不满意']}")
    if pd.notna(row["驾驶感受评论"]) and str(row["驾驶感受评论"]).strip():
        parts.append(f"【驾驶感受评论】\n{row['驾驶感受评论']}")
    if pd.notna(row["续航评论"]) and str(row["续航评论"]).strip():
        parts.append(f"【续航评论】\n{row['续航评论']}")
    if pd.notna(row["外观评论"]) and str(row["外观评论"]).strip():
        parts.append(f"【外观评论】\n{row['外观评论']}")
    if pd.notna(row["内饰评论"]) and str(row["内饰评论"]).strip():
        parts.append(f"【内饰评论】\n{row['内饰评论']}")
    if pd.notna(row["性价比评论"]) and str(row["性价比评论"]).strip():
        parts.append(f"【性价比评论】\n{row['性价比评论']}")
    if pd.notna(row["性价比评论"]) and str(row["性价比评论"]).strip():
        parts.append(f"【智能化评论】\n{row['智能化评论']}")
    if pd.notna(row["智能化评论"]) and str(row["智能化评论"]).strip():
        parts.append(f"【油耗评论】\n{row['油耗评论']}")
    if pd.notna(row["配置评论"]) and str(row["配置评论"]).strip():
        parts.append(f"【配置评论】\n{row['配置评论']}")
    if not parts:
        return None
    return "\n\n".join(parts)

df["合并评价"] = df.apply(merge_satisfaction, axis=1)

# 只保留“合并评价”非空的行
df = df[df["合并评价"].notna()].copy()
print(f"有效评论数（至少包含满意或不满意之一）: {len(df)}")

models = df["车型名称"].unique()
print(f"车型列表: {models}")

# ========== 5. API 调用函数（与原程序完全相同）==========
def analyze_comment_with_retry(comment_text, retries=RETRY_TIMES):
    if not comment_text or not isinstance(comment_text, str):
        return []

    user_prompt = f"请分析以下汽车评论，提取类别和情感。\n评论：{comment_text}"

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            content = response.choices[0].message.content

            # 清理 Markdown 代码块
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            if not content.startswith('['):
                match = re.search(r'(\[.*\])', content, re.DOTALL)
                if match:
                    content = match.group(1)
                else:
                    print(f"无法从返回内容中提取 JSON 数组: {content[:200]}...")
                    continue

            result = json.loads(content)
            if isinstance(result, list):
                validated = []
                for item in result:
                    if (item.get("category") in CATEGORIES and
                        item.get("sentiment") in ["满意", "中立", "不满意"] and
                        item.get("target") in ["this", "other"] and
                        item.get("experience_type") in ["pre-purchase", "long-term"] and
                        item.get("score") in ALLOWED_SCORES):
                        item.setdefault("experience_reason", "")
                        validated.append(item)
                return validated
            else:
                print(f"返回格式不是列表: {content[:100]}")
                return []
        except json.JSONDecodeError as e:
            print(f"  JSON解析失败 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  API调用出错 (尝试 {attempt + 1}/{retries}): {e} | 评论: {comment_text[:30]}...")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"评论多次失败，放弃: {comment_text[:30]}...")
                return []
    return []

# ========== 6. 多线程批量处理（使用合并评价列）==========
results = []
failed_indices = []

print("开始批量分析（多线程）...")
# 注意：这里使用“合并评价”列
comment_items = [(i, row["车型名称"], row["合并评价"]) for i, row in df.iterrows()]

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_item = {
        executor.submit(analyze_comment_with_retry, comment): (idx, model, comment)
        for idx, model, comment in comment_items
    }

    for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item), desc="处理评论"):
        idx, model_name, comment = future_to_item[future]
        try:
            categories = future.result()
        except Exception as e:
            print(f"处理评论时发生未知异常: {e}")
            categories = []

        if categories:
            for cat in categories:
                results.append({
                    "评论ID": idx,
                    "车型名称": model_name,
                    "类别": cat["category"],
                    "情感": cat["sentiment"],
                    "评分": cat["score"],
                    "相关段落": cat.get("evidence", ""),
                    "是否本车": cat.get("target", "this"),
                    "体验类型": cat.get("experience_type", "long-term"),
                    "体验类型理由": cat.get("experience_reason", ""),
                    "评论内容": comment   # 保存的是合并后的完整文本
                })
        else:
            failed_indices.append(idx)

        time.sleep(REQUEST_DELAY)

print(f"处理完成。成功分类 {len(results)} 条记录，失败 {len(failed_indices)} 条评论。")

# 保存原始分类结果（包含全部记录，包括 other）
results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df.to_excel(DETAIL_EXCEL, index=False)
    print(f"全量明细已保存至 {DETAIL_EXCEL}")

# ========== 7. 过滤：只保留针对本车的记录 ==========
if "是否本车" in results_df.columns:
    results_df_this = results_df[results_df["是否本车"] == "this"].copy()
    print(f"针对本车的记录数: {len(results_df_this)}")
else:
    results_df_this = results_df.copy()

if results_df_this.empty:
    print("警告：没有针对本车的有效记录，无法进行统计。")
    # 后续代码仍会执行，但将得到空结果
else:
    # ========== 8. 统计与汇总（完全复用原代码）==========
    # 8.1 整体统计（情感三分类）
    category_stats = results_df_this.groupby(["类别", "情感"]).size().unstack(fill_value=0)
    category_stats["总提及"] = category_stats.sum(axis=1)
    category_stats["满意占比"] = (category_stats.get("满意", 0) / category_stats["总提及"] * 100).round(1)
    category_stats["中立占比"] = (category_stats.get("中立", 0) / category_stats["总提及"] * 100).round(1)
    category_stats["不满意占比"] = (category_stats.get("不满意", 0) / category_stats["总提及"] * 100).round(1)
    category_stats = category_stats.sort_values("总提及", ascending=False)

    very_negative_counts = results_df_this[results_df_this["评分"] == -2].groupby("类别").size()
    very_positive_counts = results_df_this[results_df_this["评分"] == 2].groupby("类别").size()

    category_stats["非常不满意次数"] = very_negative_counts.reindex(category_stats.index, fill_value=0)
    category_stats["非常满意次数"] = very_positive_counts.reindex(category_stats.index, fill_value=0)
    category_stats["非常不满意占比"] = (category_stats["非常不满意次数"] / category_stats["总提及"] * 100).round(1)
    category_stats["非常满意占比"] = (category_stats["非常满意次数"] / category_stats["总提及"] * 100).round(1)

    # 8.2 按车型统计
    model_category_stats = results_df_this.groupby(["车型名称", "类别", "情感"]).size().unstack(fill_value=0)
    model_category_stats["总提及"] = model_category_stats.sum(axis=1)
    model_total_mentions = model_category_stats.groupby(level=0)["总提及"].sum()
    for model in model_category_stats.index.get_level_values(0).unique():
        model_mask = model_category_stats.index.get_level_values(0) == model
        model_total = model_total_mentions[model]
        if model_total > 0:
            model_category_stats.loc[model_mask, "提及占比"] = (
                    model_category_stats.loc[model_mask, "总提及"] / model_total * 100
            ).round(1)
        else:
            model_category_stats.loc[model_mask, "提及占比"] = 0

    very_negative_by_model = results_df_this[results_df_this["评分"] == -2].groupby(["车型名称", "类别"]).size()
    very_positive_by_model = results_df_this[results_df_this["评分"] == 2].groupby(["车型名称", "类别"]).size()

    model_category_stats["非常不满意次数"] = very_negative_by_model.reindex(model_category_stats.index, fill_value=0)
    model_category_stats["非常满意次数"] = very_positive_by_model.reindex(model_category_stats.index, fill_value=0)
    model_category_stats["非常不满意占比"] = (
            model_category_stats["非常不满意次数"] / model_category_stats["总提及"] * 100).round(1)
    model_category_stats["非常满意占比"] = (
            model_category_stats["非常满意次数"] / model_category_stats["总提及"] * 100).round(1)

    # 8.3 车型情感分布
    model_sentiment_stats = results_df_this.groupby(["车型名称", "情感"]).size().unstack(fill_value=0)
    model_sentiment_stats["总提及"] = model_sentiment_stats.sum(axis=1)
    for sentiment in ["满意", "中立", "不满意"]:
        if sentiment in model_sentiment_stats.columns:
            model_sentiment_stats[f"{sentiment}_占比"] = (
                    model_sentiment_stats[sentiment] / model_sentiment_stats["总提及"] * 100
            ).round(1)

    # 8.4 评分相关统计
    avg_score_by_category = results_df_this.groupby("类别")["评分"].agg(["mean", "count"]).round(2)
    avg_score_by_category.columns = ["平均评分", "提及次数"]
    avg_score_by_category = avg_score_by_category.sort_values("平均评分", ascending=False)

    score_distribution = results_df_this["评分"].value_counts().sort_index()
    model_category_avg = results_df_this.groupby(["车型名称", "类别"])["评分"].mean().round(2).unstack(fill_value=0)

    # ========== 9. 体验类型深度分析（与原程序相同）==========
    print("\n===== 体验类型分析 =====")

    exp_type_counts = results_df_this["体验类型"].value_counts()
    exp_type_pct = (exp_type_counts / exp_type_counts.sum() * 100).round(1)
    exp_type_dist = pd.DataFrame({"提及次数": exp_type_counts, "占比(%)": exp_type_pct})
    print("\n整体体验类型分布:")
    print(exp_type_dist)

    exp_cat_counts = results_df_this.groupby(["体验类型", "类别"]).size().unstack(fill_value=0)
    exp_cat_counts["总计"] = exp_cat_counts.sum(axis=1)
    print("\n按体验类型的类别提及次数:")
    print(exp_cat_counts)

    exp_avg_score = results_df_this.groupby(["体验类型", "类别"])["评分"].mean().round(2).unstack(fill_value=0)
    exp_avg_score["总体平均"] = results_df_this.groupby("体验类型")["评分"].mean().round(2)
    print("\n按体验类型的类别平均评分:")
    print(exp_avg_score)

    exp_sentiment = results_df_this.groupby(["体验类型", "情感"]).size().unstack(fill_value=0)
    exp_sentiment["总提及"] = exp_sentiment.sum(axis=1)
    for sentiment in ["满意", "中立", "不满意"]:
        if sentiment in exp_sentiment.columns:
            exp_sentiment[f"{sentiment}_占比"] = (exp_sentiment[sentiment] / exp_sentiment["总提及"] * 100).round(1)
    print("\n按体验类型的情感分布:")
    print(exp_sentiment)

    # ========== 10. 体验类型可视化 ==========
    plt.figure(figsize=(8, 6))
    plt.pie(exp_type_counts, labels=exp_type_counts.index, autopct='%1.1f%%', startangle=90,
            colors=['#66b3ff', '#ff9999'])
    plt.title("体验类型整体分布")
    plt.savefig(os.path.join(OUTPUT_DIR, "体验类型整体分布.png"), dpi=300, bbox_inches="tight")
    plt.show()

    top_cats = category_stats.head(10).index.tolist()
    exp_cat_top = exp_cat_counts[top_cats].T
    exp_cat_top.plot(kind='bar', stacked=True, figsize=(14, 6), color=['#66b3ff', '#ff9999'])
    plt.title("Top10 类别在不同体验类型中的提及次数")
    plt.xlabel("类别")
    plt.ylabel("提及次数")
    plt.legend(title="体验类型")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "体验类型类别分布堆叠图.png"), dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="体验类型", y="评分", data=results_df_this, palette="Set2")
    plt.title("不同体验类型的评分分布对比")
    plt.xlabel("体验类型")
    plt.ylabel("评分")
    plt.savefig(os.path.join(OUTPUT_DIR, "体验类型评分箱线图.png"), dpi=300, bbox_inches="tight")
    plt.show()

    exp_avg_wide = results_df_this.groupby(["类别", "体验类型"])["评分"].mean().unstack()
    exp_avg_wide = exp_avg_wide.dropna().reset_index()
    if not exp_avg_wide.empty:
        plt.figure(figsize=(12, 8))
        plt.scatter(exp_avg_wide["pre-purchase"], exp_avg_wide["long-term"], s=100, alpha=0.7)
        min_val = min(exp_avg_wide[["pre-purchase", "long-term"]].min().min(), -1)
        max_val = max(exp_avg_wide[["pre-purchase", "long-term"]].max().max(), 2)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        for _, row in exp_avg_wide.iterrows():
            plt.annotate(row["类别"], (row["pre-purchase"], row["long-term"]),
                         textcoords="offset points", xytext=(5, 5), ha='center')
        plt.xlabel("pre-purchase 平均评分")
        plt.ylabel("long-term 平均评分")
        plt.title("各类别在不同体验类型中的平均评分对比")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "体验类型类别平均评分散点图.png"), dpi=300, bbox_inches="tight")
        plt.show()

    # ========== 11. 原有可视化（全局、车型、热力图等）==========
    def plot_top_categories(data, title_prefix, model_name=None, top_n=10):
        if data.empty:
            print(f"{title_prefix} 无数据，跳过绘图")
            return
        data = data.head(top_n).copy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        bars1 = ax1.barh(data["类别"], data["总提及"], color="steelblue")
        ax1.set_xlabel("提及次数（绝对频数）")
        ax1.set_title(f"{title_prefix} - 提及次数 TOP{top_n}")
        ax1.invert_yaxis()
        for bar, val in zip(bars1, data["总提及"]):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, str(val), va="center")

        bars2 = ax2.barh(data["类别"], data["提及占比"], color="coral")
        ax2.set_xlabel("提及占比（%）")
        ax2.set_title(f"{title_prefix} - 提及占比 TOP{top_n}")
        ax2.invert_yaxis()
        for bar, val in zip(bars2, data["提及占比"]):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val}%", va="center")

        plt.suptitle(f"{title_prefix} 前{top_n}关注类别" + (f"（车型: {model_name}）" if model_name else ""))
        plt.tight_layout()
        return fig

    def plot_sentiment_distribution(data, title_prefix, model_name=None, top_n=10):
        if data.empty:
            print(f"{title_prefix} 无数据，跳过绘图")
            return
        sentiment_cols = ["满意", "中立", "不满意"]
        for col in sentiment_cols:
            if col not in data.columns:
                data[col] = 0
        data = data.sort_values("总提及", ascending=False).head(top_n).copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        data[sentiment_cols].plot(kind="bar", stacked=True, ax=ax1, color=["#4CAF50", "#FFC107", "#F44336"])
        ax1.set_xlabel("类别")
        ax1.set_ylabel("提及次数")
        ax1.set_title(f"{title_prefix} - 情感分布（绝对频数）")
        ax1.legend(title="情感")
        ax1.tick_params(axis="x", rotation=45)

        data_pct = data[sentiment_cols].div(data["总提及"], axis=0) * 100
        data_pct.plot(kind="bar", stacked=True, ax=ax2, color=["#4CAF50", "#FFC107", "#F44336"])
        ax2.set_xlabel("类别")
        ax2.set_ylabel("占比（%）")
        ax2.set_title(f"{title_prefix} - 情感分布（百分比）")
        ax2.legend(title="情感")
        ax2.tick_params(axis="x", rotation=45)

        plt.suptitle(f"{title_prefix} 情感分布" + (f"（车型: {model_name}）" if model_name else ""))
        plt.tight_layout()
        return fig

    if not category_stats.empty:
        global_category = category_stats[["总提及"]].copy()
        global_category["提及占比"] = (global_category["总提及"] / global_category["总提及"].sum() * 100).round(1)
        global_category = global_category.reset_index().rename(columns={"index": "类别"})
        plot_top_categories(global_category, "整体", top_n=10)
        plt.savefig(os.path.join(OUTPUT_DIR, "整体类别提及TOP10.png"), dpi=300, bbox_inches="tight")
        plt.show()

        plot_sentiment_distribution(category_stats, "整体", top_n=10)
        plt.savefig(os.path.join(OUTPUT_DIR, "整体情感分布.png"), dpi=300, bbox_inches="tight")
        plt.show()

    for model in models:
        model_data = results_df_this[results_df_this["车型名称"] == model]
        if model_data.empty:
            continue

        model_cat_stats = model_data.groupby(["类别", "情感"]).size().unstack(fill_value=0)
        model_cat_stats["总提及"] = model_cat_stats.sum(axis=1)
        model_total = model_cat_stats["总提及"].sum()
        if model_total > 0:
            model_cat_stats["提及占比"] = (model_cat_stats["总提及"] / model_total * 100).round(1)
        else:
            model_cat_stats["提及占比"] = 0
        model_cat_stats = model_cat_stats.sort_values("总提及", ascending=False)

        cat_for_plot = model_cat_stats[["总提及", "提及占比"]].reset_index().rename(columns={"index": "类别"})
        plot_top_categories(cat_for_plot, f"{model} 车型", model_name=model, top_n=10)
        safe_model = re.sub(r'[\\/*?:"<>|]', "_", model)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_model}_类别提及TOP10.png"), dpi=300, bbox_inches="tight")
        plt.show()

        plot_sentiment_distribution(model_cat_stats, f"{model} 车型情感", model_name=model, top_n=10)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_model}_情感分布.png"), dpi=300, bbox_inches="tight")
        plt.show()

    pivot_counts = results_df_this.groupby(["车型名称", "类别"]).size().unstack(fill_value=0)
    pivot_pct = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_counts, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5)
    plt.title("各车型满意度类别提及次数热力图（绝对频数）")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "车型类别热力图_绝对频数.png"), dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, cbar_kws={'label': '提及占比 (%)'})
    plt.title("各车型满意度类别提及占比热力图（%）")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "车型类别热力图_占比.png"), dpi=300, bbox_inches="tight")
    plt.show()

    if not avg_score_by_category.empty:
        plt.figure(figsize=(12, 6))
        top_avg = avg_score_by_category.head(15)
        bars = plt.barh(top_avg.index, top_avg["平均评分"], color="skyblue")
        plt.xlabel("平均评分")
        plt.title("各类别平均评分（-2 到 2，越高越满意）")
        for bar, val in zip(bars, top_avg["平均评分"]):
            plt.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "全局平均评分.png"), dpi=300, bbox_inches="tight")
        plt.show()

    if not model_category_avg.empty:
        plt.figure(figsize=(16, 8))
        sns.heatmap(model_category_avg, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5)
        plt.title("各车型各类别平均评分热力图（-2 到 2，绿色为满意）")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "车型类别平均评分热力图.png"), dpi=300, bbox_inches="tight")
        plt.show()

    if not score_distribution.empty:
        plt.figure(figsize=(10, 6))
        score_distribution.sort_index().plot(kind="bar", color="lightcoral")
        plt.xlabel("评分")
        plt.ylabel("频数")
        plt.title("所有本车评论的评分分布")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "评分分布直方图.png"), dpi=300, bbox_inches="tight")
        plt.show()

    # ========== 12. 保存最终报表 ==========
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        category_stats.to_excel(writer, sheet_name="整体类别统计")
        model_category_stats.reset_index().to_excel(writer, sheet_name="车型类别统计", index=False)
        model_sentiment_stats.reset_index().to_excel(writer, sheet_name="车型情感分布", index=False)
        if not results_df_this.empty:
            results_df_this.to_excel(writer, sheet_name="分类明细(本车)", index=False)
        if not results_df.empty:
            results_df.to_excel(writer, sheet_name="全量明细", index=False)
        avg_score_by_category.to_excel(writer, sheet_name="类别平均评分")
        score_distribution.to_excel(writer, sheet_name="评分分布")
        model_category_avg.to_excel(writer, sheet_name="车型类别平均评分")

        exp_type_dist.to_excel(writer, sheet_name="体验类型整体分布")
        exp_cat_counts.reset_index().to_excel(writer, sheet_name="体验类型类别提及", index=False)
        exp_avg_score.reset_index().to_excel(writer, sheet_name="体验类型类别平均评分", index=False)
        exp_sentiment.reset_index().to_excel(writer, sheet_name="体验类型情感分布", index=False)

    print(f"\n分析完成！结果已保存至文件夹: {OUTPUT_DIR}")
    print(f"汇总报表: {OUTPUT_EXCEL}")
    print(f"全量明细文件: {DETAIL_EXCEL}")
    print("所有图表已保存为PNG文件。")
    print("\n体验类型分析结论摘要：")
    print(f"- 体验类型整体分布：{exp_type_dist.to_string()}")
    print(f"- 不同体验类型的情感满意率对比：")
    for exp_type in exp_sentiment.index:
        sat_rate = exp_sentiment.loc[exp_type, "满意_占比"] if "满意_占比" in exp_sentiment.columns else 0
        print(f"  {exp_type}: 满意占比 {sat_rate}%")