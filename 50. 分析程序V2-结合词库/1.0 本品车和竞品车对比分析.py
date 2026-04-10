# -*- coding: utf-8 -*-
"""
汽车竞品对比分析系统（适配汽车之家口碑数据）
功能：
- 自动合并多个评论字段为一个完整评论文本
- 使用 DeepSeek 推理模型自动识别评论中提及的竞品车型
- 精确判断本品车相对于竞品车是 更好(better) / 持平(same) / 更差(worse)
- 生成竞品提及频次统计、优劣势分布图表
- 输出完整的 Excel 明细报告（含车型版本）
"""

import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 全局配置 ==========
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("❌ 错误: 未设置环境变量 DEEPSEEK_API_KEY")
    exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 使用推理模型以获得更好的逻辑判断能力
MODEL = "deepseek-reasoner"

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

# 输出目录
OUTPUT_DIR = "竞品对比分析结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输入文件
INPUT_EXCEL = "../10.汽车口碑数据/merged_autohome_reviews_V2.xlsx"

# 输出文件
COMPARISON_DETAIL = os.path.join(OUTPUT_DIR, "竞品对比明细.xlsx")
COMPARISON_REPORT = os.path.join(OUTPUT_DIR, "竞品对比分析报告.xlsx")
COMPARISON_STATS_TXT = os.path.join(OUTPUT_DIR, "竞品对比统计摘要.txt")

# ========== 推理模型专用提示词 ==========
system_prompt = """
你是一个专业的汽车评论分析助手。你的唯一任务是：从用户评论中提取出**本车（当前讨论的车型）**与**其他对比车型**的对比信息。

### 任务要求：
1. **识别对比车型**：找出评论中提到的所有**非本车的其他车型**，并提取其标准名称（例如从“雅阁”、“Accord”统一为“雅阁”）。
2. **判断对比类别**：确定对比涉及的方面，必须从预定义列表中选择。
3. **判断对比结果**：明确指出**本车相对于该对比车型**在此类别上是更好、持平还是更差。
4. **提供证据**：摘录评论中支持该对比关系的原文句子。

### 类别列表：
- 品牌、安全、操控、动力、质量、外观、舒适、使用成本、购车价格、配置、空间、内饰、服务、权益、智能座舱、智能驾驶

### 返回格式：
你必须返回一个 JSON 对象，包含一个 `"comparisons"` 数组。如果没有提及任何对比车型，则返回空数组。

数组中的每个元素是一个对象，包含：
- "compare_model": 对比车型的标准名称（字符串）
- "category": 类别名称（必须从上述列表中选择）
- "comparison_result": 对比结果，只能是以下三者之一：
   - "better": 本车更好（优势）
   - "same": 两者差不多（持平）
   - "worse": 本车更差（劣势）
- "evidence": 支持该对比的原文句子

**示例：**
用户评论：“思域的动力比轩逸强太多了，但内饰不如轩逸精致。”
输出：
{
  "comparisons": [
    {
      "compare_model": "轩逸",
      "category": "动力",
      "comparison_result": "better",
      "evidence": "思域的动力比轩逸强太多了"
    },
    {
      "compare_model": "轩逸",
      "category": "内饰",
      "comparison_result": "worse",
      "evidence": "内饰不如轩逸精致"
    }
  ]
}

**重要提醒：**
- 只输出 JSON，不要有任何其他文字。
- 如果评论中没有对比其他车型，返回 {"comparisons": []}。
- 仔细区分“本车更好”还是“竞品更好”，不要弄反。
- 如果评论中只提到竞品但没有明确对比结论，不要强行输出。
"""

# ========== 增强的 JSON 提取函数 ==========
def safe_parse_json(content):
    """尝试解析 JSON，若截断则使用栈式修复"""
    content = content.strip()
    # 清理 markdown
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试用正则提取第一个完整的对象或数组
    # 优先匹配完整的对象
    obj_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(1))
        except json.JSONDecodeError:
            pass

    # 若仍然失败，进行栈式修复
    try:
        repaired = repair_json(content)
        return json.loads(repaired)
    except Exception:
        # 最后尝试：将内容包装成对象
        if not content.startswith('{'):
            content = '{' + content
        if not content.endswith('}'):
            content = content + '}'
        try:
            return json.loads(content)
        except:
            pass

    # 如果所有修复都失败，抛出异常
    raise ValueError("无法修复并解析 JSON 内容")


def repair_json(broken_json):
    """
    使用栈式算法修复截断的 JSON 字符串。
    处理未闭合的字符串、括号、逗号等。
    """
    # 移除可能存在的 BOM 头
    if broken_json.startswith('\ufeff'):
        broken_json = broken_json[1:]

    stack = []
    fixed = []
    in_string = False
    escape = False
    i = 0
    n = len(broken_json)

    while i < n:
        ch = broken_json[i]
        if in_string:
            fixed.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            fixed.append(ch)
            i += 1
            continue

        # 不在字符串内，处理括号
        if ch in '{[':
            stack.append(ch)
            fixed.append(ch)
        elif ch in '}]':
            if stack:
                last = stack[-1]
                if (last == '{' and ch == '}') or (last == '[' and ch == ']'):
                    stack.pop()
                    fixed.append(ch)
                else:
                    # 括号不匹配，尝试修正：忽略此字符或补全
                    # 这里我们选择忽略不匹配的右括号
                    pass
            else:
                # 无左括号却有右括号，忽略
                pass
        elif ch == ',':
            fixed.append(ch)
        else:
            fixed.append(ch)
        i += 1

    # 处理结束后，若仍在字符串内，则闭合字符串
    if in_string:
        # 检查最后一个字符是否为引号，若不是则补上
        if fixed and fixed[-1] != '"':
            fixed.append('"')
        else:
            fixed.append('"')

    # 补全剩余的括号
    while stack:
        last = stack.pop()
        if last == '{':
            fixed.append('}')
        elif last == '[':
            fixed.append(']')

    return ''.join(fixed)


def analyze_comparison_with_reasoner(comment_text, retries=RETRY_TIMES):
    """调用推理模型，提取对比信息"""
    if not comment_text or not isinstance(comment_text, str):
        return []

    user_prompt = f"请分析以下汽车评论，提取本车与竞品的对比信息。\n评论：{comment_text}"

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = safe_parse_json(content)

            comparisons = result.get("comparisons", [])
            validated_comparisons = []

            if isinstance(comparisons, list):
                for item in comparisons:
                    if (isinstance(item, dict) and
                            item.get("compare_model") and
                            item.get("category") in CATEGORIES and
                            item.get("comparison_result") in ["better", "same", "worse"]):
                        item.setdefault("evidence", "")
                        validated_comparisons.append(item)

            return validated_comparisons

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"API调用最终失败 (尝试{retries}次后): {e}")
                return []
    return []


def merge_comment_fields(row):
    """将多个评论文本字段合并为一个完整的评论文本"""
    comment_parts = []
    fields = [
        "最满意", "最不满意", "空间评论", "驾驶感受评论", "续航评论",
        "外观评论", "内饰评论", "性价比评论", "智能化评论",
        "油耗评论", "配置评论"
    ]
    for field in fields:
        if field in row and pd.notna(row[field]) and str(row[field]).strip() != "":
            comment_parts.append(str(row[field]).strip())
    return "。".join(comment_parts)


def run_comparison_analysis():
    """读取数据、合并评论字段，调用API提取竞品对比信息"""
    print("正在读取数据...")
    df = pd.read_excel(INPUT_EXCEL, sheet_name=0)
    print(f"原始总记录数: {len(df)}")

    required_cols = ["车型名称", "车型版本"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 错误: Excel文件中缺少列 '{col}'")
            exit(1)

    # 合并评论文本
    print("正在合并评论文本字段...")
    df["评论内容"] = df.apply(merge_comment_fields, axis=1)

    # 清洗数据
    df = df[df["评论内容"].astype(str).str.strip() != ""].copy()
    print(f"有效评论数: {len(df)}")

    self_model = df["车型名称"].iloc[0] if not df.empty else "本车"
    print(f"本车车型: {self_model}")

    comparison_results = []
    failed_indices = []

    print("开始批量分析竞品对比（使用推理模型 deepseek-reasoner）...")
    comment_items = [(idx, row["车型名称"], row["车型版本"], row["评论内容"]) for idx, row in df.iterrows()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(analyze_comparison_with_reasoner, comment): (idx, model, version, comment)
            for idx, model, version, comment in comment_items
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_item),
                           total=len(future_to_item), desc="分析评论"):
            idx, model_name, version_name, comment = future_to_item[future]
            try:
                comp_items = future.result()
            except Exception:
                comp_items = []

            if comp_items:
                for comp in comp_items:
                    comparison_results.append({
                        "评论ID": idx,
                        "本车车型": model_name,
                        "本车车型版本": version_name,
                        "对比车型": comp["compare_model"],
                        "对比类别": comp["category"],
                        "对比结果": comp["comparison_result"],
                        "证据原文": comp.get("evidence", ""),
                        "原评论": comment
                    })
            else:
                failed_indices.append(idx)

            time.sleep(REQUEST_DELAY)

    comp_df = pd.DataFrame(comparison_results)

    # 保存明细
    with pd.ExcelWriter(COMPARISON_DETAIL) as writer:
        comp_df.to_excel(writer, sheet_name="竞品对比明细", index=False)
        pd.DataFrame({"失败评论ID": failed_indices}).to_excel(writer, sheet_name="分析失败记录", index=False)

    print(f"处理完成。成功提取竞品对比 {len(comp_df)} 条记录，失败 {len(failed_indices)} 条评论。")
    print(f"明细数据已保存至 {COMPARISON_DETAIL}")

    return comp_df, self_model


def perform_comparison_stats(comp_df, self_model):
    """对竞品对比数据进行统计分析和可视化"""
    if comp_df.empty:
        print("没有竞品对比数据，无法进行统计分析。")
        return

    print("\n===== 开始竞品对比统计分析 =====")

    total_comparisons = len(comp_df)
    result_counts = comp_df["对比结果"].value_counts()
    result_pct = (result_counts / total_comparisons * 100).round(1)

    # 按竞品车型统计
    model_stats = comp_df.groupby("对比车型").agg(
        提及次数=("评论ID", "nunique"),
        优势次数=("对比结果", lambda x: (x == "better").sum()),
        持平次数=("对比结果", lambda x: (x == "same").sum()),
        劣势次数=("对比结果", lambda x: (x == "worse").sum())
    )
    model_stats["净优势分"] = model_stats["优势次数"] - model_stats["劣势次数"]
    model_stats["优势率(%)"] = (model_stats["优势次数"] / model_stats["提及次数"] * 100).round(1)
    model_stats = model_stats.sort_values("提及次数", ascending=False)

    # 按类别统计
    category_stats = comp_df.groupby(["对比类别", "对比结果"]).size().unstack(fill_value=0)
    # 确保三个列都存在
    for col in ["better", "same", "worse"]:
        if col not in category_stats.columns:
            category_stats[col] = 0
    category_stats["总提及"] = category_stats.sum(axis=1)
    category_stats["净优势分"] = category_stats["better"] - category_stats["worse"]
    category_stats["优势率(%)"] = (category_stats["better"] / category_stats["总提及"] * 100).round(1)
    category_stats = category_stats.sort_values("总提及", ascending=False)

    # 优劣势具体组合
    strengths = comp_df[comp_df["对比结果"] == "better"].groupby(["对比车型", "对比类别"]).size().reset_index(name="次数")
    strengths = strengths.sort_values("次数", ascending=False).head(15)

    weaknesses = comp_df[comp_df["对比结果"] == "worse"].groupby(["对比车型", "对比类别"]).size().reset_index(name="次数")
    weaknesses = weaknesses.sort_values("次数", ascending=False).head(15)

    # 保存Excel报告
    with pd.ExcelWriter(COMPARISON_REPORT) as writer:
        comp_df.to_excel(writer, sheet_name="原始明细", index=False)
        model_stats.to_excel(writer, sheet_name="按竞品统计")
        category_stats.to_excel(writer, sheet_name="按类别统计")
        strengths.to_excel(writer, sheet_name="TOP15优势点")
        weaknesses.to_excel(writer, sheet_name="TOP15劣势点")

    print(f"统计报告已保存至 {COMPARISON_REPORT}")

    # 可视化图1
    if not model_stats.empty:
        top_models = model_stats.head(10)
        fig, ax1 = plt.subplots(figsize=(14, 7))
        x_pos = range(len(top_models))
        bars = ax1.bar(x_pos, top_models["提及次数"], color='steelblue', alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(top_models.index, rotation=45, ha='right')
        ax1.set_xlabel("竞品车型")
        ax1.set_ylabel("提及次数", color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        for bar, val in zip(bars, top_models["提及次数"]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha='center', va='bottom')

        ax2 = ax1.twinx()
        ax2.plot(x_pos, top_models["净优势分"], color='red', marker='o', linewidth=2, markersize=8)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel("净优势分", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        for i, val in enumerate(top_models["净优势分"]):
            ax2.annotate(str(val), (i, val), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
        plt.title(f"{self_model} 主要竞品提及次数与净优势分")
        fig.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "1_主要竞品分析.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 图2
    if not category_stats.empty:
        plt.figure(figsize=(14, 10))
        top_cats = category_stats.head(15).index
        plot_data = category_stats.loc[top_cats, ["better", "same", "worse"]]
        plot_data.plot(kind='barh', stacked=True, color=['#2ca02c', '#ffbb78', '#d62728'], figsize=(14, 10))
        plt.title(f"{self_model} 各类别竞品对比结果 (绿:优势, 橙:持平, 红:劣势)")
        plt.xlabel("提及次数")
        plt.ylabel("类别")
        plt.legend(title="对比结果")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "2_各类别对比结果.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 图3：热力图
    if not model_stats.empty and not category_stats.empty:
        comp_df['对比数值'] = comp_df['对比结果'].map({'better': 1, 'same': 0, 'worse': -1})
        pivot = comp_df.pivot_table(index='对比车型', columns='对比类别', values='对比数值', aggfunc='mean', fill_value=0)
        top_comp_models = model_stats.head(8).index
        top_comp_cats = category_stats["净优势分"].abs().sort_values(ascending=False).head(10).index
        pivot_filtered = pivot.loc[pivot.index.intersection(top_comp_models), pivot.columns.intersection(top_comp_cats)]
        if not pivot_filtered.empty:
            plt.figure(figsize=(16, 10))
            sns.heatmap(pivot_filtered, annot=True, cmap='RdYlGn', center=0, fmt='.2f',
                        linewidths=0.5, cbar_kws={'label': '对比得分 (1=优势, 0=持平, -1=劣势)'})
            plt.title(f"{self_model} vs 主要竞品 优劣势热力图 (正数=本品优势, 负数=本品劣势)")
            plt.xlabel("对比类别")
            plt.ylabel("竞品车型")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "3_竞品对比热力图.png"), dpi=300, bbox_inches='tight')
            plt.close()

    # 图4：优劣势TOP10
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    if not strengths.empty:
        top_strengths = strengths.head(10)
        ax1 = axes[0]
        bars1 = ax1.barh(range(len(top_strengths)), top_strengths["次数"], color='#2ca02c')
        ax1.set_yticks(range(len(top_strengths)))
        ax1.set_yticklabels([f"{row['对比车型']} - {row['对比类别']}" for _, row in top_strengths.iterrows()])
        ax1.set_xlabel("提及次数")
        ax1.set_title("本品主要优势点 TOP10")
        ax1.invert_yaxis()
        for bar, val in zip(bars1, top_strengths["次数"]):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(val), va='center')

    if not weaknesses.empty:
        top_weaknesses = weaknesses.head(10)
        ax2 = axes[1]
        bars2 = ax2.barh(range(len(top_weaknesses)), top_weaknesses["次数"], color='#d62728')
        ax2.set_yticks(range(len(top_weaknesses)))
        ax2.set_yticklabels([f"{row['对比车型']} - {row['对比类别']}" for _, row in top_weaknesses.iterrows()])
        ax2.set_xlabel("提及次数")
        ax2.set_title("本品主要劣势点 TOP10")
        ax2.invert_yaxis()
        for bar, val in zip(bars2, top_weaknesses["次数"]):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(val), va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_优劣势TOP10.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 生成文字摘要
    with open(COMPARISON_STATS_TXT, 'w', encoding='utf-8') as f:
        f.write(f"【{self_model} 竞品对比分析摘要】\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"1. 总体概况\n")
        f.write(f"   - 共分析有效对比提及 {total_comparisons} 次\n")
        f.write(f"   - 本品优势占比: {result_pct.get('better', 0)}%\n")
        f.write(f"   - 持平占比: {result_pct.get('same', 0)}%\n")
        f.write(f"   - 本品劣势占比: {result_pct.get('worse', 0)}%\n\n")

        f.write(f"2. 主要竞品 (按提及次数)\n")
        for model, row in model_stats.head(5).iterrows():
            f.write(f"   - {model}: 提及 {int(row['提及次数'])} 次, 净优势分 {int(row['净优势分'])} (优势{int(row['优势次数'])} / 持平{int(row['持平次数'])} / 劣势{int(row['劣势次数'])})\n")

        f.write(f"\n3. 本品核心优势 (提及最多的优势点)\n")
        for _, row in strengths.head(5).iterrows():
            f.write(f"   - vs {row['对比车型']} 在【{row['对比类别']}】方面: {int(row['次数'])} 次提及\n")

        f.write(f"\n4. 本品核心劣势 (提及最多的劣势点)\n")
        for _, row in weaknesses.head(5).iterrows():
            f.write(f"   - vs {row['对比车型']} 在【{row['对比类别']}】方面: {int(row['次数'])} 次提及\n")

    print(f"文字摘要已保存至 {COMPARISON_STATS_TXT}")


def main():
    print("=" * 60)
    print("汽车竞品对比分析系统 (基于 DeepSeek 推理模型)")
    print("=" * 60)

    if os.path.exists(COMPARISON_DETAIL):
        print(f"发现已有明细文件 {COMPARISON_DETAIL}，直接加载")
        comp_df = pd.read_excel(COMPARISON_DETAIL, sheet_name="竞品对比明细")
        if not comp_df.empty:
            self_model = comp_df["本车车型"].iloc[0]
        else:
            df = pd.read_excel(INPUT_EXCEL, sheet_name=0)
            self_model = df["车型名称"].iloc[0] if not df.empty else "本车"
        print(f"本车车型: {self_model}")
        print(f"已加载竞品对比记录 {len(comp_df)} 条")
    else:
        print("未找到已有明细，开始调用 DeepSeek API 进行分析...")
        comp_df, self_model = run_comparison_analysis()
        if comp_df.empty:
            print("❌ 没有提取到任何竞品对比信息，程序退出。")
            return

    perform_comparison_stats(comp_df, self_model)

    print("\n✅ 所有分析完成！")
    print(f"结果保存在目录: {OUTPUT_DIR}")
    print(f"- 对比明细: {COMPARISON_DETAIL}")
    print(f"- 统计报告: {COMPARISON_REPORT}")
    print(f"- 文字摘要: {COMPARISON_STATS_TXT}")
    print("- 可视化图表: 4张PNG图片")


if __name__ == "__main__":
    main()