import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
import jieba
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings('ignore')

# ========== 1. 字体设置优化 ==========
def setup_chinese_font():
    """设置中文字体，解决中文显示问题"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        print("字体设置完成：使用SimHei字体")
        return True
    except Exception as e:
        print(f"字体设置失败: {e}")
        return False

setup_chinese_font()
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# ========== 2. 读取数据 ==========
print("正在读取数据...")
try:
    df_reviews = pd.read_excel('merged_autohome_reviews.xlsx')
    df_models = pd.read_csv('制造商_车型_唯一组合.csv')
    print(f"读取成功: 评论数据 {df_reviews.shape[0]} 行，车型数据 {df_models.shape[0]} 行")
except Exception as e:
    print(f"读取数据失败: {e}")
    exit()

# ========== 3. 数据预处理 ==========
comment_columns = ['最满意', '最不满意', '空间评论', '驾驶感受评论',
                   '续航评论', '外观评论', '内饰评论', '性价比评论',
                   '智能化评论', '油耗评论', '配置评论']


# 列名映射（如果实际列名不同，请在此修改）
column_mapping = {col: col for col in comment_columns}

def combine_comments(row):
    """合并所有评论列为一个文本"""
    combined_text = ""
    for col in comment_columns:
        if col in column_mapping and column_mapping[col] in df_reviews.columns:
            text = row[column_mapping[col]]
            if pd.notna(text):
                combined_text += f" {str(text)}"
    return combined_text.strip()

df_reviews['combined_comments'] = df_reviews.apply(combine_comments, axis=1)

# 提取所有车型名称（唯一）
models = df_models['车型'].unique().tolist()
print(f"共 {len(models)} 个车型")

# ========== 4. 初始化 jieba 分词 ==========
print("正在初始化分词词典...")
for model in models:
    jieba.add_word(model)  # 将所有车型名称加入词典，确保分词时作为一个整体
print("词典初始化完成")

# ========== 5. 定义匹配函数 ==========
def contains_chinese(text):
    """判断字符串是否包含中文字符"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def is_model_mentioned(comment, model_name):
    """
    判断一条评论是否提及某车型
    策略：
    - 如果车型名包含非中文字符（如英文、数字），使用正则匹配（忽略大小写，处理空格变体）
    - 如果车型名纯中文，使用 jieba 分词，检查车型名是否在词列表中
    """
    if pd.isna(comment) or not isinstance(comment, str):
        return False

    # 纯中文车型：使用分词
    if not contains_chinese(model_name):
        # 非中文车型（英文/数字）使用正则
        # 构建几个变体：原样、移除空格、全小写
        variants = [model_name]
        # 如果包含空格，添加无空格版本
        if ' ' in model_name:
            variants.append(model_name.replace(' ', ''))
        # 如果包含连字符，添加无连字符版本
        if '-' in model_name:
            variants.append(model_name.replace('-', ''))
        # 统一转为小写进行匹配
        comment_lower = comment.lower()
        for variant in variants:
            # 转义正则特殊字符
            escaped = re.escape(variant.lower())
            # 使用单词边界 \b，确保独立单词
            pattern = r'\b' + escaped + r'\b'
            if re.search(pattern, comment_lower):
                return True
        return False
    else:
        # 纯中文车型：jieba 分词
        words = jieba.lcut(comment)
        return model_name in words

# ========== 6. 分析每个车型评论中的提及率 ==========
def analyze_model_mentions(df_reviews, models):
    """分析每个车型评论中其他车型的提及率（基于评论条数）"""
    results = []
    unique_models = df_reviews['车型名称'].unique()
    print(f"发现 {len(unique_models)} 个车型进行分析")

    for model_name in unique_models:
        group = df_reviews[df_reviews['车型名称'] == model_name]
        total_comments = len(group)
        if total_comments == 0:
            continue

        print(f"\n正在分析: {model_name} ({total_comments}条评论)")

        model_mentions = {}
        # 使用 tqdm 显示进度条
        for _, row in tqdm(group.iterrows(), total=total_comments, desc=f"处理中", unit="条"):
            comment = row['combined_comments']
            if pd.isna(comment):
                continue

            mentioned_in_this = set()
            for target_model in models:
                if target_model == model_name:
                    continue
                if is_model_mentioned(comment, target_model):
                    mentioned_in_this.add(target_model)

            for target in mentioned_in_this:
                model_mentions[target] = model_mentions.get(target, 0) + 1

        # 转换为提及率
        model_mention_rates = {
            target: round(count / total_comments, 4)
            for target, count in model_mentions.items()
        }

        results.append({
            '车型名称': model_name,
            '评论数量': total_comments,
            '车型提及率': model_mention_rates
        })

    return pd.DataFrame(results)


print("\n开始分析车型提及率...")
mentions_df = analyze_model_mentions(df_reviews, models)
print(f"分析完成，共分析 {len(mentions_df)} 个车型")

# ========== 7. 为每个车型创建独立的提及率关系图 ==========
def create_individual_model_mentions_charts(mentions_df, top_n=10):
    """为每个车型创建独立的车型提及率关系图"""
    output_dir = Path("车型提及率独立关系图")
    output_dir.mkdir(exist_ok=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

    print(f"\n为 {len(mentions_df)} 个车型生成独立关系图...")
    charts_created = 0
    charts_skipped = 0

    for idx, row in mentions_df.iterrows():
        model_name = row['车型名称']
        comment_count = row['评论数量']
        model_mentions = row['车型提及率']

        if not model_mentions:
            print(f"  {model_name}: 无其他车型提及，跳过")
            charts_skipped += 1
            continue

        sorted_mentions = sorted(
            model_mentions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        if not sorted_mentions:
            print(f"  {model_name}: 无有效提及数据，跳过")
            charts_skipped += 1
            continue

        mentioned_models = [item[0] for item in sorted_mentions]
        mention_rates = [item[1] for item in sorted_mentions]

        fig, ax = plt.subplots(figsize=(14, 8))
        y_pos = np.arange(len(mentioned_models))

        bar_colors = [custom_cmap(i / max(len(mentioned_models), 1)) for i in range(len(mentioned_models))]
        bars = ax.barh(y_pos, mention_rates, height=0.7, color=bar_colors,
                       edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(mentioned_models, fontsize=11, fontproperties='SimHei')
        ax.set_xlabel('提及率 (提及评论数/总评论数)', fontsize=12, fontweight='bold', fontproperties='SimHei')
        title_text = f'{model_name}评论中其他车型提及率\n(评论数: {comment_count})'
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20, fontproperties='SimHei')
        ax.invert_yaxis()
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        for i, (bar, rate) in enumerate(zip(bars, mention_rates)):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{rate:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

        total_mentions = sum([rate * comment_count for rate in mention_rates])
        avg_mention_rate = sum(mention_rates) / len(mention_rates) if mention_rates else 0
        info_text = f"共提及{len(model_mentions)}个其他车型\n"
        info_text += f"总提及次数: {total_mentions:.0f}\n"
        info_text += f"平均提及率: {avg_mention_rate:.4f}"
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontproperties='SimHei')

        plt.tight_layout()
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', model_name)
        output_path = output_dir / f"{safe_filename}_车型提及率关系图.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', format='png')
        plt.close()
        print(f"  已保存: {model_name} -> {output_path}")
        charts_created += 1

    print(f"\n图表生成完成: 成功创建 {charts_created} 张图表，跳过 {charts_skipped} 个车型")

# ========== 8. 创建车型提及率汇总分析 ==========
def create_summary_analysis(mentions_df):
    if len(mentions_df) == 0:
        print("没有数据可用于汇总分析")
        return None

    print("\n创建汇总分析图表...")
    model_mentioned_by_others = {}

    for idx, row in mentions_df.iterrows():
        source_model = row['车型名称']
        model_mentions = row['车型提及率']
        for target_model, rate in model_mentions.items():
            if target_model not in model_mentioned_by_others:
                model_mentioned_by_others[target_model] = []
            model_mentioned_by_others[target_model].append({
                'source_model': source_model,
                'rate': rate,
                'comment_count': row['评论数量']
            })

    avg_mention_rates = {}
    for model, mentions in model_mentioned_by_others.items():
        total_rate = sum(item['rate'] for item in mentions)
        avg_rate = total_rate / len(mentions) if mentions else 0
        avg_mention_rates[model] = {
            'avg_rate': avg_rate,
            'mentioned_by_count': len(mentions),
            'total_rate': total_rate
        }

    sorted_avg_rates = sorted(
        avg_mention_rates.items(),
        key=lambda x: x[1]['avg_rate'],
        reverse=True
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # 图1：车型被提及的平均率排名
    top_models = sorted_avg_rates[:min(15, len(sorted_avg_rates))]
    model_names = [item[0] for item in top_models]
    avg_rates = [item[1]['avg_rate'] for item in top_models]
    mention_counts = [item[1]['mentioned_by_count'] for item in top_models]

    y_pos = np.arange(len(model_names))
    bars1 = ax1.barh(y_pos, avg_rates, color=plt.cm.viridis(np.linspace(0.3, 0.8, len(model_names))))

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_names, fontproperties='SimHei')
    ax1.set_xlabel('平均被提及率', fontsize=12, fontweight='bold', fontproperties='SimHei')
    ax1.set_title('车型被其他车型提及的平均率排名', fontsize=14, fontweight='bold', pad=20, fontproperties='SimHei')
    ax1.invert_yaxis()

    for i, (bar, rate, count) in enumerate(zip(bars1, avg_rates, mention_counts)):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{rate:.4f} (被{count}个车型提及)', ha='left', va='center', fontsize=9)

    # 图2：热力图
    top_n = min(10, len(sorted_avg_rates))
    selected_models = [item[0] for item in sorted_avg_rates[:top_n]]
    heatmap_data = np.zeros((top_n, top_n))

    for i, source_model in enumerate(selected_models):
        source_data = mentions_df[mentions_df['车型名称'] == source_model]
        if not source_data.empty:
            model_mentions = source_data.iloc[0]['车型提及率']
            for j, target_model in enumerate(selected_models):
                if source_model != target_model and target_model in model_mentions:
                    heatmap_data[i, j] = model_mentions[target_model]

    im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(np.arange(len(selected_models)))
    ax2.set_xticklabels(selected_models, rotation=45, ha='right', fontsize=9, fontproperties='SimHei')
    ax2.set_yticks(np.arange(len(selected_models)))
    ax2.set_yticklabels(selected_models, fontsize=9, fontproperties='SimHei')
    ax2.set_title('车型间提及率关系热力图', fontsize=14, fontweight='bold', pad=20, fontproperties='SimHei')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('提及率', fontsize=10, fontproperties='SimHei')

    for i in range(len(selected_models)):
        for j in range(len(selected_models)):
            if heatmap_data[i, j] > 0:
                ax2.text(j, i, f'{heatmap_data[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    output_path = '车型提及率汇总分析.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print(f"汇总分析图表已保存: {output_path}")
    return sorted_avg_rates

# ========== 9. 保存详细数据 ==========
def save_detailed_results(mentions_df):
    print("\n保存详细分析数据...")
    if len(mentions_df) == 0:
        print("没有数据可保存")
        return

    try:
        with pd.ExcelWriter('车型提及率详细分析.xlsx', engine='openpyxl') as writer:
            detailed_data = []
            for idx, row in mentions_df.iterrows():
                model_name = row['车型名称']
                comment_count = row['评论数量']
                model_mentions = row['车型提及率']
                for mentioned_model, rate in model_mentions.items():
                    detailed_data.append({
                        '评论车型': model_name,
                        '被提及车型': mentioned_model,
                        '提及评论占比': rate,
                        '评论数量': comment_count,
                        '提及评论数': int(rate * comment_count)
                    })
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='详细提及数据', index=False)

            summary_data = []
            for idx, row in mentions_df.iterrows():
                model_name = row['车型名称']
                comment_count = row['评论数量']
                model_mentions = row['车型提及率']
                if model_mentions:
                    avg_rate = np.mean(list(model_mentions.values()))
                    max_rate = max(model_mentions.values())
                    max_model = max(model_mentions.items(), key=lambda x: x[1])[0]
                    total_mentions = sum(model_mentions.values())
                else:
                    avg_rate = max_rate = total_mentions = 0
                    max_model = "无"
                summary_data.append({
                    '车型名称': model_name,
                    '评论数量': comment_count,
                    '提及其他车型数': len(model_mentions),
                    '平均提及率': avg_rate,
                    '最高提及率': max_rate,
                    '最常提及车型': max_model,
                    '总提及率': total_mentions
                })
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='车型汇总', index=False)

            mentioned_models = {}
            for idx, row in mentions_df.iterrows():
                source_model = row['车型名称']
                comment_count = row['评论数量']
                model_mentions = row['车型提及率']
                for target_model, rate in model_mentions.items():
                    if target_model not in mentioned_models:
                        mentioned_models[target_model] = {
                            '被提及次数': 0,
                            '总提及率': 0,
                            '提及车型列表': []
                        }
                    mentioned_models[target_model]['被提及次数'] += 1
                    mentioned_models[target_model]['总提及率'] += rate
                    mentioned_models[target_model]['提及车型列表'].append(f"{source_model}({rate:.4f})")

            mentioned_data = []
            for model, stats in mentioned_models.items():
                avg_rate = stats['总提及率'] / stats['被提及次数'] if stats['被提及次数'] > 0 else 0
                mentioned_data.append({
                    '被提及车型': model,
                    '被提及次数': stats['被提及次数'],
                    '平均被提及率': avg_rate,
                    '总被提及率': stats['总提及率'],
                    '提及车型': '; '.join(stats['提及车型列表'][:10])
                })
            mentioned_df = pd.DataFrame(mentioned_data)
            mentioned_df.to_excel(writer, sheet_name='被提及车型统计', index=False)

            print("详细分析数据已保存: 车型提及率详细分析.xlsx")
    except Exception as e:
        print(f"保存Excel文件失败: {e}")

# ========== 10. 输出统计摘要 ==========
def print_statistics(mentions_df, sorted_avg_rates):
    print("\n" + "=" * 60)
    print("分析完成！统计摘要")
    print("=" * 60)
    print(f"分析车型总数: {len(mentions_df)}")
    print(f"有车型提及的车型数: {sum(1 for _, row in mentions_df.iterrows() if row['车型提及率'])}")

    all_rates = []
    for _, row in mentions_df.iterrows():
        all_rates.extend(row['车型提及率'].values())

    if all_rates:
        print(f"总提及关系数: {len(all_rates)}")
        print(f"平均提及率: {np.mean(all_rates):.4f}")
        print(f"最高提及率: {max(all_rates):.4f}")
        non_zero = [r for r in all_rates if r > 0]
        if non_zero:
            print(f"最低提及率(非零): {min(non_zero):.4f}")

    if sorted_avg_rates:
        print(f"\n最常被提及的车型TOP 5:")
        for i, (model, stats) in enumerate(sorted_avg_rates[:5], 1):
            print(f"{i}. {model}: 平均提及率{stats['avg_rate']:.4f} (被{stats['mentioned_by_count']}个车型提及)")

    print("\n" + "=" * 60)
    print("输出文件:")
    print("=" * 60)
    print("1. '车型提及率独立关系图' 文件夹 - 每个车型的独立提及率关系图")
    print("2. '车型提及率详细分析.xlsx' - 详细分析数据")
    print("3. '车型提及率汇总分析.png' - 汇总分析图表")
    print("=" * 60)

# ========== 11. 主程序 ==========
def main():
    print("=" * 60)
    print("车型提及率分析系统")
    print("=" * 60)

    if '车型名称' not in df_reviews.columns:
        print("错误: 评论数据中缺少'车型名称'列")
        return

    create_individual_model_mentions_charts(mentions_df, top_n=15)
    sorted_avg_rates = create_summary_analysis(mentions_df)
    save_detailed_results(mentions_df)
    print_statistics(mentions_df, sorted_avg_rates)

if __name__ == "__main__":
    main()