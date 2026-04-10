import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import os
from pathlib import Path
import warnings

# 忽略警告
warnings.filterwarnings('ignore')


# ========== 1. 字体设置优化 ==========
def setup_chinese_font():
    """设置中文字体，解决中文显示问题"""
    try:
        # 方法1: 直接使用系统字体（Windows）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
        plt.rcParams['axes.unicode_minus'] = False

        # 方法2: 尝试更具体的字体设置
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

        print("字体设置完成：使用SimHei字体")
        return True
    except Exception as e:
        print(f"字体设置失败: {e}")
        return False


# 初始化字体设置
if not setup_chinese_font():
    # 备选方案：设置字体属性
    font_properties = {'family': 'sans-serif',
                       'weight': 'normal',
                       'size': 12}
    plt.rcParams.update(font_properties)

# 设置专业期刊图表样式
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# ========== 2. 读取数据 ==========
print("正在读取数据...")
try:
    df_reviews = pd.read_excel('merged_civic_reviews.xlsx')
    df_models = pd.read_csv('制造商_车型_唯一组合.csv')
    print(f"读取成功: 评论数据 {df_reviews.shape[0]} 行，车型数据 {df_models.shape[0]} 行")
except Exception as e:
    print(f"读取数据失败: {e}")
    exit()

# ========== 3. 数据预处理 ==========
# 准备要分析的评论列
comment_columns = ['最满意', '最不满意', '空间评论', '驾驶感受评论',
                   '续航评论', '外观评论', '内饰评论', '性价比评论',
                   '智能化评论', '油耗评论', '配置评论']

# 修正列名映射
column_mapping = {
    '最满意': '最满意',
    '最不满意': '最不满意',
    '空间评论': '空间评论',
    '驾驶感受评论': '驾驶感受评论',
    '续航评论': '续航评论',
    '外观评论': '外观评论',
    '内饰评论': '内饰评论',
    '性价比评论': '性价比评论',
    '智能化评论': '智能化评论',
    '油耗评论': '油耗评论',
    '配置评论': '配置评论'
}


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

# 提取制造商和车型列表
manufacturers = df_models['制造商'].unique().tolist()
models = df_models['车型'].unique().tolist()


# ========== 4. 文本分析函数 ==========
def count_mentions(text, keywords):
    """统计文本中关键词的出现次数（不区分大小写）"""
    if pd.isna(text):
        return 0

    text_lower = str(text).lower()
    count = 0
    for keyword in keywords:
        # 使用正则表达式确保匹配完整单词
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        count += len(matches)
    return count


# ========== 5. 分析每个车型评论中的提及率 ==========
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

        print(f"  正在分析: {model_name} ({total_comments}条评论)")

        model_mentions = {}
        # 对每条评论进行判断
        for _, row in group.iterrows():
            comment = row['combined_comments']
            if pd.isna(comment):
                continue
            comment_str = str(comment).lower()

            # 记录本条评论已提及的车型，避免重复计数（同一评论多次提及只算一次）
            mentioned_in_this_comment = set()
            for target_model in models:
                if target_model == model_name:
                    continue  # 跳过自身
                if is_model_mentioned(comment_str, target_model):
                    mentioned_in_this_comment.add(target_model)

            # 累计到总提及统计中（按评论条数计）
            for target in mentioned_in_this_comment:
                model_mentions[target] = model_mentions.get(target, 0) + 1

        # 转换为提及率（提及评论数 / 总评论数）
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


# 执行分析
print("\n开始分析车型提及率...")
mentions_df = analyze_model_mentions(df_reviews, models)
print(f"分析完成，共分析 {len(mentions_df)} 个车型")


# ========== 6. 为每个车型创建独立的提及率关系图（优化中文显示） ==========
def create_individual_model_mentions_charts(mentions_df, top_n=10):
    """为每个车型创建独立的车型提及率关系图（优化中文显示）"""

    # 创建保存可视化图的文件夹
    output_dir = Path("车型提及率独立关系图")
    output_dir.mkdir(exist_ok=True)

    # 设置颜色映射
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

        # 如果没有提及其他车型，跳过
        if not model_mentions:
            print(f"  {model_name}: 无其他车型提及，跳过")
            charts_skipped += 1
            continue

        # 按提及率排序并选择前N个
        sorted_mentions = sorted(
            model_mentions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        if not sorted_mentions:
            print(f"  {model_name}: 无有效提及数据，跳过")
            charts_skipped += 1
            continue

        # 准备数据
        mentioned_models = [item[0] for item in sorted_mentions]
        mention_rates = [item[1] for item in sorted_mentions]

        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 8))

        # 生成条形图
        y_pos = np.arange(len(mentioned_models))

        # 为每个条形分配颜色
        bar_colors = []
        for i in range(len(mentioned_models)):
            # 使用渐变色
            color_idx = i / max(len(mentioned_models), 1)
            bar_colors.append(custom_cmap(color_idx))

        bars = ax.barh(y_pos, mention_rates, height=0.7, color=bar_colors,
                       edgecolor='black', linewidth=0.5)

        # ========== 优化中文显示部分 ==========
        # 设置图表属性（显式设置字体属性）
        ax.set_yticks(y_pos)

        # 显式设置y轴标签的字体属性
        yticklabels = mentioned_models
        ax.set_yticklabels(yticklabels, fontsize=11, fontproperties='SimHei')

        # 设置x轴标签
        ax.set_xlabel('提及率 (提及次数/评论数)', fontsize=12, fontweight='bold', fontproperties='SimHei')

        # 设置标题（显式设置字体属性）
        title_text = f'{model_name}评论中其他车型提及率\n(评论数: {comment_count})'
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20, fontproperties='SimHei')

        ax.invert_yaxis()  # 最高的在最上面

        # 添加网格线
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # 在条形上添加数值标签
        for i, (bar, rate) in enumerate(zip(bars, mention_rates)):
            width = bar.get_width()
            # 数值标签不需要中文字体
            ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{rate:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

        # 添加统计信息标注
        total_mentions = sum([rate * comment_count for rate in mention_rates])
        avg_mention_rate = sum(mention_rates) / len(mention_rates) if mention_rates else 0

        info_text = f"共提及{len(model_mentions)}个其他车型\n"
        info_text += f"总提及次数: {total_mentions:.0f}\n"
        info_text += f"平均提及率: {avg_mention_rate:.4f}"

        # 将标注放在图表的右上角（显式设置字体属性）
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontproperties='SimHei')

        # 调整布局
        plt.tight_layout()

        # 保存图片（使用安全的文件名）
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', model_name)
        output_path = output_dir / f"{safe_filename}_车型提及率关系图.png"

        # 保存时确保字体嵌入
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                    format='png')
        plt.close()

        print(f"  已保存: {model_name} -> {output_path}")
        charts_created += 1

    print(f"\n图表生成完成:")
    print(f"  成功创建: {charts_created} 张图表")
    print(f"  跳过: {charts_skipped} 个车型（无数据）")
    print(f"  所有图表已保存到: {output_dir}")


# ========== 7. 创建车型提及率汇总分析（优化中文显示） ==========
def create_summary_analysis(mentions_df):
    """创建车型提及率汇总分析（优化中文显示）"""

    if len(mentions_df) == 0:
        print("没有数据可用于汇总分析")
        return None

    print("\n创建汇总分析图表...")

    # 计算每个车型被其他车型提及的总次数
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

    # 计算每个车型被提及的平均率
    avg_mention_rates = {}
    for model, mentions in model_mentioned_by_others.items():
        total_rate = sum(item['rate'] for item in mentions)
        avg_rate = total_rate / len(mentions) if mentions else 0
        avg_mention_rates[model] = {
            'avg_rate': avg_rate,
            'mentioned_by_count': len(mentions),
            'total_rate': total_rate
        }

    # 按平均提及率排序
    sorted_avg_rates = sorted(
        avg_mention_rates.items(),
        key=lambda x: x[1]['avg_rate'],
        reverse=True
    )

    # 创建汇总图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # 图1: 车型被提及的平均率
    top_models = sorted_avg_rates[:min(15, len(sorted_avg_rates))]
    model_names = [item[0] for item in top_models]
    avg_rates = [item[1]['avg_rate'] for item in top_models]
    mention_counts = [item[1]['mentioned_by_count'] for item in top_models]

    y_pos = np.arange(len(model_names))
    bars1 = ax1.barh(y_pos, avg_rates, color=plt.cm.viridis(np.linspace(0.3, 0.8, len(model_names))))

    ax1.set_yticks(y_pos)
    # 显式设置y轴标签字体
    ax1.set_yticklabels(model_names, fontproperties='SimHei')
    ax1.set_xlabel('平均被提及率', fontsize=12, fontweight='bold', fontproperties='SimHei')
    ax1.set_title('车型被其他车型提及的平均率排名', fontsize=14, fontweight='bold',
                  pad=20, fontproperties='SimHei')
    ax1.invert_yaxis()

    # 添加数值标签
    for i, (bar, rate, count) in enumerate(zip(bars1, avg_rates, mention_counts)):
        width = bar.get_width()
        # 数值和英文不需要中文字体
        ax1.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{rate:.4f} (被{count}个车型提及)', ha='left', va='center', fontsize=9)

    # 图2: 车型提及网络关系热力图（简化版）
    # 选择前10个车型或更少
    top_n = min(10, len(sorted_avg_rates))
    selected_models = [item[0] for item in sorted_avg_rates[:top_n]]

    # 创建热力图数据
    heatmap_data = np.zeros((top_n, top_n))

    for i, source_model in enumerate(selected_models):
        source_data = mentions_df[mentions_df['车型名称'] == source_model]
        if not source_data.empty:
            model_mentions = source_data.iloc[0]['车型提及率']
            for j, target_model in enumerate(selected_models):
                if source_model != target_model and target_model in model_mentions:
                    heatmap_data[i, j] = model_mentions[target_model]

    im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    # 设置坐标轴标签（显式设置字体）
    ax2.set_xticks(np.arange(len(selected_models)))
    ax2.set_xticklabels(selected_models, rotation=45, ha='right',
                        fontsize=9, fontproperties='SimHei')
    ax2.set_yticks(np.arange(len(selected_models)))
    ax2.set_yticklabels(selected_models, fontsize=9, fontproperties='SimHei')
    ax2.set_title('车型间提及率关系热力图', fontsize=14, fontweight='bold',
                  pad=20, fontproperties='SimHei')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('提及率', fontsize=10, fontproperties='SimHei')

    # 添加数值标签
    for i in range(len(selected_models)):
        for j in range(len(selected_models)):
            if heatmap_data[i, j] > 0:
                ax2.text(j, i, f'{heatmap_data[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()

    # 保存图片
    output_path = '车型提及率汇总分析.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    print(f"汇总分析图表已保存: {output_path}")

    return sorted_avg_rates


# ========== 8. 保存详细数据 ==========
def save_detailed_results(mentions_df):
    """保存详细分析结果到Excel文件"""

    print("\n保存详细分析数据...")

    if len(mentions_df) == 0:
        print("没有数据可保存")
        return

    try:
        with pd.ExcelWriter('车型提及率详细分析.xlsx', engine='openpyxl') as writer:
            # 保存每个车型的详细提及数据
            detailed_data = []

            for idx, row in mentions_df.iterrows():
                model_name = row['车型名称']
                comment_count = row['评论数量']
                model_mentions = row['车型提及率']

                for mentioned_model, rate in model_mentions.items():
                    detailed_data.append({
                        '评论车型': model_name,
                        '被提及车型': mentioned_model,
                        '提及率': rate,
                        '评论数量': comment_count,
                        '理论提及次数': rate * comment_count
                    })

            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='详细提及数据', index=False)

            # 保存每个车型的汇总数据
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

            # 保存被提及车型统计
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
                    '提及车型': '; '.join(stats['提及车型列表'][:10])  # 只显示前10个
                })

            mentioned_df = pd.DataFrame(mentioned_data)
            mentioned_df.to_excel(writer, sheet_name='被提及车型统计', index=False)

            print("详细分析数据已保存: 车型提及率详细分析.xlsx")

    except Exception as e:
        print(f"保存Excel文件失败: {e}")


# ========== 9. 输出统计摘要 ==========
def print_statistics(mentions_df, sorted_avg_rates):
    """输出统计摘要"""

    print("\n" + "=" * 60)
    print("分析完成！统计摘要")
    print("=" * 60)

    print(f"分析车型总数: {len(mentions_df)}")
    print(f"有车型提及的车型数: {sum(1 for _, row in mentions_df.iterrows() if row['车型提及率'])}")

    # 计算提及率统计
    all_rates = []
    for _, row in mentions_df.iterrows():
        all_rates.extend(row['车型提及率'].values())

    if all_rates:
        print(f"总提及次数: {len(all_rates)}")
        print(f"平均提及率: {np.mean(all_rates):.4f}")
        print(f"最高提及率: {max(all_rates):.4f}")
        if any(r > 0 for r in all_rates):
            print(f"最低提及率(非零): {min([r for r in all_rates if r > 0]):.4f}")

    # 输出最常被提及的车型
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


# ========== 10. 主程序 ==========
def main():
    """主程序"""

    print("=" * 60)
    print("车型提及率分析系统")
    print("=" * 60)

    # 检查数据
    if '车型名称' not in df_reviews.columns:
        print("错误: 评论数据中缺少'车型名称'列")
        return

    # 为每个车型创建独立的提及率关系图
    create_individual_model_mentions_charts(mentions_df, top_n=15)

    # 创建汇总分析
    sorted_avg_rates = create_summary_analysis(mentions_df)

    # 保存详细结果
    save_detailed_results(mentions_df)

    # 输出统计摘要
    print_statistics(mentions_df, sorted_avg_rates)


# 执行主程序
if __name__ == "__main__":
    main()