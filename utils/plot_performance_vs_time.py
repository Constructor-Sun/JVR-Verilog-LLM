import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def plot_performance_vs_time(models, pass_at_1, inference_time, font_size=12, save_path="pics/performance_vs_time.png"):
    """
    绘制模型推理性能 (Pass@1) 和耗时的综合散点图。
    
    Args:
        models (list): 模型名称列表
        pass_at_1 (list): Pass@1 分数列表 (百分比)
        inference_time (list): 推理耗时列表 (秒)
        font_size (int): 字体大小
        save_path (str): 图表保存路径
    """
    
    # 设置中文字体 (优先寻找 Noto Serif CJK SC, 它是 Linux 下常用的宋体风格字体)
    # 如果环境中有 SimSun 或其他 Songti 也可以指定
    target_font = 'Noto Serif CJK SC'
    
    # 检查是否为无界面环境
    if 'DISPLAY' not in os.environ:
        import matplotlib
        matplotlib.use('Agg') # 使用非交互式后端
        
    # 尝试查找系统中可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if target_font in available_fonts:
        plt.rcParams['font.serif'] = [target_font]
        plt.rcParams['font.family'] = 'serif'
    else:
        # 备选：查找任何包含 'Song' 或 'Serif CJK' 的字体
        song_fonts = [f for f in available_fonts if 'Song' in f or 'Serif CJK' in f]
        if song_fonts:
            plt.rcParams['font.serif'] = [song_fonts[0]]
            plt.rcParams['font.family'] = 'serif'
        else:
            print(f"警告：未找到字体 '{target_font}'，将使用默认字体。")
            
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = font_size

    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    # 为不同的模型生成不同的颜色
    cmap = plt.get_cmap('tab10')
    for i, (model, p1, t) in enumerate(zip(models, pass_at_1, inference_time)):
        plt.scatter(t, p1, color=cmap(i % 10), label=model, s=150, edgecolors='white', linewidth=1.5, alpha=0.9, zorder=3)
        # 在点旁边添加模型名称标注
        plt.annotate(model, (t, p1), textcoords="offset points", xytext=(5,5), ha='left', fontsize=font_size-2, zorder=4)

    # 设置标题和标签
    plt.title("推理性能和耗时综合散点图", fontsize=font_size + 4, fontweight='bold', pad=20)
    plt.xlabel("推理耗时 (秒)", fontsize=font_size + 2, labelpad=10)
    plt.ylabel("Pass@1 (%)", fontsize=font_size + 2, labelpad=10)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # 添加图例
    plt.legend(loc='best', fontsize=font_size, frameon=True, shadow=True)

    # 自动调整布局
    plt.tight_layout()
    
    # 保存并显示
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存至: {os.path.abspath(save_path)}")
    
    if 'DISPLAY' in os.environ:
        plt.show()

if __name__ == "__main__":
    # 示例数据：6 个不同的模型
    models = [
        "RTLCoder-Mistral-6.7B", 
        "RTLCoder-Deepseek-6.7B", 
        "Origen-7B", 
        "VeriRL-7B", 
        "Qming-CodeV-R1-7B", 
        "Ours"
    ]
    
    pass_at_1 = [62.5, 41.6, 54.4, 69.3, 69.9, 70.3]
    inference_time = [9.12, 2.31, 3.62, 14.01, 21.35, 17.21]

    # pass_at_1 = [36.7, 61.2, 74.1, 76.5, 76.6]
    # inference_time = []
    
    # 调用绘图函数，设置字号为 12
    plot_performance_vs_time(models, pass_at_1, inference_time, font_size=12)
