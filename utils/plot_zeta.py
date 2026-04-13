import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams
import platform

system_name = platform.system()
if system_name == 'Windows':
    # Windows 下通常叫 SimSun
    rcParams['font.family'] = ['SimSun', 'Microsoft YaHei', 'DejaVu Sans']
elif system_name == 'Darwin': # Mac
    # Mac 下通常叫 Songti SC 或 STSong
    rcParams['font.family'] = ['Songti SC', 'STSong', 'SimHei', 'DejaVu Sans']
else:
    # Linux 或其他，尝试常用中文字体
    rcParams['font.family'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']

rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 设置五号字大小 (约 10.5 pt)
font_size = 14

# 1. 准备数据
zeta = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
# human
time_saved_percent = np.array([0.0, 0.0577, 0.1474, 0.2308, 0.2821, 0.3205, 0.3333, 0.3910, 0.4808, 0.5192, 0.5321]) * 100
perf_growth = -np.array([0.0, 0.0, -0.0064, -0.0064, -0.0128, -0.0064, 0.0, 0.0128, 0.0256, 0.0321, 0.0385]) * 100
# machine
# time_saved_percent = np.array([0.0, 0.1888, 0.3147, 0.3776, 0.4406, 0.4965, 0.5455, 0.6224, 0.6573, 0.6713, 0.7063]) * 100
# perf_growth = -np.array([0.0, 0.0, -0.0070, -0.0070, 0.0, 0.007, 0.0, -0.0070, -0.0140, -0.0140, -0.0070]) * 100


# 2. 创建图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 设置颜色
color_switch = '#1f77b4'  # 蓝色
color_perf = '#d62728'    # 红色

# --- 绘制左轴 (非思考模式比例) ---
ax1.set_xlabel(r'困惑度阈值$\zeta$', fontsize=font_size, fontweight='bold')
ax1.set_ylabel('非思考模式比例（%）', color=color_switch, fontsize=font_size, fontweight='bold')
line1 = ax1.plot(zeta, time_saved_percent, color=color_switch, marker='o', linewidth=2, label='非思考模式比例（%）')
ax1.tick_params(axis='y', labelcolor=color_switch)
ax1.grid(True, linestyle='--', alpha=0.5)

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# --- 绘制右轴 (性能增长) ---
ax2 = ax1.twinx()  # 共享X轴
ax2.set_ylabel('pass@1增长百分点（%）', color=color_perf, fontsize=font_size, fontweight='bold')
line2 = ax2.plot(zeta, perf_growth, color=color_perf, marker='s', linewidth=2, label='pass@1增长百分点（%）')
ax2.tick_params(axis='y', labelcolor=color_perf)
ax2.set_ylim(-5, 5)

ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=font_size)

# 3. 添加关键点的标注
# 找到最大值索引 (增长百分点)
max_perf_idx = np.argmax(perf_growth)

# 标注性能增长最高点 (保持高精度)
perf_val_str = f"{perf_growth[max_perf_idx]:.2f}"
ax2.annotate(f'最高增长\n({zeta[max_perf_idx]}, {perf_val_str})', 
             xy=(zeta[max_perf_idx], perf_growth[max_perf_idx]), 
             xytext=(-10, 50), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color=color_perf),
             color=color_perf, fontweight='bold', fontsize=font_size-1)

ax2.axhline(y=0, color=color_perf, linestyle='--', linewidth=1.5, alpha=0.7)


# 设置标题
# plt.title(r'Trade-off Analysis: Switching Ratio vs. Performance Drop across different $\zeta$', fontsize=14, pad=20)

# 调整布局
fig.tight_layout()

# 显示图表
plt.show()

# 如果需要保存
plt.savefig('pics/zeta_tradeoff_analysis.png', dpi=300)