import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 8)
ax.set_ylim(0, 10)
ax.axis('off')

boxes = [
    (4, 9.2, '1. 加载 LaDe 数据\n与参数标定'),
    (4, 8.0, '2. 初始化仿真环境\n(horizon=720min)'),
    (4, 6.8, '3. 单种子基准对比\n(4 种策略)'),
    (4, 5.6, '4. 多重复随机实验\n(3 seeds / 策略)'),
    (4, 4.4, '5. 敏感性分析\n(N_CROWD_MAX)'),
    (4, 3.2, '6. 理论下界计算\n(Fluid bound)'),
    (4, 2.0, '7. 经验 KPI 校验\n(vs LaDe 真实值)'),
    (4, 0.8, '8. 输出结果\n(Excel / 表格 / 图表)'),
]

for x, y, text in boxes:
    box = FancyBboxPatch((x-1.8, y-0.4), 3.6, 0.8,
                         boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor='#E7F3FF', edgecolor='#4472C4', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=11, wrap=True)

for i in range(len(boxes)-1):
    x1, y1, _ = boxes[i]
    x2, y2, _ = boxes[i+1]
    arrow = FancyArrowPatch((x1, y1-0.45), (x2, y2+0.45),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#4472C4')
    ax.add_patch(arrow)

ax.set_title('实验流程图', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('output/figures/G_experiment_flow.png', dpi=300, bbox_inches='tight')
plt.close()
print('Flowchart saved.')
