import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# 데이터 불러오기
file_path = 'Planning_depth.xlsx'
data = pd.ExcelFile(file_path)

df = data.parse('Sheet1')

# 데이터 전처리
df_melted = df.melt(var_name="Algorithm", value_name="Planning Depth")
df_melted['Step'] = df_melted.groupby('Algorithm').cumcount()
df_melted['nmcts'] = df_melted['Algorithm'].str.extract(r'nmcts(\d+)').astype(int)
df_melted['Model Name'] = df_melted['Algorithm'].str.extract(r'([A-Za-z]+)_')

nmcts_order = [2, 10, 50, 100, 400]

import matplotlib.cm as cm

unique_nmcts = sorted(df_melted['nmcts'].unique(), key=lambda x: nmcts_order.index(x))
color_map = dict(zip(unique_nmcts, cm.tab10(np.linspace(0, 1, len(unique_nmcts)))))

# Plot 설정
plt.figure(figsize=(8, 5))
plt.rcParams.update({'font.size': 16})

# Moving Average Window 설정
window_size = 10  # 조정 가능

line_styles = {'QRQAC': '--', 'EQRQAC': '-'}

for (model_name, algorithm), group in df_melted.groupby(['Model Name', 'Algorithm']):
    color = color_map[group['nmcts'].iloc[0]]  # nmcts에 따라 색상 결정
    line_style = line_styles.get(model_name, '-')  # 모델에 따라 선 스타일 결정
    ma_depth = moving_average(group['Planning Depth'], window_size)  # Moving Average 적용
    steps = np.arange(len(ma_depth))

    # 모델 이름 치환 (범례용)
    display_model_name = model_name.replace("QRQAC", "QR-QAC").replace("EQRQAC", "EQR-QAC")

    # linewidth=2 추가
    plt.plot(steps, ma_depth, label=f"{display_model_name}_nmcts{group['nmcts'].iloc[0]}",
             color=color, linestyle=line_style, linewidth=2.5)

# 그래프 꾸미기
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# x축, y축 설정
ax.set_xlabel('Step')
ax.xaxis.set_label_coords(0.98, -0.1)  # 오른쪽 하단에서 조금 위로 조정
plt.ylabel('Planning Depth')


# 범례 정렬 및 표시
handles, labels = ax.get_legend_handles_labels()

sorted_legend = sorted(zip(labels, handles), key=lambda x: (
    x[0].startswith('QR-QAC'),
    nmcts_order.index(int(x[0].split('nmcts')[-1]))))
sorted_labels, sorted_handles = zip(*sorted_legend)

plt.legend(sorted_handles, sorted_labels, title='Models', bbox_to_anchor=(0.96, 1),
           edgecolor='black')
plt.xticks([20, 40, 60, 80, 100])
plt.tight_layout()

# 그래프 출력
plt.show()
