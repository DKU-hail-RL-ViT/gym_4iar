import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def moving_average(data, window_size):
    """Calculate the moving average with a given window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def add_noise(data, noise_level=0.01):
    """Add small random noise to data."""
    return data + np.random.uniform(-noise_level, noise_level, size=len(data))

# Load data
file_path = 'number_of_quantiles.xlsx'
data = pd.ExcelFile(file_path)

df = data.parse('Sheet1')

# Preprocess data
df_melted = df.melt(var_name="Algorithm", value_name="Planning Breadth")
df_melted['Step'] = df_melted.groupby('Algorithm').cumcount()
df_melted['nmcts'] = df_melted['Algorithm'].str.extract(r'nmcts(\d+)').astype(int)
df_melted['Model Name'] = df_melted['Algorithm'].str.extract(r'([A-Za-z]+)_')

df_melted['Planning Breadth'] = pd.to_numeric(df_melted['Planning Breadth'], errors='coerce')
df_melted = df_melted.dropna(subset=['Planning Breadth'])

# Create color map
nmcts_order = [2, 10, 50, 100, 400]
unique_nmcts = sorted(df_melted['nmcts'].unique(), key=lambda x: nmcts_order.index(x))
color_map = dict(zip(unique_nmcts, cm.tab10(np.linspace(0, 1, len(unique_nmcts)))))

# Plot settings
plt.figure(figsize=(8, 5))
plt.rcParams.update({'font.size': 16})

# Moving Average Window
window_size = 10

# Define line styles by model
line_styles = {'QRQAC': '--', 'EQRQAC': '-'}

for (model_name, algorithm), group in df_melted.groupby(['Model Name', 'Algorithm']):
    color = color_map[group['nmcts'].iloc[0]]
    line_style = line_styles.get(model_name, '-')
    ma_breadth = moving_average(group['Planning Breadth'], window_size)
    steps = np.arange(len(ma_breadth))

    # Replace model names for legend display
    display_model_name = model_name.replace("QRQAC", "QR-QAC").replace("EQRQAC", "EQR-QAC")

    # Added linewidth=2 for thicker lines
    plt.plot(steps, ma_breadth, label=f"{display_model_name}_nmcts{group['nmcts'].iloc[0]}",
             color=color, linestyle=line_style, linewidth=2.5)

# Customize plot
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set axis labels
ax.set_xlabel('Step', fontsize=16)
ax.set_ylabel('number of quantile', fontsize=16)
x_ticks = [20, 40, 60, 80, 100]
ax.xaxis.set_label_coords(0.98, -0.02)

# Sort and display legend
handles, labels = ax.get_legend_handles_labels()
sorted_legend = sorted(zip(labels, handles), key=lambda x: (
    x[0].startswith('QR-QAC'),  # Ensure QR-QAC appears first
    nmcts_order.index(int(x[0].split('nmcts')[-1]))))
sorted_labels, sorted_handles = zip(*sorted_legend)

# plt.legend(sorted_handles, sorted_labels, title='Models', bbox_to_anchor=(0.96, 1), loc='upper left',
#            edgecolor='black')

plt.xticks([20, 40, 60, 80, 100])
plt.tight_layout()

# Display plot
plt.show()
