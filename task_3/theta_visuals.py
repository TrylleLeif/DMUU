import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Define the theta values
theta_list = [([-6.03081899e+02,  2.76125110e+01,  1.12446419e+01, -1.76738553e+00,
         8.09552912e+00, -3.25787501e-02, -2.71456735e-01, -1.76738553e+00,
        -4.94669105e-01,  5.37801419e-01, -5.11076160e+01,  0.00000000e+00]),
 ([-5.96617834e+02,  2.96346302e+01,  9.77509508e+00, -1.97206163e+00,
         1.88473359e+01,  1.44824398e-01, -3.91520110e-01, -1.97206163e+00,
        -9.98275423e-01,  1.36651709e-02, -6.29544151e+01, -1.51814782e+01]),
 ([-4.95375071e+02,  2.67841275e+01,  3.69632300e+00,  6.59739957e+00,
         1.24465762e+01,  1.95765585e-01,  9.69264054e-02,  6.59739957e+00,
        -9.36600099e-01, -1.46748014e-02, -7.05549661e+01, -4.05359364e+00]),
 ([-4.51446510e+02,  2.25456976e+01, -1.97232853e+00,  1.83566373e+00,
         6.17679098e+00,  1.00198105e+00,  3.99937753e-01,  1.83566373e+00,
        -7.99296690e-01,  2.34830623e-01, -4.84145997e+01,  0.00000000e+00]),
 ([-4.30561042e+02,  2.73384721e+01, -3.07262780e+00,  3.64728141e-01,
         5.24908953e+00,  5.77006782e-01,  7.18623157e-01,  3.64728141e-01,
        -6.68489649e-01, -7.93440373e-03, -4.73811284e+01,  0.00000000e+00]),
 ([-4.47119306e+02,  2.60560541e+01,  5.23128608e-01,  8.71104632e+00,
         7.73317834e+00,  9.21590165e-02,  6.36220541e-01,  8.71104632e+00,
        -9.40925729e-02, -2.51947269e-01, -4.85842349e+01,  0.00000000e+00]),
 ([-4.51906123e+02,  2.51480374e+01,  2.43871727e+00, -1.12832903e+01,
         1.42481075e+01,  4.66832469e-01,  2.07487186e-01, -1.12832903e+01,
        -2.10294380e-01, -3.71414328e-01, -4.20857901e+01,  0.00000000e+00]),
 ([-3.88757107e+02,  2.60170646e+01,  1.57421234e+00, -8.01202489e+00,
         1.34498518e+01,  3.04595463e-01,  3.50877154e-01, -8.01202489e+00,
        -5.19724760e-01, -4.62705569e-01, -4.39391236e+01, -2.18099011e+01]),
 ([-4.07499578e+02,  2.99751884e+01,  9.78015738e-01, -8.82434942e+00,
         2.07151803e+01,  7.61881375e-01, -2.34164395e-02, -8.82434942e+00,
        -1.63014073e+00, -1.96940027e-01, -2.64370332e+01,  0.00000000e+00]),
 ([-2.99078319e+02,  2.58137860e+01,  1.42418469e-01, -1.99745850e+00,
         1.43312105e+01,  1.15460806e-01,  2.72311352e-01, -1.99745850e+00,
        -1.34870582e+00, -1.25091051e-01, -3.28398680e+01,  0.00000000e+00]),
 ([-2.21983592e+02,  1.81694832e+01,  1.60388523e+00, -3.83351797e+00,
         9.27632811e+00, -3.06099902e-01,  3.99934269e-01, -3.83351797e+00,
        -7.51764310e-01, -1.49282282e-01, -3.72116063e+01,  0.00000000e+00]),
 ([-2.21198382e+02,  1.63227474e+01,  2.06457732e+00, -2.00714761e+00,
         1.49788598e+01, -1.08639091e-01,  1.34686860e-01, -2.00714761e+00,
        -8.33000692e-01, -2.59185359e-01, -2.99891590e+01,  2.02605928e+00]),
 ([-2.10686210e+02,  1.56892660e+01,  2.71936566e+00, -3.32341076e+00,
         1.63124588e+01, -8.80673574e-02,  4.46357137e-02, -3.32341076e+00,
        -8.82393295e-01, -3.37093668e-01, -2.30106886e+01,  1.99108133e+01]),
 ([-1.60632154e+02,  1.39432957e+01,  1.59478484e+00,  6.79980406e-01,
         1.01005061e+01, -6.72657381e-02,  1.22566892e-01,  6.79980406e-01,
        -9.81588601e-01, -2.80461724e-02, -1.91845967e+01,  5.11441655e+01]),
 ([-1.36745077e+02,  9.25231749e+00,  2.01853616e+00, -5.98363490e-01,
         1.26101883e+01, -9.64376489e-02,  1.23082191e-01, -5.98363490e-01,
        -6.70447140e-01, -3.39820324e-01, -1.80448973e+01,  0.00000000e+00]),
 ([-1.09608253e+02,  5.53073429e+00,  5.37605252e-01, -1.83650225e+00,
         8.44813627e+00,  8.71284903e-02,  1.80481974e-01, -1.83650225e+00,
        -1.08420616e-01, -3.24004900e-01, -9.16789743e+00,  1.52557917e+01]),
 ([-97.33560098,   5.94632778,   1.16257293,   1.70223096,
          8.09831411,  -0.10074638,   0.1719641 ,   1.70223096,
         -0.11727388,  -0.38089449, -10.80564091,  -7.34537372]),
 ([-1.01572789e+02,  7.67628354e+00,  2.04344293e+00,  1.13001664e+00,
         1.03849692e+01, -1.81381497e-01,  9.32166831e-02,  1.13001664e+00,
        -3.92001747e-01, -4.26742276e-01, -1.38232240e+01,  0.00000000e+00]),
 ([-9.18886251e+01,  6.12976221e+00,  1.96267666e+00,  1.02049744e+00,
         9.26870682e+00, -1.43888475e-01,  4.82285322e-02,  1.02049744e+00,
        -2.61082758e-01, -3.41325283e-01, -1.21154079e+01,  0.00000000e+00]),
 ([-8.60538942e+01,  5.93111642e+00,  1.32752360e+00, -2.23586154e+00,
         1.00794162e+01,  7.71609518e-03,  5.77044041e-03, -2.23586154e+00,
        -4.06560397e-01, -3.08866134e-01, -1.22842550e+01,  0.00000000e+00]),
 ([-5.70853215e+01,  5.37462704e+00,  4.19047074e-01, -1.03424731e+00,
         9.10419297e+00, -5.41350339e-02,  5.46472181e-02, -1.03424731e+00,
        -5.74751315e-01, -2.56999547e-01, -1.44326716e+01, -1.64854738e+00]),
 ([-4.65990979e+01,  4.20188497e+00, -8.31470602e-02, -1.64581525e+00,
         8.21794696e+00, -1.37783530e-03,  3.67804352e-02, -1.64581525e+00,
        -4.57759638e-01, -2.50035963e-01, -8.87295751e+00,  0.00000000e+00]),
 ([-2.25251850e+01,  1.19387845e+00, -5.82803975e-01,  1.88271284e-02,
         5.40552259e+00,  5.80476635e-02,  4.14908937e-02,  1.88271284e-02,
        -1.76654890e-01, -2.08583202e-01, -5.19233548e+00,  0.00000000e+00]),
 ([-9.25169738,  0.52823673, -0.30377188, -0.59517221,  2.97691243,
         0.02554511,  0.01591453, -0.59517221, -0.10268038, -0.14087296,
        -1.73656711,  0.        ]),
 ([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]
# Convert to numpy array
theta_array = np.array(theta_list)

# Define feature names
feature_names = [
    "Bias Term",
    "Wind",
    "Price",
    "Electrolyzer Status",
    "Hydrogen Level",
    "Wind × Price",
    "Hydrogen × Price",
    "Electrolyzer Cost",
    "Wind × Hydrogen",
    "Hydrogen²",
    "ReLU(5-Wind)",
    "ReLU(Price-35)"
]

# Time steps
time_steps = list(range(len(theta_list)))

# Create a color palette with more colors
n_colors = len(feature_names)
palette = sns.color_palette("husl", n_colors)

# Visualization 1: Parameters evolution over time
plt.figure(figsize=(15, 10))
for i, feature in enumerate(feature_names):
    plt.plot(time_steps, theta_array[:, i], label=feature, color=palette[i], linewidth=2, alpha=0.8)

plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Parameter Value', fontsize=14)
plt.title('Evolution of Value Function Parameters Over Time', fontsize=18)
plt.grid(True)
plt.xticks(time_steps)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('parameter_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 2: Parameters evolution over time (separate by feature)
fig, axes = plt.subplots(4, 3, figsize=(20, 15), sharex=True)
axes = axes.flatten()

for i, feature in enumerate(feature_names):
    if i < len(axes):
        ax = axes[i]
        ax.plot(time_steps, theta_array[:, i], color=palette[i], linewidth=3)
        ax.set_title(feature, fontsize=14)
        ax.grid(True)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add a small annotation of the final non-zero value
        # Find the last non-zero value
        last_non_zero_idx = np.where(theta_array[:, i] != 0)[0]
        if len(last_non_zero_idx) > 0:
            last_non_zero_idx = last_non_zero_idx[-1]
            last_value = theta_array[last_non_zero_idx, i]
            ax.annotate(f'{last_value:.2f}', 
                        xy=(last_non_zero_idx, last_value),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=10,
                        color='red')

# Remove empty subplots if there are any
for i in range(len(feature_names), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
fig.text(0.5, 0.04, 'Time Step', ha='center', fontsize=16)
fig.text(0.04, 0.5, 'Parameter Value', va='center', rotation='vertical', fontsize=16)
plt.suptitle('Individual Parameter Evolution', fontsize=22, y=1.02)
plt.savefig('individual_parameters.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 3: Heatmap of parameter values over time
# Skip the bias term (first column) and normalize the data for better visualization
data_for_heatmap = theta_array[:, 1:]
feature_names_for_heatmap = feature_names[1:]

# Create a pandas DataFrame for better labeling
df_heatmap = pd.DataFrame(data_for_heatmap, 
                         index=[f'Time {t}' for t in time_steps],
                         columns=feature_names_for_heatmap)

plt.figure(figsize=(15, 12))
# Create a custom diverging colormap
cmap = LinearSegmentedColormap.from_list('custom_diverging', 
                                         ['#d73027', '#f7f7f7', '#1a9641'], 
                                         N=256)

# Use a symmetric colormap around zero
vmax = np.abs(data_for_heatmap).max()
sns.heatmap(df_heatmap, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
            annot=False, linewidths=0.5, fmt=".2f",
            xticklabels=feature_names_for_heatmap, 
            yticklabels=df_heatmap.index)

plt.title('Heatmap of Feature Parameters Over Time', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('parameter_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 4: Bar plot for a specific time step
selected_time_step = 12  # Change this to visualize different time steps

plt.figure(figsize=(14, 8))
bars = plt.bar(feature_names, theta_array[selected_time_step], color=[
    'red' if val < 0 else 'green' for val in theta_array[selected_time_step]
])

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title(f'Parameter Values at Time Step {selected_time_step}', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.ylabel('Parameter Value', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., 
             height + (0.1 if height >= 0 else -5),
             f'{height:.2f}',
             ha='center', va='bottom' if height >= 0 else 'top',
             rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig(f'parameters_at_timestep_{selected_time_step}.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 5: Most significant parameters at each time step
# Calculate the absolute values to identify the most significant parameters
abs_theta = np.abs(theta_array)

# Get the top 3 most significant parameters at each time step
top_n = 3
most_significant_params = []

for t in range(len(theta_list) - 1):  # Exclude the last time step (all zeros)
    time_data = abs_theta[t]
    top_indices = np.argsort(time_data)[-top_n:][::-1]  # Get indices of top n values
    
    # Original values (not absolute)
    top_values = theta_array[t, top_indices]
    
    # Get feature names
    top_features = [feature_names[i] for i in top_indices]
    
    most_significant_params.append({
        'time_step': t,
        'features': top_features,
        'values': top_values
    })

# Create a DataFrame for easier visualization
significant_data = []
for entry in most_significant_params:
    for i in range(len(entry['features'])):
        significant_data.append({
            'Time Step': entry['time_step'],
            'Feature': entry['features'][i],
            'Value': entry['values'][i],
            'Abs Value': abs(entry['values'][i]),
            'Rank': i + 1  # 1 = most significant
        })

df_significant = pd.DataFrame(significant_data)

# Plot
plt.figure(figsize=(16, 10))

for rank in range(1, top_n + 1):
    rank_data = df_significant[df_significant['Rank'] == rank]
    plt.scatter(rank_data['Time Step'], rank_data['Value'], 
                s=rank_data['Abs Value'] * 5, 
                alpha=0.7,
                label=f'Rank {rank}')
    
    # Add feature names as annotations
    for _, row in rank_data.iterrows():
        plt.annotate(row['Feature'], 
                     xy=(row['Time Step'], row['Value']),
                     xytext=(5, 5 if row['Value'] >= 0 else -15),
                     textcoords='offset points',
                     fontsize=8,
                     alpha=0.8)

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title(f'Most Significant Parameters at Each Time Step (Top {top_n})', fontsize=18)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Parameter Value', fontsize=14)
plt.xticks(time_steps[:-1])  # Exclude the last time step (all zeros)
plt.grid(True, alpha=0.3)
plt.legend(title='Significance Rank')
plt.tight_layout()
plt.savefig('most_significant_parameters.png', dpi=300, bbox_inches='tight')
plt.show()