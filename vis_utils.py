import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as path_effects
import colorsys
import re
import seaborn as sns
from collections import Counter
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import matplotlib.colors as mcolors
from utils import get_topk_acc

def safe_title(title):
    """Sanitize title for a safe filename by removing invalid characters and extra underscores."""
    title = re.sub(r'[^\w\s-]', '', title)  # Remove invalid characters
    title = re.sub(r'\s+', '_', title)  # Replace spaces with underscores
    title = re.sub(r'_+', '_', title)  # Remove multiple consecutive underscores
    return title.strip('_')  # Remove leading/trailing underscores


def visualize_beam_search(generated_history, iclass, prefix_pred, cfg):
    """
    Visualize beam search history as a heatmap with text annotations.

    Args:
        generated_history (list): The beam search history for each timestep.
        iclass: An instance with a decode_ids method to decode beam IDs.
        prefix_pred (list): The prefix to compare against for bold text.
        cfg: Configuration object containing inference parameters.
    """
    def change_to_c(value:list):
        r = []
        for item in value:
            try:
                blieb = float(item)
                if blieb in [-1, 1, 2, ]:
                    r.append(item)
                    continue
                r.append("c")
            except ValueError:
                r.append(item)
        # print(r)      
        return r

    # Decode the beam sequences
    beam_sequences = []
    for timestep in range(len(generated_history)):
        timestepie = []
        for beam in range(len(generated_history[timestep])):
            decoded = iclass.decode_ids(generated_history[timestep][beam])
            timestepie.append(decoded[1:])
        beam_sequences.append(timestepie)

    beam_matrix = np.array(beam_sequences)

    # Create a matrix to detect changes
    change_matrix = np.ones((beam_matrix.shape[0], beam_matrix.shape[1]), dtype=int)
    for i in range(1, beam_matrix.shape[0]):
        for j in range(0, beam_matrix.shape[1]):
            change_matrix[i, j] = np.all(beam_matrix[i, j, :i-1] == beam_matrix[i - 1, j, :i-1])

    beam_matrix = np.array([
        [" ".join(map(str, beam)).rstrip(' <PAD>') for beam in step]
        for step in beam_sequences
    ])

    for i in range(change_matrix.shape[0]):
        for j in range(change_matrix.shape[1]):
            value = beam_matrix[i, j]
            if value.split(" ") == change_to_c(prefix_pred):
                print(value.split(" "), prefix_pred)
                change_matrix[i, j] = 2

    fig, ax = plt.subplots(figsize=(6 * cfg.inference.beam_size, 5))
    cmap = ListedColormap([[.8, .4, .4], 'white', 'green'])

    for i in range(change_matrix.shape[0]):
        for j in range(change_matrix.shape[1]):
            value = beam_matrix[i, j]
            color = 'black'
            weight = 'normal'
            ax.text(j, i, value, ha='center', va='center', fontsize=8, color=color, weight=weight)

    if np.max(change_matrix) == 1:
        change_matrix[0, 0] = 2
    im = ax.imshow(change_matrix.astype(float), cmap=cmap, aspect='auto')

    # Formatting the plot
    ax.set_title("Beam Search Visualization")
    ax.set_xlabel("Beam Index")
    ax.set_ylabel("Time Step")
    ax.set_xticks(np.arange(beam_matrix.shape[1]))
    ax.set_yticks(np.arange(beam_matrix.shape[0]))
    plt.show()

def importance_map(data, title='Importance of attention heads in the circuit',
                   threshold=0.0, show_threshold="positive",
                   log_scale=False,
                   save_fig=False,
                   save_path="pictures/circuitfinding/usageofcomponentsinallcircuits",
                   show_border=True,
                   color_way=True):


    big_size = 30
    small_size = 27
    
    plot_data = np.log1p(np.abs(data)) if log_scale else data
    vmax = np.max(plot_data)
    vmin = np.min(plot_data) if not log_scale else 0  # log1p always ≥ 0

    plt.figure(figsize=(12, 10))
    im = plt.imshow(plot_data, cmap='Blues', aspect='auto', vmin=vmin, vmax=vmax)
    if color_way:
        cbar = plt.colorbar(im)
        
        cbar.set_label('Logit Change', fontsize=big_size, weight="bold")
        # cbar.set_label('# Circuits Using Component', fontsize=23, weight="bold")
        cbar.ax.tick_params(labelsize=small_size)

    plt.xlabel('Attention Heads', fontsize=25, weight="bold")
    plt.ylabel('Layers (2 MABs per Layer)', fontsize=25, weight="bold")
    plt.title(title, fontsize=big_size, weight='bold')
    plt.xticks(ticks=np.arange(9), labels=['MLP'] + [f'H{i+1}' for i in range(8)], fontsize=small_size)
    plt.tick_params(axis='x', length=10, width=2)
    plt.tick_params(axis='y', length=10, width=2)
    plt.yticks(ticks=np.arange(13), labels=[f'L{i//2+1}-MAB{i%2+1}' for i in range(12)] + ["OUT"], fontsize=small_size)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            raw_value = data[i, j]
            log_value = np.log1p(abs(raw_value)) if log_scale else raw_value
            show = (
                (show_threshold == "both" and abs(raw_value) > threshold) or
                (show_threshold == "positive" and raw_value > threshold) or
                (show_threshold == "negative" and raw_value < -threshold)
            )
            if show:
                display_val = f'{log_value:.2f}' if log_scale else f'{raw_value:.0f}'
                text = plt.text(j, i, display_val, ha='center', va='center',
                                fontsize=12, color='white', fontweight='bold')
                text.set_path_effects([
                    path_effects.Stroke(linewidth=1.5, foreground='black'),
                    path_effects.Normal()
                ])

    max_val = np.max(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == max_val and show_border:
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         linewidth=2.5, edgecolor='gold', facecolor='none', zorder=120)
                plt.gca().add_patch(rect)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")

    if save_fig:
        plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}.png", format="png", bbox_inches="tight")

    plt.show()


def importance_mapall(data, title='Importance of attention heads in the circuit',
                   threshold=0.0, show_threshold="positive",
                   log_scale=False,
                   save_fig=False,
                   save_path="pictures/circuitfinding/usageofcomponentsinallcircuits",
                   show_border=True,
                   color_way=True,
                   ax=None,
                   individual=False):

    big_size = 30
    small_size = 27

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure

    plot_data = np.log1p(np.abs(data)) if log_scale else data
    vmax = np.max(plot_data)
    vmin = np.min(plot_data) if not log_scale else 0

    base_rgb = (0.1, 0.4, 0.8) 

    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_blue_cmap',
        [(1, 1, 1), base_rgb]
    )

    im = ax.imshow(plot_data, cmap=custom_cmap, aspect='auto', vmin=vmin, vmax=1)

    thresholds = [threshold, threshold * 2, threshold * 3]
    activated_coords = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val > thresholds[2]:
                if individual:
                    color = (0.2, 0.5, 0.9, 0.5)  
                else:
                    color = (0.9, 0, 0, 0.5)     
                activated_coords.append((i, j, color))
            elif val > thresholds[1]:
                if individual:
                    color = (0.2, 0.5, 0.9, 0.3)  
                else:
                    color = (0.5, 0, 0, 0.3)       
                activated_coords.append((i, j, color))
            elif val > thresholds[0]:
                if individual:
                    color = (0.2, 0.5, 0.9, 0.2)
                else:
                    color = (0.2, 0, 0, 0.2)    
                activated_coords.append((i, j, color))

    for (i, j, c) in activated_coords:
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 linewidth=0, edgecolor=None, facecolor=c, zorder=50)
        ax.add_patch(rect)

    if ax is None and color_way:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Logit Change', fontsize=big_size, weight="bold")
        cbar.ax.tick_params(labelsize=small_size)

    ax.set_xlabel('Attention Heads', fontsize=25, weight="bold")
    ax.set_ylabel('Layers (2 MABs per Layer)', fontsize=25, weight="bold")
    ax.set_title(title, fontsize=big_size, weight='bold')

    ax.set_xticks(np.arange(9))
    ax.set_xticklabels(['MLP'] + [f'H{i+1}' for i in range(8)], fontsize=small_size)
    ax.tick_params(axis='x', length=10, width=2)

    ax.set_yticks(np.arange(13))
    ax.set_yticklabels([f'L{i//2+1}-MAB{i%2+1}' for i in range(12)] + ["OUT"], fontsize=small_size)
    ax.tick_params(axis='y', length=10, width=2)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            raw_value = data[i, j]
            log_value = np.log1p(abs(raw_value)) if log_scale else raw_value
            show = (
                (show_threshold == "both" and abs(raw_value) > threshold) or
                (show_threshold == "positive" and raw_value > threshold) or
                (show_threshold == "negative" and raw_value < -threshold)
            )
            if show:
                display_val = f'{log_value:.2f}' if log_scale else f'{raw_value:.0f}'
                text = ax.text(j, i, display_val, ha='center', va='center',
                               fontsize=12, color='white', fontweight='bold')
                text.set_path_effects([
                    path_effects.Stroke(linewidth=1.5, foreground='black'),
                    path_effects.Normal()
                ])

    max_val = np.max(data)
    if show_border:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] == max_val:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                             linewidth=2.5, edgecolor='gold', facecolor='none', zorder=120)
                    ax.add_patch(rect)

    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")

    if save_fig and ax is None:
        fig.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(f"{save_path}.png", format="png", bbox_inches="tight")

    if ax is None:
        plt.show()


def importance_map_2(data, title='Importance of attention heads in the circuit', idx_list=None):
    plt.figure(figsize=(12, 6))
    
    im = plt.imshow(data, cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(data)), vmax=np.max(np.abs(data)))
    plt.colorbar(label='Mean Results')

    plt.xlabel('Attention Heads')
    plt.ylabel('Layers (2 MABs per Layer)')

    plt.xticks(ticks=np.arange(11), labels=['MLP', "ln1", "ln2"] + [f'H{i+1}' for i in range(8)])
    plt.yticks(ticks=np.arange(13), labels=[f'L{i//2+1}-MAB{i%2+1}' for i in range(12)] + ["OUT"])

    if idx_list is not None:
        for idx in idx_list:
            i, j = divmod(idx, data.shape[1])
            if 0 <= i < data.shape[0] and 0 <= j < data.shape[1]:
                value = data[i, j]
                text = plt.text(j, i, f'{value:.2f}', ha='center', va='center', 
                                fontsize=10, color='white', fontweight='bold')
                text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'),
                                       path_effects.Normal()])

    plt.tight_layout()
    plt.show()


def importance_map_3(highlight_idx, shape=(13, 9), title='Highlighted Elements', x_labels=None, y_labels=None):
    data = np.ones(shape) * np.nan

    plt.figure(figsize=(12, 6))
    plt.imshow(data, cmap='gray', aspect='auto', vmin=0, vmax=1)

    for idx in highlight_idx:
        i, j = divmod(idx, shape[1])
        if 0 <= i < shape[0] and 0 <= j < shape[1]:
            plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=(0.1, 0.6, 0.8),))

    plt.title(title, fontsize=16, weight='bold')

    if x_labels is None:
        x_labels = ['MLP',] + [f'H{i+1}' for i in range(shape[1] - 1)]
    if y_labels is None:
        y_labels = [f'L{i//2+1}-MAB{i%2+1}' for i in range(shape[0] - 1)] + ["OUT"]

    plt.xticks(ticks=np.arange(shape[1]), labels=x_labels, ha='right')
    plt.yticks(ticks=np.arange(shape[0]), labels=y_labels)

    plt.tight_layout()
    plt.show()


def plot_frequency(sorted_keys, sorted_values, title="Frequency plot",
                   color=(0.2, 0.5, 0.9), x_axis_name="Elements", y_axis_name="Frequency",
                   show_percentage=False, save_fig=True, save_name=False, height_text_adjust=0.5):
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(sorted_keys, sorted_values, color=color, edgecolor='black', linewidth=1.2)

    big_size = 27
    small_size = 24
    ax.set_title(title, fontsize=big_size, weight='bold')
    ax.set_xlabel(x_axis_name, fontsize=small_size, weight='bold')
    ax.set_ylabel(y_axis_name, fontsize=small_size, weight='bold')

    ax.tick_params(axis='x', which='both', labelsize=small_size-2, width=3, length=6)
    ax.set_xticklabels(sorted_keys, weight='bold')
    ax.tick_params(axis='y', which='both', labelsize=small_size-2, width=3, length=6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    total = sum(sorted_values)

    for bar, value in zip(bars, sorted_values):
        height = bar.get_height()
        if show_percentage:
            text = f'{(value / total) * 100:.3f}%'
        else:
            text = f'{value:.0f}'  # Absolute count

        ax.text(
            bar.get_x() + bar.get_width() / 2,  
            height + height_text_adjust,  
            text,
            ha='center', va='bottom', fontsize=small_size-5, color='black', weight='bold'
        )

    print("total_entries:", total)
    plt.tight_layout()
    if save_fig:
        if not save_name:
            plt.savefig(f"pictures/modelperformance/{safe_title('_'.join(title.split(' ')))}.pdf", format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_name, format="pdf", bbox_inches="tight")
    plt.show()
    
def plot_confusion_matrix_performance(debugFullModel, debugCircuit, title='Confusion Matrix of Predictions', threshold=0.0, show_threshold='positive'):
    full_model_counts = Counter(debugFullModel)
    circuit_counts = Counter(debugCircuit)

    labels = sorted(list(set(debugFullModel))) + ['4+']

    conf_matrix = np.zeros((len(labels), len(labels)-1))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for a, b in zip(debugFullModel, debugCircuit):
        if b > 3:
            conf_matrix[label_to_idx['4+'], label_to_idx[a]] += 1
            continue
        conf_matrix[label_to_idx[b], label_to_idx[a]] += 1

    plt.figure(figsize=(12, 6))
    im = plt.imshow(conf_matrix, cmap='Blues', aspect='auto', vmin=0)
    cbar = plt.colorbar(im)  
    cbar.set_label('Count', fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    plt.ylabel('Circuit Model Predictions', fontsize=16)
    plt.xlabel('Full Model Predictions', fontsize=16)
    plt.title(title, fontsize=20, weight='bold')

    plt.xticks(ticks=np.arange(len(labels)-1), labels=labels[:-1], fontsize=16)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=16)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            value = conf_matrix[i, j]
            text = plt.text(j, i, f'{int(value)}', ha='center', va='center', 
                            fontsize=16, color='white', fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'),
                                    path_effects.Normal()])

    plt.tight_layout()
    plt.show()


def generate_shades(base_color, n):
    h, l, s = colorsys.rgb_to_hls(*base_color)
    shades = []
    l_min = max(0, l - 0.2)
    l_max = min(1, l + 0.2)
    if n == 1:
        return [base_color]
    for i in range(n):
        new_l = l_min + (l_max - l_min) * (i / (n - 1))
        new_color = colorsys.hls_to_rgb(h, new_l, s)
        shades.append(new_color)
    return shades


def plot_frequency_stacked(names, heights, title="Frequency plot",
                   stacked_base_color=(0.2, 0.5, 0.9),
                   bar_colors=None,
                   x_axis_name="Elements", y_axis_name="Frequency", show_percentage=True,
                   title_padding=20, legend_padding=1.5, legend_include_percentage=True,
                   fig_size=(12,6),
                   save_fig=True,
                   save_name=False,
                   legend_cols=3, adjust_x_ticks=0):
    """
    Plots bars (stacked or single) and annotates each with its percentage relative to the overall total.
    """
    big_size = 27
    small_size = 24

    overall_total = 0
    height = []
    for h in heights:
        if isinstance(h, (list, tuple)):
            overall_total += sum(h)
            height.append(sum(h))
        else:
            overall_total += h
            height.append(h)

    fig, ax = plt.subplots(figsize=fig_size)
    n_bars = len(names)
    x_positions = np.arange(n_bars)
    x_tick_labels = []

    default_bar_colors = [(0.8, 0.2, 0.1), (0.6, 0.4, 0.8), (0.8, 0.8, 0.8), (0.9, 0.7, 0.0), (0.3, 0.7, 0.3), ]
    if bar_colors is None:
        bar_colors = default_bar_colors

    legend_handles = []
    legend_labels = []
    stacked_legend_added = False

    for i, (name_item, height_item) in enumerate(zip(names, heights)):
        if isinstance(name_item, dict):
            if len(name_item) != 1:
                raise ValueError("When using a dict for a name, it must have exactly one key")
            x_label = list(name_item.keys())[0]
            seg_legend_names = name_item[x_label]
            x_tick_labels.append(x_label)

            if not isinstance(height_item, (list, tuple)):
                raise ValueError("For a stacked bar (dict name), heights must be provided as a list/tuple")
            bar_total = sum(height_item)
            percent_bar = bar_total / overall_total * 100

            shades = generate_shades(stacked_base_color, len(height_item))

            bottom = 0
            for j, seg_value in enumerate(height_item):
                seg_color = shades[j]
                bar_plot = ax.bar(x_positions[i], seg_value, bottom=bottom,
                                  color=seg_color, edgecolor='black', linewidth=1.2)
                bottom += seg_value

                if not stacked_legend_added:
                    seg_percent = seg_value / overall_total * 100
                    if j < len(seg_legend_names):
                        if legend_include_percentage:
                            legend_label = f"{seg_legend_names[j]} ({seg_percent:.1f}%)"
                        else:
                            legend_label = f"{seg_legend_names[j]}"
                    else:
                        if legend_include_percentage:
                            legend_label = f"Segment {j+1} ({seg_percent:.1f}%)"
                        else:
                            legend_label = f"{seg_legend_names[j]}"
                    legend_handles.append(bar_plot[0])
                    legend_labels.append(legend_label)
            stacked_legend_added = True

            if show_percentage:
                ax.text(x_positions[i], bar_total + overall_total*0.005, f'{percent_bar:.1f}%',
                        ha='center', va='bottom', fontsize=small_size, color='black', weight='bold')
            else:
                ax.text(x_positions[i], bar_total + overall_total*0.005, f'{height[i]:.0f}',
                        ha='center', va='bottom', fontsize=small_size, color='black', weight='bold')
        else:
            x_label = name_item
            x_tick_labels.append(x_label)
            if not isinstance(height_item, (int, float)):
                raise ValueError("For a simple bar, height must be a number")
            bar_total = height_item
            percent_bar = bar_total / overall_total * 100
            color_idx = i % len(bar_colors)
            bar_plot = ax.bar(x_positions[i], bar_total,
                              color=bar_colors[color_idx], edgecolor='black', linewidth=1.2)
            if show_percentage:
                ax.text(x_positions[i], bar_total + overall_total*0.005, f'{percent_bar:.1f}%',
                        ha='center', va='bottom', fontsize=small_size, color='black', weight='bold')
            else:
                ax.text(x_positions[i], bar_total + overall_total*0.005, f'{height[i]:.0f}',
                        ha='center', va='bottom', fontsize=small_size, color='black', weight='bold')

    ax.set_xticks(x_positions)
    if not adjust_x_ticks:
        ax.set_xticklabels(x_tick_labels, fontsize=small_size, weight='bold')
    else:
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=small_size, weight='bold')
    
    ax.set_title(title, fontsize=big_size, weight='bold', pad=title_padding)
    ax.set_xlabel(x_axis_name, fontsize=small_size, weight='bold')
    ax.set_ylabel(y_axis_name, fontsize=small_size, weight='bold')

    ax.tick_params(axis='x', which='both', labelsize=small_size-2, width=3, length=6, )
    ax.tick_params(axis='y', which='both', labelsize=small_size-2, width=3, length=6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    if legend_handles:
        ax.legend(
            legend_handles, legend_labels,
            fontsize=small_size - 2,
            loc='upper center',
            bbox_to_anchor=(0.5, legend_padding),
            ncol=legend_cols,
            frameon=True
        )

    if adjust_x_ticks:
        ax.legend(
            legend_handles, legend_labels,
            fontsize=small_size - 2,
            loc='upper right',
            ncol=1,
            frameon=True
        )

    print("Total entries:", overall_total)
    plt.tight_layout()
    if save_fig:
        if save_name is None:
            plt.savefig(f"../../pictures/modelperformance/{safe_title('_'.join(title.split(' ')))}.pdf", format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_name, format="pdf", bbox_inches="tight")

    plt.show()


def plot_violin(data, labels, title="Violin Plot", y_axis_name="Values",
                x_axis_name="Category", violin_color=[.1, .6, .8], save_fig=True,
                save_name=False):

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.violinplot(data=data, ax=ax, inner=None, linewidth=2.5, 
                   scale="width", facecolor=violin_color, edgecolor="black")

    means = [np.mean(d) for d in data]
    ax.scatter(range(len(data)), means, color='black', edgecolors='white',
               s=150, linewidth=2.5, label="Mean", zorder=3)


    big_size = 27
    small_size = 24
    ax.set_title(title, fontsize=big_size, weight='bold', color='black')
    ax.set_xlabel(x_axis_name, fontsize=small_size, weight='bold', color='black')
    ax.set_ylabel(y_axis_name, fontsize=small_size, weight='bold', color='black')


    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=small_size - 2, weight='bold', color='black')

    ax.tick_params(axis='y', which='both', labelsize=small_size - 2, 
                   width=2, length=6, color='black', labelcolor='black')
    ax.tick_params(axis='x', which='both', labelsize=small_size - 2, 
                   width=2, length=6, color='black', labelcolor='black')

    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(loc="upper right", fontsize=small_size - 6, frameon=True, facecolor='white', edgecolor='black')

    plt.tight_layout()
    if save_fig:
        if not save_name:
            plt.savefig(f"pictures/modelperformance/{safe_title('_'.join(title.split(' ')))}.pdf", format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_name, format="pdf", bbox_inches="tight")
    plt.show()


def plot_distribution(data, smooth=True, fill=True, scale='linear',
                      title="Distribution", x_axis_name="Value", y_axis_name="Frequency",
                      line_color=[0.0, 0.0, 0.0], linthresh=1.0, save_fig=True, save_name=False):

    counts = Counter(data)
    x = np.array(sorted(counts.keys()))
    y = np.array([counts[val] for val in x])

    fig, ax = plt.subplots(figsize=(12, 6))

    if smooth and len(x) >= 3:
        x_new = np.linspace(x.min(), x.max(), 300)
        if scale in ['log', 'symlog']:
            safe_y = np.where(y > 0, y, 1e-9) 
            y_log = np.log(safe_y)
            spline = make_interp_spline(x, y_log, k=2)
            y_smooth = np.exp(spline(x_new))
        else:
            spline = make_interp_spline(x, y, k=2)
            y_smooth = spline(x_new)
        plot_x, plot_y = x_new, y_smooth
    else:
        plot_x, plot_y = x, y

    ax.plot(plot_x, plot_y, color=line_color, linewidth=3, label="Distribution")

    if fill:
        ax.fill_between(plot_x, plot_y, 0, color=(0.2, 0.5, 0.9), alpha=1)

    if scale == 'log':
        ax.set_yscale('log')
        ylabel = f"{y_axis_name} (log scale)"
    elif scale == 'symlog':
        ax.set_yscale('symlog', linthresh=linthresh)
        ylabel = f"{y_axis_name} (log scale)"
    else:
        ylabel = y_axis_name

    big_size = 27
    small_size = 24

    ax.set_title(title, fontsize=big_size, weight='bold', color='black')
    ax.set_xlabel(x_axis_name, fontsize=small_size, weight='bold', color='black')
    ax.set_ylabel(ylabel, fontsize=small_size, weight='bold', color='black')

    ax.tick_params(axis='y', which='both', labelsize=small_size - 2,
                   width=2, length=6, color='black', labelcolor='black')
    ax.tick_params(axis='x', which='both', labelsize=small_size - 2,
                   width=2, length=6, color='black', labelcolor='black')
    ax.set_xticklabels([0, 0, 5, 10, 15, 20, 25, 30], fontsize=small_size - 2, weight='bold', color='black')

    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(left=-0.2)
    ax.set_ylim(bottom=0)
    plt.margins(0)

    plt.tight_layout()
    if save_fig:
        if not save_name:
            plt.savefig(f"pictures/modelperformance/{safe_title('_'.join(title.split(' ')))}.pdf", format="pdf", bbox_inches="tight")
        else:
            plt.savefig(save_name, format="pdf", bbox_inches="tight")
    plt.show()


def plot_topk_accuracies(performances_file="performances.npy", x_values=None):
    """
    Plots the Top-k Accuracy vs. Number of Heads Included with a modernized style.

    Args:
        performances_file (str): Path to the saved performances .npy file.
        x_values (list, optional): List of x-axis values. Defaults to range(1, 101, 5).
    """
    performances = np.load(performances_file, allow_pickle=True)

    if x_values is None:
        x_values = list(range(0, 100, 5))
    else:
        x_values = list(x_values)

    filtered_performances = [performances[i] for i in x_values]

    topk_accuracies = {k: [get_topk_acc(filtered_performances[show], k) for show in range(len(x_values))] for k in range(1, 11)}

    fig, ax = plt.subplots(figsize=(12, 7))

    line_styles = ['o-', 's-', '^-', 'd-', 'x-', 'P-', '*-', 'h-', '+-', 'v-']
    colors = generate_shades((0.1, 0.6, 0.8), 10)

    for (k, y_values), style, color in zip(topk_accuracies.items(), line_styles, colors):
        ax.plot(x_values, y_values, style, label=f"Top-{k}", markersize=8, linewidth=3, color=color)

    big_size = 27
    small_size = 24
    ax.set_title("Top-k Accuracy vs. Number of Heads Included", fontsize=big_size, weight='bold', color='black')
    ax.set_xlabel("Number of Heads Included", fontsize=small_size, weight='bold', color='black')
    ax.set_ylabel("Accuracy", fontsize=small_size, weight='bold', color='black')

    ax.set_xticks(x_values[::5])
    ax.set_xticklabels(x_values[::5], fontsize=small_size - 2, weight='bold', color='black')
    ax.tick_params(axis='y', which='both', labelsize=small_size - 2, width=2, length=6, color='black', labelcolor='black')
    ax.tick_params(axis='x', which='both', labelsize=small_size - 2, width=2, length=6, color='black', labelcolor='black')

    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(loc="lower right", fontsize=small_size - 4, frameon=True, facecolor='white')

    plt.tight_layout()
    plt.show()


def plot_performance(performances):
    """
    Plots the mean performance with standard deviation, styled similarly to plot_topk_accuracies.

    Args:
        performances (list or np.array): List or array of performance values.
    """
    performances = np.array(performances)
    mean_vals = performances.mean(axis=1)
    std_vals = performances.std(axis=1)

    x = np.arange(len(mean_vals))

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(x, mean_vals, label="Mean Performance", markersize=8, linewidth=3, color=(0.1, 0.6, 0.8))
    ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, color=(0.1, 0.4, 0.6), alpha=0.2)

    big_size = 27
    small_size = 24

    ax.set_title("Mean logit difference over nodes patched", fontsize=big_size, weight='bold', color='black')
    ax.set_xlabel("Nodes patched", fontsize=small_size, weight='bold', color='black')
    ax.set_ylabel("Mean logit difference", fontsize=small_size, weight='bold', color='black')

    ax.set_xticks(x[::5])
    ax.set_xticklabels(x[::5], fontsize=small_size - 2, weight='bold', color='black')
    ax.tick_params(axis='y', which='both', labelsize=small_size - 2, width=2, length=6, color='black', labelcolor='black')
    ax.tick_params(axis='x', which='both', labelsize=small_size - 2, width=2, length=6, color='black', labelcolor='black')

    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
