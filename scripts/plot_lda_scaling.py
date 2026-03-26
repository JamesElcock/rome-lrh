"""Plot LDA scaling experiment results."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif'})

data = json.load(open("results/exp4/lda_scaling_v4.json"))
ga = data["grand_average"]

scales = [1, 2, 3, 4, 8, 16, 32, 64]

# Extract data
vmean_eff = [ga[f"vmean_{s}x"]["efficacy"] for s in scales]
vmean_prob = [ga[f"vmean_{s}x"]["target_prob"] for s in scales]

res_eff = [ga[f"residual_{s}x"]["efficacy"] for s in scales]
res_prob = [ga[f"residual_{s}x"]["target_prob"] for s in scales]

lda_eff = [ga[f"lda_{s}x"]["efficacy"] for s in scales]
lda_prob = [ga[f"lda_{s}x"]["target_prob"] for s in scales]

lda_res_eff = [ga[f"lda_{s}x_plus_residual"]["efficacy"] for s in scales]
lda_res_prob = [ga[f"lda_{s}x_plus_residual"]["target_prob"] for s in scales]

conc_eff = [ga[f"concept_{s}x"]["efficacy"] for s in scales]
conc_prob = [ga[f"concept_{s}x"]["target_prob"] for s in scales]

conc_lda_eff = [ga[f"concept_plus_lda_{s}x"]["efficacy"] for s in scales]
conc_lda_prob = [ga[f"concept_plus_lda_{s}x"]["target_prob"] for s in scales]

own_eff = ga["own_v"]["efficacy"]
own_prob = ga["own_v"]["target_prob"]
rand_eff = ga["random_64x"]["efficacy"]
rand_prob = ga["random_64x"]["target_prob"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot order: group logically (single components, combinations, baselines)
for ax, eff_data, title, fmt_val in [
    (ax1,
     dict(res=res_eff, vmean=vmean_eff, lda=lda_eff, conc=conc_eff,
          lda_res=lda_res_eff, conc_lda=conc_lda_eff),
     'Efficacy vs. Component Scale',
     lambda v: f'{v:.2f}'),
    (ax2,
     dict(res=res_prob, vmean=vmean_prob, lda=lda_prob, conc=conc_prob,
          lda_res=lda_res_prob, conc_lda=conc_lda_prob),
     'Target Probability vs. Component Scale',
     lambda v: f'{v:.3f}'),
]:
    # Single components
    ax.plot(scales, eff_data['res'], 'o-', color='#d62728', linewidth=2.5, markersize=8, label='Dark Residual')
    ax.plot(scales, eff_data['vmean'], 'v-', color='#ff7f0e', linewidth=2.5, markersize=8, label='$v_{mean}$')
    ax.plot(scales, eff_data['lda'], '^-', color='#1f77b4', linewidth=2.5, markersize=8, label='LDA')
    ax.plot(scales, eff_data['conc'], 'x-', color='#8c564b', linewidth=2, markersize=8, label='Concept')
    # Combinations
    ax.plot(scales, eff_data['lda_res'], 's-', color='#9467bd', linewidth=2, markersize=7, label='LDA + Dark Residual')
    ax.plot(scales, eff_data['conc_lda'], 'D-', color='#2ca02c', linewidth=2, markersize=7, label='Concept + LDA')
    # Baselines
    own_v = own_eff if ax == ax1 else own_prob
    rand_v = rand_eff if ax == ax1 else rand_prob
    ax.axhline(own_v, color='gray', linestyle=':', linewidth=1.5, label=f'Own $v_i$ ({fmt_val(own_v)})')
    ax.axhline(rand_v, color='lightgray', linestyle='--', linewidth=1, label=f'Random ({fmt_val(rand_v)})')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Scale factor (× $\\|v_{mean}\\|$)')
    ax.set_xticks(scales)
    ax.set_xticklabels([f'{s}×' for s in scales])
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5, ncol=1,
              handlelength=1.8, handletextpad=0.5)

ax1.set_ylabel('Efficacy')
ax2.set_ylabel('Target probability')

plt.tight_layout()
plt.savefig("results/exp4/lda_scaling.png", dpi=200, bbox_inches='tight')
plt.savefig("results/exp4/lda_scaling.pdf", bbox_inches='tight')
print("Saved to results/exp4/lda_scaling.png and .pdf")
