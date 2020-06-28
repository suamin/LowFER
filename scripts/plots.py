# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt  

plt.style.use("default")


def plot_effect_of_k():
    
    k = [1, 5, 10, 30, 50, 100, 150, 200]
    hits_at_1 = [0.526, 0.640, 0.665, 0.697, 0.712, 0.725, 0.727, 0.732]
    mrr = [0.623, 0.718, 0.738, 0.763, 0.775, 0.784, 0.786, 0.789]

    plt.figure(figsize=(6.5, 4.5))
    ax = plt.subplot(111)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left() 
    ax.xaxis.set_ticks(np.arange(0, 210, 50))
    ax.yaxis.set_ticks(np.arange(0.40, 0.85, 0.1))

    plt.yticks(fontsize=12)    
    plt.xticks(fontsize=12) 

    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,        # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
    ) 

    ax.plot(k,    
            hits_at_1, linewidth=2, 
            lw=2.5, color=(44/ 255., 160/ 255., 44/ 255.), marker='^', label='Hits@1', markersize=8)
    ax.plot(k,    
            mrr, linewidth=2, 
            lw=2.5, color=(31/ 255., 119/ 255., 180/ 255.), marker='o', label='MRR', markersize=8)

    ax.set_xlim([-3, 210])
    ax.set_ylim([0.40, 0.85])
    ax.set_xlabel("Factorization Rank (k)", size=13, labelpad=15)
    ax.set_ylabel("MRR/Hits@1", size=13, labelpad=15)
    ax.tick_params(labelsize=10)
    ax.grid(b=True, color="black", linestyle='--', alpha=0.3, axis='y', lw=0.5)
    ax.grid(False)

    plt.legend(loc="lower right", fontsize=15)

    plt.tight_layout()

    fig.savefig("./k_vs_mrr_h1_fb15k_e200_r30_2_nogrid.png", format="png", dpi=400)




def plot_effect_of_embed():
    
    entity_embed_dims = [30, 50, 100, 150, 200, 250, 300, 350, 400]
    hits_at_1 = [0.2389, 0.3445, 0.4932, 0.6167, 0.7119, 0.7658, 0.7815, 0.7482, 0.7186]
    params_in_million = [0.619, 1.02, 2.22, 3.67, 5.37, 7.31, 9.51, 11.96, 14.66]


    plt.figure(figsize=(6.5, 4.5))
    ax = plt.subplot(111)

    plt.yticks(fontsize=12)    
    plt.xticks(fontsize=12) 

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))

    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,
        right=False,
        bottom=False, 
        ) 

    ax1.get_xaxis().tick_bottom()    
    ax1.get_yaxis().tick_left() 

    ax1.plot(entity_embed_dims,    
            hits_at_1, linewidth=2, 
            lw=2.5, color=(31/ 255., 119/ 255., 180/ 255.), marker='o', label='Hits@1', markersize=8)  

    ax1.set_xlim([0, 410])
    ax1.set_xlabel("Entity Dimension", size=13, labelpad=15)
    ax1.set_ylabel("Hits@1", size=13, labelpad=15)
    ax1.tick_params(labelsize=10)
    
    ax1.xaxis.set_ticks(np.arange(0, 410, 50))

    ax1.legend(loc="upper left", fontsize=15)

    ax2 = ax1.twinx()
    ax2.plot(entity_embed_dims, params_in_million, linewidth=2, 
            lw=2.5, color=(214/ 255., 39/ 255., 40/ 255.), marker='^', label='Parameters', markersize=8)

    ax2.set_ylabel("Parameters (M)", size=13, labelpad=15)
    ax2.tick_params(labelsize=10)
    ax2.legend(loc="lower right", fontsize=15)

    plt.tight_layout()

    fig.savefig("./de_vs_h1_params_fb15k_r30_k50_2.png", format="png", dpi=400)



plot_effect_of_k()
plot_effect_of_embed()
