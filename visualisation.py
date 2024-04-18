import numpy as np
import matplotlib.pyplot as plt

from visualization_extra import _input_validation, max_rulelength_visual
from visualization_extra import define_relative_position, plot_arrow

def plot_rules(rules, preds, baselines, weights, max_rulelen=None,
               other_preds=None, preds_distr=None, b_box_pred=None, 
               density_smoothing='normal',
               round_digits=3, cmap="RdYlGn_r",
               base_fontsize=13):
    """
    A visualisation tool for BELLATREX, a local random forest explainability
    toolbox.

    @param rules: A list of lists, where each inner list contains strings 
        representing the decision rules that are taken.
    @param preds: A list of lists, of the same shape as `rules`, where each
        inner list contains numbers representing the prediction at each point
        of the rule path.
    @param baselines: A list indicating the baseline prediction for each rule.
    @param weights: A list indicating the weight of each rule.
    @param max_rulelen: Maximum number of rules shown for each decision path.
    @param other_preds: Optional list of lists containing `preds` for other
        trees in the random forest.
    @param preds_distr: Optional list of predictions made by the random forest
        on a set of training/testing patients. Determines the x-limits for the density plot
    @param cmap: The colormap used for visualization. Use "RdYlGn_r" if lower
        predictions is better. Omit the "_r" if the reverse holds.
    @param b_box_pred: Optional float (or list of) with prediction of the 
        original black-box model, for the sake of comparison
    @return: List of axes handles, for further finetuning of the graph.
    """
    
    # Validate inputs and determine maximum rule length
    _input_validation(rules, preds, baselines, weights)
    
    max_rulelen_visual = max_rulelength_visual(rules, max_rulelen=max_rulelen)
    
    nrules = len(rules)
    
    for i in range(nrules):
        assert len(rules[i]) == len(preds[i])
        if len(rules[i]) > max_rulelen:
            # +1 because we need to replace the last one
            omitted = len(rules[i]) - max_rulelen + 1
            rules[i][max_rulelen-1] = f"+{omitted} other rule splits"
            preds[i][max_rulelen-1] = preds[i][-1]
            rules[i] = rules[i][:max_rulelen]
            preds[i] = preds[i][:max_rulelen]
    
    if other_preds is not None: # estimates distribution of all rule esitmates
        for i in range(len(other_preds)):
            if len(other_preds[i]) > max_rulelen:
                other_preds[i][max_rulelen-1] = other_preds[i][-1] #check leaf node
                other_preds[i] = other_preds[i][:max_rulelen]

    if preds_distr is not None:
        from scipy import stats
        density = stats.gaussian_kde(preds_distr)

        extent = preds_distr.max() - preds_distr.min()
        x = np.linspace(preds_distr.min()-0.00*extent, 
                        preds_distr.max()+0.00*extent, 100)

    # Make a colorpicker
    cmap = plt.get_cmap(cmap)
    maxdev = max([np.max(np.abs(baselines[i] - np.array(preds[i]))) 
                  for i in range(nrules)])
    norm = plt.matplotlib.colors.Normalize(vmin=-maxdev, vmax=+maxdev)
    get_color = lambda value, baseline: cmap(norm(value - baseline))
    
    # Initialize the plot (rules and arrows only)
    plot_height_rulebased = 0.9*max(max_rulelen, 4)
    if preds_distr is None: #no extra axis objects for density plot
        fig, aaxs = plt.subplots(figsize=(5*nrules+2, plot_height_rulebased+2),
                                 nrows=2, ncols=nrules, sharey=True,
                                 gridspec_kw={"hspace":0, "height_ratios":[plot_height_rulebased, 1]})
        # axs = np.atleast_1d(aaxs)
        if len(aaxs.shape) == 1:
            aaxs = np.atleast_2d(aaxs).T
        axs     = aaxs[0,:]
    
    else: #create extra axis object for density plots (2d array)
        fig, aaxs = plt.subplots(figsize=(5*nrules+2, plot_height_rulebased+3), 
            nrows=3, ncols=nrules, sharex=True, sharey="row", 
            gridspec_kw={"hspace":0, "height_ratios":[plot_height_rulebased,1, 1]})
        if len(aaxs.shape) == 1:
            aaxs = np.atleast_2d(aaxs).T
        axs     = aaxs[0,:]
        distaxs = aaxs[1,:]

    
    margin = 0.01 * 2*maxdev # 1% margin left and right
    min_rel_x_axis = np.min(baselines)-maxdev-margin
    max_rel_x_axis = np.max(baselines)+maxdev+margin
    
    if preds_distr is not None: # then we have a linspace with underlying preds_distr
        max_rel_x_axis = max(max_rel_x_axis, x[-1])
        min_rel_x_axis = min(min_rel_x_axis, x[0])

    
    for i, ax in enumerate(axs):
        ax.invert_yaxis()
        # make the x axis include all partial prediction of the forest internal nodes in interval, plus some margin
        ax.set_xlim([min_rel_x_axis, max_rel_x_axis])
        
        
        
        ax.set_ylim([max_rulelen+0.75, -0.75])
        # ax.set_xlabel(f"Prediction\nThis rule: {(preds[i][-1])} with weight {weights[i]}")
        axs[0].set_ylabel("Rule depth", fontsize=base_fontsize)
        ax.set_yticks(range(max_rulelen+(max_rulelen==max_rulelen_visual)))
        ax.tick_params(axis='y', labelsize=base_fontsize)
        ax.grid(axis="x", zorder=-999, alpha=0.5)
        font_rule_title = base_fontsize + max(0, 4-len(axs)) # decreases with increasing number of final (selected, plotted) rules
        # ax.set_title(f"Selected rule {i+1}\nprediction = {(preds[i][-1]):.{round_digits}f}, weight = {weights[i]:.{round_digits-1}f}",
                     # fontsize=font_rule_title)# (weighted {weights[i]:.2f})")
        ax.set_title(f"Selected rule {i+1}\n (weight = {100*weights[i]:.0f}%)",
                      fontsize=font_rule_title)# (weighted {weights[i]:.2f})")    
    
    plt.subplots_adjust(wspace=0.12)
    # alt: max_rulelen --> fig.get_size_inches()[0]
    aspect = 20 * (max_rulelen / 5) # because aspect=20 is ideal when max_rulelen=5
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=aaxs, pad=0.04,
                 aspect=aspect, label="Change w.r.t. baseline")
    cbar.ax.tick_params(labelsize=base_fontsize+1)
    cbar.set_label("Change w.r.t. baseline", fontsize=base_fontsize+1)  # Adjusting the main label size (again)
    
    # Visualize the entire forest
    if other_preds:
        for bsl, ax in zip(baselines, axs):
            for pred in other_preds:
                ax.plot([bsl, *pred], np.arange(len(pred)+1), c=[0.9,0.9,0.9], 
                        alpha=1.0, zorder=-500, lw=0.6)


    # Highlight the rule of interest on each plot
    for bsl, rule, pred, ax in zip(baselines, rules, preds, axs):
        traj = [bsl, *pred]
        pad = 0.3
        ax.text(s=f"Baseline\n{bsl:.{round_digits}f}", fontsize=base_fontsize,
                x=bsl, y=-pad, ha="center", va="center", 
                bbox=dict(boxstyle=f"square,pad={pad}", fc="w", ec="k", alpha=0.5))
        isRight = (pred[-1] < bsl)
        ha = ["left","right"][isRight]
        # ha = "center"
        ax.text(s=f"Prediction\n{pred[-1]:.{round_digits}f}", fontsize=base_fontsize, 
                x=pred[-1] - (isRight-0.5)*pred[-1]/50,
                y=len(pred)+pad, ha=ha, va="center",
                bbox=dict(boxstyle=f"square,pad={pad}", fc="w", ec="k", alpha=0.5, zorder=5))
        for j in range(len(rule)):
            color = get_color(pred[j], bsl)
            # Draw the arrow
            ax.annotate(
                text="", xy=(traj[j+1], j+1), xytext=(traj[j], j),
                arrowprops=dict(
                    arrowstyle="-|>",
                    linewidth=2, 
                    shrinkB=0,
                    mutation_scale=20,
                    edgecolor=color,
                    facecolor=color,
                )
            )
            # Draw the text
            xtext = (4*traj[j]+6*traj[j+1])/10
            xmin, xmax = ax.get_xlim()
            closest = np.argmin([xtext-xmin, xtext-(xmin+xmax)/2, xmax-xtext])
            ha = ["left","center","right"][closest]
            ax.text(
                s=parse(rule[j]),
                x=xtext, y=j+1/2.1,
                ha=ha, va="center",
                fontsize=base_fontsize,
                bbox=dict(boxstyle="square,pad=0", fc="w", ec="w", lw=1, alpha=0.75),
            )

    # Draw the distribution (density) on each rule-plot
    if preds_distr is not None:
        # Training set distribution (as provided by preds_distr)
        for i, (bsl, pred, ax) in enumerate(zip(baselines, preds, distaxs)):
            ax.plot(x, density(x), "k")
            # ax.vlines(x=pred[-1], ymin=0, ymax=density(pred[-1]), colors="k", linewidth=5)
            col1 = "gray" #get_color(bsl     , bsl)
            col2 = get_color(pred[-1], bsl)
            ax.plot(bsl     , density(bsl     ), ".", c=col1, ms=15)
            ax.plot(pred[-1], density(pred[-1]), ".", c=col2, ms=15)
            ax.vlines(x=bsl     , ymin=0, ymax=density(bsl     ), colors=col1, linestyles=":")
            ax.vlines(x=pred[-1], ymin=0, ymax=density(pred[-1]), colors=col2, linestyles=":")
            ax.set_ylim([0, 1.1*ax.get_ylim()[1]])
            ax.set_yticks([])
            ax.set_xlabel("Prediction", fontsize=base_fontsize)
            ax.grid(axis="x", zorder=-999, alpha=0.5)
        distaxs[0].set_ylabel("Density", fontsize=base_fontsize)
        

            
        # Connect density better to the rest of the plot
        for bsl, pred, ax, distax in zip(baselines, preds, axs, distaxs):
            # Draw dotted vline from baseline to density
            ax.vlines(x=bsl, ymin=0, ymax=ax.get_ylim()[0],
                      colors="gray", linestyles=":")
            distax.vlines(x=bsl, ymin=density(bsl), ymax=distax.get_ylim()[1],
                      colors="gray", linestyles=":")
            # Draw dotted line from final prediction to density
            ax.vlines(x=pred[-1], ymin=max_rulelen, ymax=ax.get_ylim()[0],
                      colors=get_color(pred[-1], bsl), linestyles=":")
            distax.vlines(x=pred[-1], ymin=density(pred[-1]), ymax=distax.get_ylim()[1],
                    colors=get_color(pred[-1], bsl), linestyles=":")
            
            
    ## Add weighted average contribution to the bottom of the plot
    # on the arroaxs Axes object
        
    n_cols = aaxs.shape[1]
    #define relative position of aroow subplot depending on n_cols
    pos_list = define_relative_position(n_cols)

    rule_preds = [preds[i][-1] for i in range(len(rules))]

    for j, pos in zip(range(n_cols), pos_list):
        
        # add the averaging arrows the last row of axes
        aaxs[-1, j] = plot_arrow(aaxs[-1, j], pos,
                                 weight=weights[j],
                                 pred_out=rule_preds[j],
                                 fontsize=base_fontsize)
                
    # Add final prediction to the plot
    final_pred = np.sum([weights[i] * preds[i][-1] for i in range(len(rules))])
    final_pred_str = f"Bellatrex weighted prediction: {final_pred:.{round_digits}f}"
    final_pred_str += " (= " + " + ".join([
        rf"{preds[i][-1]:.{round_digits}f}$\times${weights[i]:.{round_digits-1}f}" 
        for i in range(len(rules))
    ]) + ")        " # extra space to the right to move text slighlty to the left
    
    if b_box_pred is not None:
        bbox_pred_str = "\n(compared to black-box model prediction: "
        bbox_pred_str += ", ".join([f"{pred:.{round_digits}f}" 
                                     for pred in np.atleast_1d(b_box_pred)])
        bbox_pred_str += ")"
            
    # fig.suptitle(bbox_pred_str, va="top", y=0.99, fontsize=font_rule_title)
    
    plt.figtext(0.5, 0.05, final_pred_str+bbox_pred_str, fontsize=font_rule_title, ha="center")
    
    return fig, aaxs


def parse(rulesplit):
    """Parses a rulesplit outputted by Bellatrex into a form suitable for visualisation."""
    # Remove information related to the current value
    if "(" in rulesplit:
        rulesplit = rulesplit[:rulesplit.rfind("(")].strip()
    # Replace special characters by LaTeX symbols
    rulesplit = rulesplit.replace("≤" , "$\leq$")
    rulesplit = rulesplit.replace("<=", "$\leq$")
    rulesplit = rulesplit.replace("≥" , "$\geq$")
    rulesplit = rulesplit.replace(">=", "$\geq$")

    return rulesplit


def read_rules(file, file_extra=None):
    rules = []
    preds = []
    baselines = []
    weights = []
    with open(file, "r") as f:
        btrex_rules = f.readlines()
    for line in btrex_rules:
        if "RULE WEIGHT" in line:
            weights.append( float(line.split(":")[1].strip("\n").strip(" #")) )
        if "Baseline prediction" in line:
            baselines.append( float(line.split(":")[1].strip(" \n")) )
            rule = []
            pred = []
        if "node" in line:
            fullrule = line.split(":")[1].strip().strip("\n").split("-->")
            index_thresh = max([fullrule[0].find(char) for char in ["=","<",">"]])
            fullrule[0] = fullrule[0][0:index_thresh+8]
            rule.append( fullrule[0] )
            pred.append( float(fullrule[1]) )
        if "leaf" in line:
            rules.append(rule)
            preds.append(pred)

    if file_extra:
        other_preds = []
        with open(file_extra, "r") as f:
            btrex_extra = f.readlines()
        for line in btrex_extra:
            if "Baseline prediction" in line:
                pred = []
            if "node" in line:
                pred.append(float(line.split("-->")[1]))
            if "leaf" in line:
                other_preds.append(pred)
    else:
        other_preds = None

    return rules, preds, baselines, weights, other_preds


    
if __name__ == "__main__":
    
    rules, preds, baselines, weights, other_preds = read_rules(
        file       = "example-explanations/Rules_boston_housing_f0_id1.txt",
        file_extra = "example-explanations/Rules_boston_housing_f0_id1-extra.txt"
    )
    preds_distr = np.load("example-data/blood_y_train_preds.npy")
    fig, aaxs = plot_rules(rules, preds, baselines, weights, 
                max_rulelen=5, other_preds=other_preds, 
                preds_distr=preds_distr,
                b_box_pred=0.6 # just a random number
    )
    # aaxs[0,0].set_xlim([0,1])
    plt.savefig("visualisation.pdf", bbox_inches="tight")
    plt.show()
