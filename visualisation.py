import numpy as np
import matplotlib.pyplot as plt

def plot_rules(rules, preds, baselines, weights, max_rulelen=None,
               other_preds=None, preds_distr=None, b_box_pred=None, 
               round_digits=3, cmap="RdYlGn_r"):
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
        on a set of training/testing patients.
    @param cmap: The colormap used for visualization. Use "RdYlGn_r" if lower
        predictions is better. Omit the "_r" if the reverse holds.
    @param b_box_pred: Optional float (or list of) with prediction of the 
        original black-box model, for the sake of comparison
    @return: List of axes handles, for further finetuning of the graph.
    """
    # Input validation and processing
    assert len(rules) == len(preds) == len(baselines) == len(weights)
    nrules = len(rules)
    max_rulelen_model = max([len(rule) for rule in rules])
    if max_rulelen is None:
        max_rulelen = max_rulelen_model
    else:
        max_rulelen = min(max_rulelen_model, max_rulelen)
    for i in range(nrules):
        assert len(rules[i]) == len(preds[i])
        if len(rules[i]) > max_rulelen:
            # +1 because we need to replace the last one
            omitted = len(rules[i]) - max_rulelen + 1
            rules[i][max_rulelen-1] = f"+{omitted} other rule splits"
            preds[i][max_rulelen-1] = preds[i][-1]
            rules[i] = rules[i][:max_rulelen]
            preds[i] = preds[i][:max_rulelen]
    if other_preds:
        for i in range(len(other_preds)):
            if len(other_preds[i]) > max_rulelen:
                other_preds[i][max_rulelen-1] = other_preds[i][-1]
                other_preds[i] = other_preds[i][:max_rulelen]
    if preds_distr is not None:
        from scipy import stats
        density = stats.gaussian_kde(preds_distr)
        extent = preds_distr.max() - preds_distr.min()
        x = np.linspace(preds_distr.min()-0.05*extent, 
                        preds_distr.max()+0.05*extent, 100)

    # Make a colorpicker
    cmap = plt.get_cmap(cmap)
    maxdev = max([np.max(np.abs(baselines[i] - np.array(preds[i]))) 
                  for i in range(nrules)])
    norm = plt.matplotlib.colors.Normalize(vmin=-maxdev, vmax=+maxdev)
    get_color = lambda value, baseline: cmap(norm(value - baseline))
    
    # Initialize the plot
    plot_height_rulebased = max(max_rulelen, 4)
    if preds_distr is None:
        fig, aaxs = plt.subplots(figsize=(5*nrules+2, plot_height_rulebased), 
                                ncols=nrules, sharey=True)
        axs = np.atleast_1d(aaxs)
    else:
        fig, aaxs = plt.subplots(figsize=(5*nrules+2, plot_height_rulebased+1), 
            nrows=2, ncols=nrules, sharex=True, sharey="row", 
            gridspec_kw={"hspace":0, "height_ratios":[plot_height_rulebased,1]})
        if len(aaxs.shape) == 1:
            aaxs = np.atleast_2d(aaxs).T
        axs     = aaxs[0,:]
        distaxs = aaxs[1,:]
    for i,ax in enumerate(axs):
        margin = 0.01 * 2*maxdev # 1% margin left and right
        ax.invert_yaxis()
        ax.set_xlim([np.min(baselines)-maxdev-margin, np.max(baselines)+maxdev+margin])
        ax.set_ylim([max_rulelen+0.75, -0.75])
        ax.set_xlabel("Prediction")
        axs[0].set_ylabel("Rule depth")
        ax.set_yticks(range(max_rulelen+(max_rulelen_model==max_rulelen)))
        ax.grid(axis="x", zorder=-999, alpha=0.5)
        ax.set_title(f"Rule {i+1} (weight {weights[i]:.2f})")
    plt.subplots_adjust(wspace=0.05)
    # alt: max_rulelen --> fig.get_size_inches()[0]
    aspect = 20 * (max_rulelen / 5) # because aspect=20 is ideal when max_rulelen=5
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=aaxs, pad=0.02,
                 aspect=aspect, label="Change w.r.t. baseline")
    # colorbar_export = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Visualize the entire forest
    if other_preds:
        for bsl, ax in zip(baselines, axs):
            for pred in other_preds:
                # ax.plot([bsl, *pred], -np.arange(len(pred)+1), c="gray", 
                # alpha=0.05, zorder=-500)
                # ax.plot([bsl, *pred], -np.arange(len(pred)+1), c=[0.8,0.8,0.8], 
                # alpha=0.5, zorder=-500)
                ax.plot([bsl, *pred], np.arange(len(pred)+1), c=[0.8,0.8,0.8], 
                        alpha=1.0, zorder=-500, lw=0.5)

    # # Visualize the chosen rules (small multiples background)
    # for bsl, ax in zip(baselines, axs):
    #     for pred in preds:
    #         ax.plot([bsl, *pred], np.arange(len(pred)+1), c="k", zorder=-500)

    # Highlight the rule of interest on each plot
    for bsl, rule, pred, ax in zip(baselines, rules, preds, axs):
        traj = [bsl, *pred]
        fontsize = 10
        pad = 0.3
        ax.text(s=f"Baseline\n{bsl}", fontsize=fontsize,
                x=bsl, y=-pad, ha="center", va="center", 
                bbox=dict(boxstyle=f"square,pad={pad}", fc="w", ec="k", alpha=0.5))
        ha = ["left","right"][pred[-1] < bsl]
        ha = "center"
        ax.text(s=f"Prediction\n{pred[-1]}", fontsize=fontsize, 
                x=pred[-1], y=len(pred)+pad, ha=ha, va="center",
                bbox=dict(boxstyle=f"square,pad={pad}", fc="w", ec="k", alpha=0.5))
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
            xtext = (2*traj[j]+traj[j+1])/3
            xmin, xmax = ax.get_xlim()
            closest = np.argmin([xtext-xmin, xtext-(xmin+xmax)/2, xmax-xtext])
            ha = ["left","center","right"][closest]
            ax.text(
                s=parse(rule[j]),
                x=xtext, y=j+1/3,
                ha=ha, va="center",
                fontsize=10,
                bbox=dict(boxstyle="square,pad=0", fc="w", ec="w", lw=1, alpha=0.75),
            )

    # Draw the distribution on each plot
    if preds_distr is not None:
        for bsl, pred, ax in zip(baselines, preds, distaxs):
            ax.plot(x       , density(x       ), "k")
            ax.plot(pred[-1], density(pred[-1]), ".", 
                    c=get_color(pred[-1], bsl), ms=15)
            ax.vlines(x=bsl     , ymin=0, ymax=density(bsl     ), colors="k")
            ax.vlines(x=pred[-1], ymin=0, ymax=density(pred[-1]), 
                      colors=get_color(pred[-1], bsl))
            ax.set_ylim([0, ax.get_ylim()[1]])
            ax.set_yticks([])
            ax.set_xlabel("Prediction")
            ax.grid(axis="x", zorder=-999, alpha=0.5)
        distaxs[0].set_ylabel("Density")
        # Connect density better to the rest of the plot
        for bsl, pred, ax, distax in zip(baselines, preds, axs, distaxs):
            # Draw dotted vline from baseline to density
            ax.vlines(x=bsl, ymin=0, ymax=ax.get_ylim()[0],
                      colors="k", linestyles=":")
            distax.vlines(x=bsl, ymin=density(bsl), ymax=distax.get_ylim()[1],
                      colors="k", linestyles=":")
            # Draw dotted line from final prediction to density
            ax.vlines(x=pred[-1], ymin=max_rulelen, ymax=ax.get_ylim()[0],
                      colors=get_color(pred[-1], bsl), linestyles=":")
            distax.vlines(x=pred[-1], ymin=density(pred[-1]), ymax=distax.get_ylim()[1],
                    colors=get_color(pred[-1], bsl), linestyles=":")

        
    # Add final prediction to the plot
    # PREVIOUS VERSION: 
    # string = "Bellatrex prediction:  {pred1}\nBlack-box prediction: {pred2}"
    # fig.text(0.3, 0.02, string, ha='left', va='center', fontsize=14)
    final_pred = np.sum([weights[i] * preds[i][-1] for i in range(len(rules))])
    final_pred_str = f"Final BellaTrex prediction = {final_pred:.{round_digits}f}"
    final_pred_str += " = " + " + ".join([
        rf"{weights[i]:.{round_digits-1}f}$\times${preds[i][-1]:.{round_digits}f}" 
        for i in range(len(rules))
    ])
    if b_box_pred is not None:
        final_pred_str += "\n(compared to black-box model which predicts "
        final_pred_str += ", ".join([f"{pred:.{round_digits}f}" 
                                     for pred in np.atleast_1d(b_box_pred)])
        final_pred_str += ")"
    figheight = plot_height_rulebased + (preds_distr is not None)
    # y = np.sqrt(figheight) / 100 * 2.2
    y = figheight/100 - 0.04
    fig.supxlabel(final_pred_str, va="top", y=y)
    return aaxs

def parse(rule):
    """Parses a rule outputted by bellatrex into a form suitable for visualisation."""
    # Remove information related to the current value
    if "(" in rule:
        rule = rule[:rule.rfind("(")].strip()
    # Replace special characters by LaTeX symbols
    rule = rule.replace("≤" , "$\leq$")
    rule = rule.replace("<=", "$\leq$")
    rule = rule.replace("≥" , "$\geq$")
    rule = rule.replace(">=", "$\geq$")
    # If split on binary variable, change format
    for i, comparator in enumerate(["<", "$\leq$", "$\geq$", ">"]):
        if comparator in rule:
            value = rule.split(comparator)[1]
            if (float(value) == 0.5) and ("is" in rule): # TODO improve
                rule = rule.replace("is", [r"$\neq$", "$=$"][i>1])
                rule = rule[:rule.find(comparator)].strip()
    # rule = rule.encode().decode('unicode_escape')
    return rule

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
    preds_distr = np.load("example-data/bin_tutorial_y_train_preds.npy")
    aaxs = plot_rules(rules, preds, baselines, weights, 
                max_rulelen=5, other_preds=other_preds, preds_distr=preds_distr,
                b_box_pred=0.6 # just a random number
    )
    # aaxs[0,0].set_xlim([0,1])
    plt.savefig("visualisation.pdf", bbox_inches="tight")
