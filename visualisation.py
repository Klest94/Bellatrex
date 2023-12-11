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
    @param bbox_pred: Optional flaot (or list of) with prediction of the 
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
            omitted = len(rules[i]) - max_rulelen + 1 # +1 because we need to replace the last one
            rules[i][max_rulelen-1] = f"+{omitted} other features"
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
    maxdev = max([np.max(np.abs(baselines[i] - np.array(preds[i]))) for i in range(nrules)])
    norm = plt.matplotlib.colors.Normalize(vmin=-maxdev, vmax=+maxdev)
    get_color = lambda value, baseline: cmap(norm(value - baseline))
    
    plot_height_rulebased = max(max_rulelen, 4)
    # Initialize the plot
    if preds_distr is None:
        fig, axs = plt.subplots(figsize=(5*nrules+2, plot_height_rulebased), ncols=nrules, sharey=True)
        axs = np.atleast_1d(axs)
    else:
        fig, aaxs = plt.subplots(figsize=(5*nrules+2, plot_height_rulebased+1), nrows=2, ncols=nrules, sharex=True, sharey="row", 
                                 gridspec_kw={"hspace":0, "height_ratios":[plot_height_rulebased,1]})
        if len(aaxs.shape) == 1:
            aaxs = np.atleast_2d(aaxs).T
        axs     = aaxs[0,:]
        distaxs = aaxs[1,:]
    for i,ax in enumerate(axs):
        margin = 0.01 * 2*maxdev # 1% margin left and right
        ax.set_xlim([np.min(baselines)-maxdev-margin, np.max(baselines)+maxdev+margin])
        ax.set_ylim([-max_rulelen-0.5, 1])
        ax.set_xlabel("Prediction")
        axs[0].set_ylabel("Rule depth")
        ax.set_yticks([])
        ax.grid(axis="x", zorder=-999, alpha=0.5)
        ax.set_title(f"Rule {i+1} (weight {weights[i]:.2f})")
    plt.subplots_adjust(wspace=0.05)
    # alt: max_rulelen --> fig.get_size_inches()[0]
    aspect = 20 * (max_rulelen / 5) # because aspect=20 is ideal when max_rulelen=5
    caxs = axs if preds_distr is None else aaxs
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=caxs, pad=0.02,
                 aspect=aspect, label="Change w.r.t. baseline")
    
    # colorbar_export = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Visualize the entire forest
    if other_preds:
        for bsl, ax in zip(baselines, axs):
            for pred in other_preds:
                # ax.plot([bsl, *pred], -np.arange(len(pred)+1), c="gray", alpha=0.05, zorder=-500)
                # ax.plot([bsl, *pred], -np.arange(len(pred)+1), c=[0.8,0.8,0.8], alpha=0.5, zorder=-500)
                ax.plot([bsl, *pred], -np.arange(len(pred)+1), c=[0.8,0.8,0.8], alpha=1.0, zorder=-500, lw=0.5)

    # # Visualize the chosen rules (small multiples background)
    # for bsl, ax in zip(baselines, axs):
    #     for pred in preds:
    #         ax.plot([bsl, *pred], -np.arange(len(pred)+1), c="k", zorder=-500)

    # Highlight the rule of interest on each plot
    for bsl, rule, pred, ax in zip(baselines, rules, preds, axs):
        traj = [bsl, *pred]
        fontsize = 10
        ax.text(s=f"Baseline\n{bsl}", fontsize=fontsize,
                x=bsl, y=0.5, ha="center", va="center", 
                bbox=dict(boxstyle="square,pad=0.3", fc="w", ec="k"))
        for j in range(len(rule)):
            color = get_color(pred[j], bsl)
            # Draw the arrow
            ax.annotate(
                text="", xy=(traj[j+1], -j-1), xytext=(traj[j], -j),
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
            # isLeft = (pred[j] - preds.min() < preds.max() - pred[j])
            # ha = ["right","left"][isLeft]
            # pad = [-1, 1][isLeft]*0.05*(preds.max() - preds.min())
            ax.text(
                s=parse(rule[j]),
                x=(2*traj[j]+traj[j+1])/3, y=-j-1/3,
                ha="center", va="center",
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
            ax.vlines(x=pred[-1], ymin=0, ymax=density(pred[-1]), colors=get_color(pred[-1], bsl))
            ax.set_ylim([0, ax.get_ylim()[1]])
            ax.set_yticks([])
            ax.set_xlabel("Prediction")
            ax.grid(axis="x", zorder=-999, alpha=0.5)
        distaxs[0].set_ylabel("Density")
        
    # Add final prediction to the plot as an annotation
    pred_terms = [f"{round(weights[i], round_digits-1):.{round_digits-1}f}*{round(preds[i][-1], round_digits):.{round_digits}f}" 
                  for i in range(len(rules))]
    final_pred = np.sum(weights[i] * preds[i][-1] for i in range(len(rules)))
    final_pred_str = f"{round(final_pred, round_digits):.{round_digits}f} = " + " + ".join(pred_terms)

    # Add final predictions to the bottom of the plot
    # First: determine format of b_box_pred variable for display
    if b_box_pred is not None:
        if isinstance(b_box_pred, list):
            b_box_pred_str = ', '.join([f'{round(pred, round_digits):.{round_digits}f}' for pred in b_box_pred])
        else:
            b_box_pred_str = f'{round(b_box_pred, round_digits):.{round_digits}f}'
    
    # # Add the text to the figure, display the first line
    fig.text(0.3, 0.02, f'Bellatrex prediction:  {final_pred_str}', ha='left', va='center', fontsize=14)
    # Display the second line (left-aligned to the start of the first line)
    # if b_box_pred is not None:
    #     fig.text(0.3, +0.075, f'Black-box prediction: {b_box_pred_str}', ha='left', va='center', fontsize=14)

    return axs

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
    import pandas as pd

    rules, preds, baselines, weights, other_preds = read_rules(
        file       = "example-explanations/Rules_boston_housing_f0_id0.txt",
        file_extra = "example-explanations/Rules_boston_housing_f0_id0-extra.txt"
    )
    preds_distr = np.load("example-data/bin_tutorial_y_train_preds.npy")
    axs = plot_rules(rules, preds, baselines, weights, 
                max_rulelen=None, other_preds=other_preds, preds_distr=preds_distr,
    )
    axs[0].set_xlim([0,1])
    plt.savefig("visualisation.pdf")
