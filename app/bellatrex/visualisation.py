import warnings
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .visualization_extra import _input_validation, max_rulelength_visual
from .visualization_extra import define_relative_position, plot_arrow
from .utilities import frmt_pretty_print


def plot_rules(rules, preds, baselines, weights, max_rulelen=None,
               other_preds=None, preds_distr=None, b_box_pred=None,
               conf_level=None,
               tot_digits=4, cmap="RdYlGn_r",
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
    @param conf_level: Optional float, it greys out the interval corresponding to the confidence level
    @param tot_digits: Default: 4. number of significative digits to show
    @param cmap: The colormap used for visualization. Use "RdYlGn_r" if lower
        predictions is better. Omit the "_r" if the reverse holds.
    @param b_box_pred: Optional float (or list of) with prediction of the
        original black-box model, for the sake of comparison
    @return: List of figure and axes objects, for further customizing the graph.
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

    if other_preds is not None: # sets up distribution of all rule esitmates
        for i, other_pred in enumerate(other_preds):
            if len(other_pred) > max_rulelen:
                other_pred[max_rulelen-1] = other_pred[-1] #check leaf node
                other_pred = other_pred[:max_rulelen]

    if preds_distr is not None:
        density = stats.gaussian_kde(preds_distr)

        extent = preds_distr.max() - preds_distr.min()
        x = np.linspace(preds_distr.min()-0.00*extent,
                        preds_distr.max()+0.00*extent, 100)

    # Make a colorpicker

    if cmap is None:
        cmap = 'RdYlGn_r'

    if cmap != "shap":
        cmap = plt.get_cmap(cmap)
    else:
        shap_blue = "#008bfb"  # SHAP blue
        shap_red = "#ff0051"
        # Create the colormap
        cmap = LinearSegmentedColormap.from_list("shap_cmap", [shap_blue, shap_red], N=256)


    deviations = [np.array(preds[i]) - baselines[i] for i in range(nrules)]

    dev_min = min(np.min(dev) for dev in [0, *deviations]) #include 0 if all positives
    dev_max = max(np.max(dev) for dev in [0, *deviations]) #include 0 if all negatives

    # choose best colorbar to taste:
    # norm = plt.matplotlib.colors.Normalize(vmin=-dev_max, vmax=dev_max)
    norm = plt.matplotlib.colors.Normalize(vmin=-0.5*(dev_max-dev_min), vmax=0.5*(dev_max-dev_min))

    get_color = lambda value, baseline: cmap(norm(value - baseline))

    # Initialize the plot (rules and arrows only)
    plot_height_rulebased = 0.9*max(max_rulelen, 4)

    if preds_distr is None: #no extra axis objects for density plot
        fig, aaxs = plt.subplots(figsize=(5*nrules+2, plot_height_rulebased+2),
                                 nrows=2, ncols=nrules, sharey=True,
                                 gridspec_kw={"hspace": 0,
                                              "height_ratios":[plot_height_rulebased, 1]})
        # rule_axs = np.atleast_1d(aaxs)
        if len(aaxs.shape) == 1:
            aaxs = np.atleast_2d(aaxs).T
        rule_axs = aaxs[0,:]
        dens_axs = [None] * len(rule_axs)  # useful placeholder for later iterations (?)

    else: #create extra axis object for density plots (2d array)
        fig, aaxs = plt.subplots(figsize=(5*nrules+2, plot_height_rulebased+3),
                                 nrows=3, ncols=nrules, sharey="row",
                                 gridspec_kw={"hspace":0,
                                              "height_ratios":[plot_height_rulebased, 1, 1]})
        if len(aaxs.shape) == 1:
            aaxs = np.atleast_2d(aaxs).T
        rule_axs = aaxs[0,:]
        dens_axs = aaxs[1,:]

    # Prepare to set minimum and maximum deviation as xaxis limits
    margin = 0.01 * (dev_max-dev_min) # 1% margin left and right
    min_rel_x_axis = np.min(baselines)+dev_min-margin
    max_rel_x_axis = np.max(baselines)+dev_max+margin

    if preds_distr is not None: # extend axis to fit underlying preds_distr min and max values
        max_rel_x_axis = max(max_rel_x_axis, x[-1])
        min_rel_x_axis = min(min_rel_x_axis, x[0])

    # set the y_label only on the leftmost plot (saves horizontal space)
    rule_axs[0].set_ylabel("Rule depth", fontsize=base_fontsize)

    for i, ax in enumerate(rule_axs):

        ax.invert_yaxis()
        # make the x axis include all partial prediction of the forest internal nodes
        # plus some margin, given by the few lines above. We choose to use
        # the same x-axis limits for all plots, for a better interpretability
        ax.set_xlim([min_rel_x_axis, max_rel_x_axis])
        ax.set_ylim([max_rulelen+0.75, -0.75])
        ax.set_yticks(range(max_rulelen_visual+1))
        ax.tick_params(axis='y', labelsize=base_fontsize)
        ax.grid(axis="x", zorder=-99, alpha=0.5)
        ax.set_title(f"Selected rule {i+1}\n (weight = {100*weights[i]:.0f}%)",
                      fontsize=base_fontsize)# (weighted {weights[i]:.2f})")

    plt.subplots_adjust(wspace=0.12)
    # alt: max_rulelen --> fig.get_size_inches()[0]
    aspect = 20 * (max_rulelen / 5) # because aspect=20 is ideal when max_rulelen=5
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=aaxs, pad=0.04,
                 aspect=aspect, label="Change w.r.t. baseline")
    cbar.ax.tick_params(labelsize=base_fontsize+1)
    cbar.set_label("Change w.r.t. baseline", fontsize=base_fontsize+1)  # Adjusting the main label size (again)

    # Visualize the entire forest
    if other_preds:
        for bsl, ax in zip(baselines, rule_axs):
            for pred in other_preds:
                ax.plot([bsl, *pred], np.arange(len(pred)+1), c=[0.9,0.9,0.9],
                        alpha=1.0, zorder=-500, lw=0.6) # zorder compared to zorder in rule_axs


    # Highlight the rule of interest on each plot
    for bsl, rule, pred, ax in zip(baselines, rules, preds, rule_axs):
        traj = [bsl, *pred]
        pad = 0.3
        xmin, xmax = ax.get_xlim()

        xtext_base_block = max(bsl, 0.85*xmin+0.15*xmax)
        xtext_base_block = min(xtext_base_block , 0.15*xmin+0.85*xmax)

        ax.text(s=f"Baseline\n{frmt_pretty_print(bsl, tot_digits)}",
                fontsize=base_fontsize,
                x=xtext_base_block,
                y=-pad, ha="center", va="center",
                bbox=dict(boxstyle=f"square,pad={pad}",
                          fc="w", ec="k", alpha=0.5))

        isRight = (pred[-1] < bsl)
        ha = ["left","right"][isRight]
        xtext_pred_block = max(pred[-1], 0.7*xmin+0.3*xmax)
        xtext_pred_block = min(xtext_pred_block , 0.3*xmin+0.7*xmax)

        ax.text(s=f"Prediction\n{frmt_pretty_print(pred[-1], tot_digits)}",
                fontsize=base_fontsize,
                x=xtext_pred_block,
                y=len(pred)+pad, ha=ha, va="center",
                bbox=dict(boxstyle=f"square,pad={pad}", fc="w",
                          ec="k", alpha=0.5, zorder=5))

        for j, _ in enumerate(rule):
            color = get_color(pred[j], bsl)
            # Draw the arrow
            ax.annotate(text="", xy=(traj[j+1], j+1),
                        xytext=(traj[j], j),
                        arrowprops=dict(arrowstyle="-|>",
                                        linewidth=2, shrinkB=0, mutation_scale=20,
                                        edgecolor=color,
                                        facecolor=color,
                                        )
                        )
            # Draw text, aligned betweeb the arrow trajectory and the center of the plot
            # xmin, xmax = ax.get_xlim()
            xtext = (2*traj[j]+2*traj[j+1]+6*(xmax+xmin)/2)/10 #traj[0] originally

            closest = np.argmin([xtext-xmin, np.abs(xtext-(xmin+xmax)/2), xmax-xtext])
            ha = ["left", "center", "right"][closest]
            ax.text(s=parse(rule[j]),
                    x=xtext,
                    y=j+1/2.1,
                    ha=ha, va="center", fontsize=base_fontsize,
                    bbox=dict(boxstyle="square,pad=0", fc="w",
                              ec="w", lw=1, alpha=0.75),
                    )

    # Draw the distribution (density) on each rule-plot
    if preds_distr is not None:
        # Training set distribution (as provided by preds_distr)
        for i, (bsl, pred, ax) in enumerate(zip(baselines, preds, dens_axs)):
            ax.plot(x, density(x), "k")
            col1 = "gray" #get_color(bsl     , bsl)
            col2 = get_color(pred[-1], bsl)
            ax.plot(bsl     , density(bsl     ), ".", c=col1, ms=15)
            ax.plot(pred[-1], density(pred[-1]), ".", c=col2, ms=15)
            ax.vlines(x=bsl     , ymin=0, ymax=density(bsl     ), colors=col1, linestyles=":")
            ax.vlines(x=pred[-1], ymin=0, ymax=density(pred[-1]), colors=col2, linestyles=":")
            ax.set_ylim([0, 1.1*ax.get_ylim()[1]])
            ax.set_yticks([])
            ax.grid(axis="x", zorder=-99, alpha=0.5)
            ax.set_xlabel("Prediction", fontsize=base_fontsize-1)
            ax.tick_params(axis='x', labelsize=base_fontsize)

        dens_axs[0].set_ylabel("Density", fontsize=base_fontsize)


        # Connect density plot axes to the rest of the plot
        for bsl, pred, ax, dax in zip(baselines, preds, rule_axs, dens_axs):
            # Draw dotted vline from baseline to density
            ax.vlines(x=bsl, ymin=0, ymax=ax.get_ylim()[0],
                      colors="gray", linestyles=":")
            dax.vlines(x=bsl, ymin=density(bsl), ymax=dax.get_ylim()[1],
                      colors="gray", linestyles=":")
            # Draw dotted line from final prediction to density
            ax.vlines(x=pred[-1], ymin=max_rulelen, ymax=ax.get_ylim()[0],
                      colors=get_color(pred[-1], bsl), linestyles=":")
            dax.vlines(x=pred[-1], ymin=density(pred[-1]), ymax=dax.get_ylim()[1],
                    colors=get_color(pred[-1], bsl), linestyles=":")

    ### Here confidence interval:

    if other_preds is None and conf_level is not None:
        warnings.warn("No confidsence level can be plotted if \'other_preds\' is None")

    # Do so by plotting distribution integrated in the preds_distr
    if other_preds is not None and conf_level is not None and dens_axs[0] is not None:

        final_preds = [pred[-1] for pred in other_preds] + [pred[-1] for pred in preds]

        # conf_level:  size of the confidence interval (0.9 = 90% of the tree predictions)
        if conf_level <= 0 or conf_level > 1:
            raise ValueError("conf_level must be in the (0, 1] range. Found:", conf_level)

        vmin = np.quantile(final_preds, (1-conf_level)/2)
        vmax = np.quantile(final_preds, (1+conf_level)/2)

        for ax in aaxs[-2,:]: # last aaxs row is for the arrows, we need the axis just above them
            ax.fill_between(
                x=[vmin, vmax],
                y1=[0,0],
                y2=[ax.get_ylim()[1],ax.get_ylim()[1]],
                color="gray",
                alpha=0.2)

    ## Add weighted average contribution to the bottom of the plot on the arrow Axes object
    n_cols = aaxs.shape[1]
    #define relative position of aroow subplot depending on n_cols
    pos_list = define_relative_position(n_cols)

    rule_preds = [preds[i][-1] for i in range(len(rules))]
    xlim_values = [ax.get_xlim() for ax in aaxs[0,:]]

    # Assing xlim values of arrowplot (aaxs[-1,:]) to be
    # equal to the xlim values of the ruleplot (aaxs[0,:])
    for ax, xlim in zip(aaxs[-1, :], xlim_values):
        ax.set_xlim(xlim)

    for j, pos in zip(range(n_cols), pos_list):

        # TODO: make class out of all this so that plot_arrow can inherit
        # attribtues from plot_rules and/or instances of BellatrexExplain()
        aaxs[-1, j] = plot_arrow(aaxs[-1, j], pos,
                                 weight=weights[j],
                                 pred_out=rule_preds[j],
                                 fontsize=base_fontsize,
                                 tot_digits=tot_digits)

    # Add final prediction to the plot
    final_pred = np.sum([weights[i] * preds[i][-1] for i in range(len(rules))])
    final_pred_str = frmt_pretty_print(final_pred, tot_digits)
    final_pred_str = f"Bellatrex weighted prediction: {final_pred_str}"
    final_pred_str += " (= " + " + ".join([
                            rf"{frmt_pretty_print(preds[i][-1], tot_digits)}"
                            rf"$\times${weights[i]:.2f}"
                            for i in range(len(rules))
                                            ]) + ")" + " "*12
    # extra spaces to the right to move text to the left

    if b_box_pred is not None:
        ypos_fin_text = 0.05
        bbox_pred_str = frmt_pretty_print(b_box_pred, tot_digits) # works both with single values and multi-output
        bbox_pred_str = f"\n(compared to black-box model prediction: {bbox_pred_str})      "
        # bbox_pred_str += ", ".join([f"{frmt_pretty_print(pred)}"
        #                              for pred in np.atleast_1d(b_box_pred)])
        # bbox_pred_str += ")"
    else:
        ypos_fin_text = 0.08
        bbox_pred_str = ""

    # plt.figtext(0.5, ypos_fin_text, final_pred_str+bbox_pred_str, fontsize=base_fontsize+2, ha="center")
    fig.text(0.5, ypos_fin_text, final_pred_str+bbox_pred_str, fontsize=base_fontsize+2, ha="center")

    return fig, aaxs


def parse(rulesplit):
    """Parses a rulesplit outputted by Bellatrex into a form suitable for visualisation."""

    # 1) Replace special characters by LaTeX symbols
    rulesplit = rulesplit.replace("≤" , "$\leq$")
    rulesplit = rulesplit.replace("<=", "$\leq$")
    rulesplit = rulesplit.replace("≥" , "$\geq$")
    rulesplit = rulesplit.replace(">=", "$\geq$")

    # 2) Remove information related to the current value, situated after the threshold value (right hand side)
    end_math_idx = rulesplit.rfind('$') + 1
    if end_math_idx and end_math_idx < len(rulesplit):
        # Get the substring after the last closing dollar sign
        right_substring = rulesplit[end_math_idx:]
        # Check and process for "(" in the substring after the last $
        if "(" in right_substring:
            # Find the last "(" in the substring and adjust it relative to the whole string
            last_paren_index = right_substring.rfind("(") + end_math_idx

            # Update rulesplit without the part after the last "("
            rulesplit = rulesplit[:last_paren_index].strip()


    return rulesplit


def read_rules(file, file_extra=None):

    """
    Reads and parses rules, predictions, baselines, and weights from a given file,
    with an optional additional file for extra predictions.

    The function processes the content of the specified file to extract rule weights, baseline predictions, rules,
    and corresponding predictions. If an additional file is provided, it extracts extra predictions as well.

    Args:
        file (str): The path to the primary file containing rules, weights, baselines, and predictions.
        file_extra (str, optional): The path to the additional file containing extra predictions. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - f_rules (list): A list of rules, where each rule is a list of conditions.
            - f_preds (list): A list of predictions corresponding to each rule.
            - f_baselines (list): A list of baseline predictions.
            - f_weights (list): A list of weights for each rule.
            - other_preds (list or None): List of ununsed rules stored in the -extra file,
                if provided; otherwise, None.
    """
    f_rules = []
    f_preds = []
    f_baselines = []
    f_weights = []
    with open(file, "r") as f:
        btrex_rules = f.readlines()
    for line in btrex_rules:
        if "RULE WEIGHT" in line:
            f_weights.append( float(line.split(":")[1].strip("\n").strip(" #")) )
        if "Baseline prediction" in line:
            f_baselines.append( float(line.split(":")[1].strip(" \n")) )
            rule = []
            pred = []
        if "node" in line:
            fullrule = line.split(":")[1].strip().strip("\n").split("-->")
            index_thresh = max([fullrule[0].find(char) for char in ["=","<",">"]])
            fullrule[0] = fullrule[0][0:index_thresh+8]
            rule.append( fullrule[0] )
            pred.append( float(fullrule[1]) )
        if "leaf" in line:
            f_rules.append(rule)
            f_preds.append(pred)

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

    return f_rules, f_preds, f_baselines, f_weights, other_preds
