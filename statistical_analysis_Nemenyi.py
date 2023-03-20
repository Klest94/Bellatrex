import numpy as np
import os
import pandas as pd

SAVE_OUTPUTS = True

#from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from IPython import get_ipython
from utilities import rename_data_index

get_ipython().run_line_magic('matplotlib', 'inline')

SETUP = "bin"
PROJ_METHOD = "PCA"
FEAT_REPRESENTATION = "by_samples"
STRUCTURE = "rules"
#EXTRA_NOTES = ""

SIRUS_RULES = 10

NOTES = PROJ_METHOD + "_" + FEAT_REPRESENTATION + "_"

##########################################################################
#root_folder = "C:/Users/u0135479/Documents/GitHub/Explain-extract-trees/"
root_folder = os.getcwd()

results_folder = os.path.join(root_folder, "Final_results")


''' MAIN CORE: performance. also include dissimilarity related measures '''

competing_filename = NOTES + STRUCTURE.capitalize() + "_" + SETUP.capitalize() + "_Competing.csv"
simple_setup_filename = PROJ_METHOD + "_simple_" + STRUCTURE.capitalize() + "_" + SETUP.capitalize() + ".csv"
weight_setup_filename = PROJ_METHOD +  "_by_samples_" + STRUCTURE.capitalize() + "_" + SETUP.capitalize() + ".csv"

''' ABLATION studies '''

ablat2_filename = "Ablat2_" + NOTES + STRUCTURE.capitalize() + "_" + SETUP.capitalize()   + ".csv"
noTrees_filename = "noTrees_" + NOTES + STRUCTURE.capitalize() + "_" + SETUP.capitalize()   + ".csv"
noDims_filename = "noDims_"+ NOTES + STRUCTURE.capitalize() + "_" + SETUP.capitalize()   + ".csv"

''' rule length files here'''

rule_length_file = NOTES + STRUCTURE.capitalize() + "_l_rules_" + SETUP.capitalize() + ".csv"



''' read all needed files here, store in dataframe '''

import Orange as orange

df_competing = pd.read_csv(os.path.join(results_folder, competing_filename), index_col=[0])
df_simple = pd.read_csv(os.path.join(results_folder, simple_setup_filename), index_col=[0])
df_weight = pd.read_csv(os.path.join(results_folder, weight_setup_filename), index_col=[0])
df_rules = pd.read_csv(os.path.join(results_folder, rule_length_file), index_col=[0])


data_names = df_competing.index

# all COMMON competitors list
common_perf_list = ["Best (S)T", "Single D(S)T", "orig. R(S)F", "Mini R(S)F "]
common_dissim_list = ["BestTs diss", "Mini R(S)F diss"]
#common_dissim_list_surv = ["BestTs diss", "Mini R(S)F diss"]


df_noTrees =  pd.read_csv(os.path.join(results_folder, noTrees_filename), index_col=[0])
df_noDims =  pd.read_csv(os.path.join(results_folder, noDims_filename), index_col=[0])
df_Ablat2 =  pd.read_csv(os.path.join(results_folder, ablat2_filename), index_col=[0])


#%%


#df_simple = df_simple["Local perf"].to_frame()
df_simple.rename(columns={"Local perf" : "Bellatrex, simple"}, inplace=True)
df_weight.rename(columns={"Local perf" : "Bellatrex, wt."}, inplace=True)
df_weight.rename(columns={"Local dissim" : "Bellatrex"}, inplace=True)


if SETUP in ["bin", "multi"]:
    common_perf_list.append("Logistic-Regr")
    if SETUP == "bin":
        df_SIRUS = pd.read_csv(os.path.join(results_folder, "SIRUS_binary.csv"), index_col=[0])
        df_C443 = pd.read_csv(os.path.join(results_folder, "C443_results.csv"), index_col=[0])
        df_COSI = pd.read_csv(os.path.join(results_folder, "RuleCOSI_Bin.csv"), index_col=[0])
        #df_RuleFit = pd.read_csv(os.path.join(results_folder, "RuleFit_Bin.csv"), index_col=[0])
        df_HS = pd.read_csv(os.path.join(results_folder, "HS_Bin.csv"), index_col=[0])

        
        df_complete = pd.concat([df_competing[common_perf_list],
                                 df_SIRUS[str(SIRUS_RULES)+"_rules_perf"],
                                 df_C443["AUROC"],
                                 df_COSI["RuleCOSI"],
                                 #df_RuleFit["RuleFit20"], # or 200 for longer rules
                                 df_HS["HS_20"], # or 200
                                 df_simple["Bellatrex, simple"],
                                 df_weight["Bellatrex, wt."]
                                 ], axis=1)
        
        
    else: # multi
        #df_RuleFit = pd.read_csv(os.path.join(results_folder, "RuleFit_Multi.csv"), index_col=[0])
        df_HS = pd.read_csv(os.path.join(results_folder, "HS_Multi.csv"), index_col=[0])


        df_complete = pd.concat([df_competing[common_perf_list],
                                 df_simple["Bellatrex, simple"],
                                 df_HS["HS_20"],
                                 #df_RuleFit["RuleFit20"], # or 200 for longer rules
                                 df_weight["Bellatrex, wt."]
                                 ], axis=1)
    
elif SETUP in ["regress", "mtr"]:
    common_perf_list.append("Ridge-Regr")
    if SETUP == "regress":
        df_SIRUS = pd.read_csv(os.path.join(results_folder, "SIRUS_regression.csv"), index_col=[0])
        #df_RuleFit = pd.read_csv(os.path.join(results_folder, "RuleFit_Regress.csv"), index_col=[0])
        df_HS = pd.read_csv(os.path.join(results_folder, "HS_Regress.csv"), index_col=[0])

        
        df_complete = pd.concat([df_competing[common_perf_list],
                                 df_SIRUS[str(SIRUS_RULES)+"_rules_perf"],
                                 df_HS["HS_20"],
                                 #df_RuleFit["RuleFit20"], # or 200
                                 df_simple["Bellatrex, simple"],
                                 df_weight["Bellatrex, wt."]
                                 ], axis=1)
        
        df_complete.rename(columns={
                                    str(SIRUS_RULES)+"_rules_perf" : "SIRUS",
                                    }, inplace=True, errors="ignore")
    else: # in mtr setting
        #df_RuleFit = pd.read_csv(os.path.join(results_folder, "RuleFit_Mtr.csv"), index_col=[0])
        df_HS = pd.read_csv(os.path.join(results_folder, "HS_Mtr.csv"), index_col=[0])
        
        df_complete = pd.concat([df_competing[common_perf_list],
                                 #df_RuleFit["RuleFit20"], # or 200
                                 df_HS["HS_20"], # or 200
                                 df_simple["Bellatrex, simple"],
                                 df_weight["Bellatrex, wt."],
                                 ], axis=1)

elif SETUP in ["surv"]:
    common_perf_list.append("Cox-PH")
    df_complete = pd.concat([df_competing[common_perf_list],
                             df_simple["Bellatrex, simple"],
                             df_weight["Bellatrex, wt."]
                             ], axis=1)

else:
    KeyError("the SETUP variable seems wrong, double check.")
    
    
    
''' rename all columns except for the R(S)F related ones:

        - rename dataset names to converge to a constant, shortened form
'''
    
df_complete.rename(columns={"10_rules_perf" : "SIRUS",
                            str(SIRUS_RULES)+"_rules_perf" : "SIRUS",
                            "AUROC": "C443",
                            "Ridge-Regr": "LR", #same name corss scenarios
                            # hopefully repeated keys will not cause problems...
                            "Logistic-Regr" : "LR",
                            "RuleFit20": "RuleFit",
                            "RuleFit200": "RuleFit",
                            "RuleCOSI": "RuleCOSI+",
                            "HS_20": "HS",
                            "HS_200": "HS"
                            }, inplace=True, errors="ignore")


#### dissimilarity analysis

df_dissim = pd.concat([df_competing[common_dissim_list],
                           df_weight["Bellatrex"]
                            ], axis=1)


####   rule length

#df_trees = df_competing[["BestT n_splits", "n_rules", "MiniRF n_splits"]] #simple comeptitors and n paths
#df_rules["Avg. rule length"] # Ltreex (weighted)

df_compare_rules = pd.concat([df_competing[["BestT n_splits",
                                            "MiniRF n_splits",
                                            "R(S)F n_splits",
                                            "Single D(S)T n_splits"]],
                              df_rules["Avg. rule length"],
                              df_competing["n_rules"]
                            ], axis=1)


## add rule-length competitors: SIRUS, RuleFit, RuleCOSI
if SETUP in ["bin", "regress"]:
    df_compare_rules["SIRUS"] = df_SIRUS[str(SIRUS_RULES)+"_rules_length"]
    if SETUP == "bin":
        df_compare_rules["RuleCOSI+ (display)"] = df_COSI["all ruleslength"]
        df_compare_rules["RuleCOSI+"] = df_COSI["ruleslength read"]
        df_compare_rules["C443"] = df_C443["complexity"]
        # ruleCOSI hass "read length" and "tot rulesplit"options

if SETUP != "surv":
    #df_compare_rules["RuleFit"] = df_RuleFit["rulelength_20"] # or 2000
    df_compare_rules["HS"] = df_HS["rulelength_20"] # or 2000
    df_compare_rules["HS (display)"] = df_HS["n leaves20"]-1 # num internal nodes
    

#%% ablation study statistics here:

#df_ablations = pd.DataFrame()
df_noTrees = df_noTrees["Local perf"].rename("no step 1")
df_noDims = df_noDims["Local perf"].rename("no step 3")
df_Ablat2 = df_Ablat2["Local perf"].rename("no steps 1 \& 3")

df_ablations = pd.concat([df_weight["Bellatrex, wt."],
                          df_noTrees,
                          df_noDims,
                          df_Ablat2], axis=1
                          )

df_ablations.rename(columns={"Bellatrex, wt.": "Bellatrex"}, inplace=True, errors="ignore")



### rename table columns for better plotting readability ###

if SETUP not in ["surv"]:
    df_complete.rename(columns={"orig. R(S)F": "RF",
                                "Best (S)T": "OOB Trees",
                                "Single D(S)T": "DT",
                                "Mini R(S)F ": "Small RF"},
                       inplace=True, errors="ignore")
    
    df_dissim.rename(columns={"BestTs diss": "OOB Trees",
                              "Mini R(S)F diss": "Small RF"
                              },
                     inplace=True, errors="ignore")
    
    if SETUP == "bin":
        df_dissim["C443"] = df_C443["dissimilarity"]
    
    df_compare_rules.rename(columns={"BestT n_splits": "OOB Trees",
                                     "n_rules" : "n. rules",
                                     "MiniRF n_splits": "Small RF",
                                     "Avg. rule length": "Bellatrex",
                                     "R(S)F n_splits": "RF",
                                     "Single D(S)T n_splits": "DT"
                            },  inplace=True, errors="ignore")
    
    
    
    df_complete.insert(0, 'RF', df_complete.pop('RF'))
    df_complete.insert(1, 'Bellatrex, wt.', df_complete.pop('Bellatrex, wt.'))
    df_complete.insert(2, 'Bellatrex, simple', df_complete.pop('Bellatrex, simple'))
        
    
else: # if is "surv"
    df_complete.rename(columns={"Cox-PH" : "Cox PH",
                                "orig. R(S)F": "RSF",
                                "Best (S)T": "OOB S. Trees",
                                "Single D(S)T": "SDT",
                                "Mini R(S)F ": "Small RSF"},
                       inplace=True, errors="ignore")
    
    
    df_dissim.rename(columns={"BestTs diss": "OOB S.Trees",
                              "Mini R(S)F diss": "Small RSF"
                              },
                     inplace=True, errors="ignore")
    
    
    df_compare_rules.rename(columns={"BestT n_splits": "OOB S. Trees",
                                     "n_rules" : "n. rules",
                                     "MiniRF n_splits": "Small RSF",
                                     "Avg. rule length": "Bellatrex",
                                     "Single D(S)T n_splits": "SDT",
                                     "R(S)F n_splits": "RSF"
                                     },
                            inplace=True, errors="ignore")
    
    
    df_complete.insert(0, 'RSF', df_complete.pop('RSF'))
    df_complete.insert(1, 'Bellatrex, wt.', df_complete.pop('Bellatrex, wt.'))
    df_complete.insert(2, 'Bellatrex, simple', df_complete.pop('Bellatrex, simple'))
    
    
if SETUP == "surv":
    df_complete.drop("ALS-imputed-70", axis=0, errors="ignore", inplace=True)
    df_competing.drop("ALS-imputed-70", axis=0, errors="ignore", inplace=True)
    df_dissim.drop("ALS-imputed-70", axis=0, errors="ignore", inplace=True)
    df_compare_rules.drop("ALS-imputed-70", axis=0, errors="ignore", inplace=True)    
    df_ablations.drop("ALS-imputed-70", axis=0, errors="ignore", inplace=True)    


# update averages , given that ALS-imputed dataset is dropped

df_complete.loc['average'] = np.round(df_complete.mean(axis=0), 4)
#df_competing['average'] = df_competing.mean(axis=0)
df_dissim.loc['average'] =  np.round(df_dissim.mean(axis=0), 4)
df_ablations.loc['average'] =  np.round(df_ablations.mean(axis=0), 4)
df_compare_rules.loc['average'] =  np.round(df_compare_rules.mean(axis=0), 4)


#%%

if SAVE_OUTPUTS == True:
    
    df_complete.to_csv(os.path.join(results_folder, "Performance_" + SETUP.upper() + ".csv"))
    #df_complete.to_excel(os.path.join(results_folder, "FINAL_perform_" + SETUP.upper() + ".xlsx"),
    #                     float_format="%.4f")
    df_dissim.to_csv(os.path.join(results_folder, "Dissimil_" + SETUP.upper() + ".csv"))
    #df_dissim.to_excel(os.path.join(results_folder, "FINAL_dissim_" + SETUP.upper() + ".xlsx"),
    #                     float_format="%.4f")
    df_compare_rules.to_csv(os.path.join(results_folder, "Complexity_" + SETUP.upper() + ".csv"))


df_compare_rules.insert(0, 'Bellatrex', df_compare_rules.pop('Bellatrex'))
    
df_analysis = df_complete.iloc[:-1] # drop average performance
df_dissim_anal = df_dissim.iloc[:-1]
df_rules_anal = df_compare_rules[:-1]#.drop("avg. paths", axis=1)
df_ablations_anal = df_ablations.iloc[:-1]

smaller_is_better = False if SETUP in ["bin", "multi", "surv"] else True

### final computations here, everything is set

from scipy import stats

#%%

def orange_plot(df, filename, is_ascending, save_outputs, exclude=[],
                plot_title="Nemenyi test", reverse_ranks=False,
                dpi_figure=100):
    
    # is ascending: for computing ranks. If true lowest ir ranked 1
    # reverse ranks: if true, greatest average rank is plotted to the left
    # instead of the right
    
    df.drop(exclude, axis=1, errors="ignore", inplace=True) # "overloading" the list, bad practise...
    
    list_friedman = []
    for col in df.columns:
        list_friedman.append(df[col])

    Friedman_stat = stats.friedmanchisquare(*list_friedman)
    print('Friedman H0 p-value:', Friedman_stat[1])
    print("df:", Friedman_stat[0])
    
    ranks = df.T.rank(method="dense", ascending=is_ascending)
    avg_ranks = ranks.mean(axis=1)

    names = df.columns
    n_samples = df.shape[0]

    ### TODO consider Wilcoxon-Holm post-hoc procedure!

    cd = orange.evaluation.compute_CD(avg_ranks, n_samples)
    
    plot_top= int(min(avg_ranks))
    plot_bott= int(max(avg_ranks))+1
    
    orange.evaluation.graph_ranks(avg_ranks, names, cd=cd,
                                  lowv=plot_top, highv=plot_bott,
                                  width=3.7, textspace=1,
                                  reverse=reverse_ranks)
    plt.rcParams['figure.dpi'] = dpi_figure
    plt.tight_layout()
    plt.title(plot_title)
    figsize = plt.rcParams['figure.figsize']
        
    figsize[1] = figsize[1]*1.1
    
    plt.rcParams['figure.figsize'] = figsize


    if SAVE_OUTPUTS == True: # some setting here (and the few lines above, are off)
        plt.savefig(os.path.join(results_folder, SETUP.upper() + "_" + filename + ".png"))#,
                    #bbox_inches="tight")
        plt.savefig(os.path.join(results_folder, SETUP.upper() + "_" + filename + ".pdf")
                    , bbox_inches='tight', pad_inches=0.1, transparent=True
                    )

    return plt.show()


#%%

def output_latex(df, col_format, highlight_min, drop_cols=[], exclude=[]): # {:,.4f}
    assert isinstance(col_format, str)
    
    df = df.drop(drop_cols, axis=1, errors="ignore", inplace=False)
    
    df = df.applymap(col_format.format) #'{:,.4f}' or '{:.2f}'
    #df2 = df.style.format('{:.2f}')
    for col in df.columns:
        df[col] = df[col].astype(float)
    
    latex_names = []#list(df.index)
    for name in df.index:
        latex_names.append(name.replace("_", " "))
    df.index = latex_names
    
    subsets = df.columns.drop(exclude, errors="ignore") # "overloading" the list, bad practise...
    
    if highlight_min == True:
        out_latex = df.style.highlight_min(axis=1, subset=subsets,
                                props='textbf:--rwrap;').to_latex(hrules=True)
    elif highlight_min == False:
        out_latex = df.style.highlight_max(axis=1, subset=subsets,
                                props='textbf:--rwrap;').to_latex(hrules=True)
        
    elif highlight_min in ["None", None]:
        out_latex = df.to_latex(hrules=True)
    else:
        KeyError("Highlight option not recognized")
        
    out_latex = out_latex.replace("_", " ") #test this... does it work?
    
    return out_latex
    

df_analysis.rename(columns={"Bellatrex, wt.": "Bellatrex"}, inplace=True, errors="ignore")
df_dissim_anal.rename(columns={"Bellatrex, wt.": "Bellatrex"}, inplace=True, errors="ignore")
df_rules_anal.rename(columns={"Bellatrex, wt.": "Bellatrex"}, inplace=True, errors="ignore")


orange_plot(df_analysis, "Performance", smaller_is_better, save_outputs=True,
            exclude=["Bellatrex, simple"],
            plot_title="Performance comparison",
            dpi_figure=300) # perf: higher is better except (mt)r

orange_plot(df_dissim_anal, "Dissimil_"+FEAT_REPRESENTATION, False, save_outputs=True,
            exclude=["n. rules"],
            plot_title="Dissimilarity comparison",
            dpi_figure=300) #dissim: higher is better

orange_plot(df_rules_anal, "Complexity_"+FEAT_REPRESENTATION, is_ascending=True,
            save_outputs=False,
            exclude=["n. rules", "HS (display)", "RuleCOSI+ (display)",
                     "R(S)F n_splits", "RF", "RSF"],#, "HS (show)", "RuleCOSI (show)"],
            plot_title="Complexity of explanations",
            reverse_ranks=False,
            dpi_figure=300) #rules: lower is better

orange_plot(df_ablations_anal, "Ablation", smaller_is_better, save_outputs=True, exclude=["n. trees"],
            plot_title="Ablation study",
            dpi_figure=300) #ablat. perf: higher is better except (mt)r

df_nrules = df_compare_rules.pop("n. rules")
df_compare_rules.insert(1, "n. rules", df_nrules)

df_dissim["n. rules"] = df_competing["n_rules"]
#df_compare_rules#.drop("n. trees", axis=1, inplace=True, errors="ignore")

for df in [df_complete, df_dissim, df_compare_rules, df_ablations]:
    df = rename_data_index(df, SETUP)
    
#%%

latex_perfs = output_latex(df_complete, '{:,.4f}', smaller_is_better,
                           exclude=["RF", "RSF"],
                           drop_cols=["Bellatrex, simple"])

latex_diss = output_latex(df_dissim, '{:,.4f}', False, exclude=["n. rules"])

latex_rules = output_latex(df_compare_rules, '{:,.2f}', True,
                           drop_cols=["RF", "RSF", "HS (display)", "RuleCOSI+ (display)", "R(S)F n_splits"],
                           exclude=["n. rules"])

latex_ablation = output_latex(df_ablations, '{:,.4f}', highlight_min=smaller_is_better, exclude=["n. trees"])



#%% PERFORMANCE - INTERPRETABILITY TRADE OFF

performances = df_complete.loc["average"]
performances.rename({"Bellatrex, wt.": "Bellatrex"}, inplace=True, errors="ignore")
complexity = df_compare_rules.loc["average"]

perf_idx = performances.index
complex_idx = complexity.index

#%% intersect to find methods in common ( for which both PERF and COMPLEX is available)
perf_idx = perf_idx.intersection(complex_idx)

performances = performances.loc[perf_idx]
complexity = complexity.loc[perf_idx]

#%%

measures_dict = {"bin": "AUROC",
                "surv": "C-index",
                "mtr": "weighted MAE",
                "regress" : "MAE",
                "multi": "weighted AUROC"}

measure = measures_dict[SETUP]
#%% gwtting the data right ( indeces must correspond) for trade-off plotting
interpretability = 1/complexity

x = interpretability.values
y = performances.values
names = performances.index

#%% plot trade-off and Pareto Frontier

def plot_pareto_frontier(Xs, Ys, maxX=True, maxY=True, fontsize=12):
    
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    #sorting according to (descending) Xs. Highest Xs is guaranteed \in Pareto
    pareto_front = [sorted_list[0]] #initialise list with first member
    for pair in sorted_list[1:]:
        if maxY:        #if bigger is better
            if pair[1] >= pareto_front[-1][1]: #if y >= last y of the front, add
                pareto_front.append(pair)
        else:           # if smaller y is better
            if pair[1] <= pareto_front[-1][1]:  # add if y <= than last in Pareto
                pareto_front.append(pair)
    
    '''Plotting process'''
    plt.scatter(Xs,Ys)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    lines, = ax.plot(pf_X, pf_Y, color='red', marker='o', linestyle='dashed')
    lines.set_label("Pareto frontier")
    ax.legend(fontsize=fontsize)
    
    
    return None


fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

plt.rcParams['text.usetex'] = True
# fig.set_size_inches(7.5, 4)
# fig.set_dpi(100)
FIG_FONT = 15

#plt.figure(figsize=(5,3), dpi=80)
ax.scatter(x, y, s=FIG_FONT-6)
plt.title("Performance-interpretability trade-off", fontsize=FIG_FONT+2)

# We change the fontsize of minor ticks label 
ax.tick_params(axis='both', which='major', labelsize=FIG_FONT-2)
ax.tick_params(axis='both', which='minor', labelsize=FIG_FONT-4)

plot_pareto_frontier(x,y, maxX=True, maxY=(not smaller_is_better), fontsize=FIG_FONT-5)

if smaller_is_better:
    ax.invert_yaxis()
#plt.xlabel(r'\textbf{time (s)}')
plt.ylabel("{}".format(measure), fontsize=FIG_FONT-1)

y_lims = list(ax.get_ylim())
x_lims = list(ax.get_xlim())

pad_prop = 0.05 #add some padding to the plot ( increase axis range)
                # since the FIG_FONT is bigger than usual

# formula works also when axis are flipped (pads top and right corner anyway)
y_lims[1] = y_lims[1]+(y_lims[1]-y_lims[0])*(pad_prop) 
x_lims[1] = x_lims[1]+(x_lims[1]-x_lims[0])*(pad_prop/2)

text_pad = 1.5*1e-3

plt.xlabel('$1 / \mathcal{C}$', fontsize=FIG_FONT+1)
for i, txt in enumerate(names):
    ax.annotate(txt, xy=(x[i], y[i]), 
                xytext=(x[i]+text_pad, y[i]+text_pad),
                fontsize=FIG_FONT)

ax.set_ylim(tuple(y_lims))
ax.set_xlim(tuple(x_lims))

plt.savefig(os.path.join(results_folder, "Trade-off_"+SETUP.upper() +".pdf"))
plt.show()


#%%

'''
BETTER STRING MANAGEMENT WITH REGULAR EXPRESSIONS

STANDARDISE DATASET NAMES: shorten them consistently,
manually provide list of indeces
'''


import re
import copy


# re.compile to define the pattern to be searched

# find occurences with .finditer -> generates iterable with all matches


pattern = re.compile("(\.\d{2,4})(00)")

def regex_replacer(in_string, in_pattern):
    
    matches = in_pattern.finditer(in_string)
    for match in matches:
        in_string = in_string.replace(match.group(0), match.group(1))
        
    return in_string

latex_ablation = latex_ablation.replace(r"average", "\\midrule \nAverage")
latex_perfs = latex_perfs.replace(r"average", "\\midrule \nAverage")
latex_diss = latex_diss.replace(r"average", "\\midrule \nAverage")
latex_rules = latex_rules.replace(r"average", "\\midrule \nAverage")


latex_perfs = latex_perfs.replace(", wt.", "")


latex_diss2 = regex_replacer(latex_diss, pattern)
latex_perfs2 = regex_replacer(latex_perfs, pattern)
latex_rules2 = regex_replacer(latex_rules, pattern)
#repeat for latex_rules, since it has only 2 digits after the . separator
latex_rules2 = regex_replacer(latex_rules2, pattern)
latex_ablation2 = regex_replacer(latex_ablation, pattern)
    
    
#%%
#latex_rules3 = copy.copy(latex_rules2)
latex_rules3 = ""

pattern2 = re.compile("\d{1,2}\s(\&\s(\textbf{)?\d{1,2}\.\d{2}\}?\s\&)")


#  WANRING: CAL500 & 29.29 ... does not work as expected! 

for line in latex_rules2.split("\n"):
    #print(line)
    line_clean = line.replace("lrr", "lr", 1)
    line_clean = line.replace("& n. rules &", "(n. rules) &")
    
    matches = pattern2.finditer(line)
    matchn = list(matches)[0:1] #works beacuse n. rules is the 2nd column

    for match in matchn:
        #print(match.group(1))
        out_string =  copy.copy(match.group(1))
        out_string = out_string.replace("& ", "(", 1)
        out_string = out_string.replace(" &", ") &", 1)
        
        line_clean = line.replace(match.group(1), out_string)
        #print(line_clean)
    latex_rules3 += line_clean + "\n"

        
latex_rules2 = copy.copy(latex_rules3)

del latex_rules3

#df_ablations_extra = copy.copy(df_ablations)
#df_ablations_extra["delta step1"] = df_ablations_extra['Bellatrex'] - df_ablations_extra['no step 1']
#df_ablations_extra["delta step2"] = df_ablations_extra['Bellatrex'] - df_ablations_extra['no step 2']
#df_ablations_extra["delta step12"] = df_ablations_extra['Bellatrex'] - df_ablations_extra['no steps 1-2']