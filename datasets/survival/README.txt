when adding a new dataset to the folder, please make sure it respoects the format:

- last column: "T_survival" for survival time
- 2nd to last:L "Status" for observation status: =0 censored, =1 observed

UPADTE: column names can change, but column order cannot


DAT CLEANING PROCEDURE (see file preprocessing Survival Data.py)

- drop columns with more than 50% missing values (threshold can vary)
- drop rows with more than 50% missing values (threshold can vary)
- one-hot encoding of categorical variables
- make sure / force "Status" and "T_survival" columns are still the last ones of the DataFrame
- MICE imputation with IterativeImputer(max_iter=20, random_state=1)


gbsg2 dataset: not sure whether the original "cens" column is indeed equal to 1-Status
Probably NOT: "cens"= "Status"

NHANES-I dataset: what is the event?
