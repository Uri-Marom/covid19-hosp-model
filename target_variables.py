import pandas as pd


csv_path = "H:/MDClone query results/combined/[2436][Synthetic][uri.marom][qe_1732][20200604_191726].csv"
df = pd.read_csv(csv_path)


def get_worst_severity(row):
    if not(pd.isna(row["target - first severe corona hosp ever-Hospital entry date-Days from Reference"])):
        return "severe"
    elif not(pd.isna(row["target - first moderate corona hosp ever-Hospital entry date-Days from Reference"])):
        return "moderate"
    elif not(pd.isna(row["target - first mild corona hosp ever-Hospital entry date-Days from Reference"])):
        return "mild"
    elif not(pd.isna(row["target - first corona hosp ever-Hospital entry date-Days from Reference"])):
        return "missing"
    else:
        return None


def get_should_hosp(row):
    return row["target_deceased"] | row["target_corona_hosp"]


def add_target_variables(orig_df):
    copy_df = orig_df.copy()
    # handle censored
    # define target_deceased
    copy_df["target_deceased"] = copy_df["Reference Event-Decease date"].notnull()
    # define target_corona_hosp - based on EVER (and not at/after) corona hosp event
    copy_df["target_corona_hosp"] = copy_df["target - first corona hosp ever-Hospital entry date-Days from Reference"].notnull()
    # define target_worst_severity - based on EVER (and not at/after) corona hosp events by severity
    copy_df["target_worst_severity"] = copy_df.apply(lambda row: get_worst_severity(row), axis=1)
    # define target_should_hosp - a unified target, based on deceased OR any corona hosp
    copy_df["target_should_hosp"] = copy_df.apply(lambda row: get_should_hosp(row), axis=1)
    # print some stats
    print("\ndeceased: %d (%d %%)" % (copy_df["target_deceased"].sum(), copy_df["target_deceased"].mean() * 100))
    print("corona hosp: %d (%d %%)" % (copy_df["target_corona_hosp"].sum(), copy_df["target_corona_hosp"].mean() * 100))
    # print("# df["target_corona_hosp"].sum())
    print("\nworst severity (for corona hosp patients):")
    print(copy_df["target_worst_severity"].value_counts())
    print("\nshould hosp: %d (%d %%)" % (copy_df["target_should_hosp"].sum(), copy_df["target_should_hosp"].mean() * 100))
    return copy_df


df = add_target_variables(df)
print(df.head())
