import pandas as pd


# csv_path = "H:/MDClone query results/combined/[2436][Synthetic][uri.marom][qe_1732][20200604_191726].csv"

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


def get_target_variable(orig_df, target_vars = ["target_should_hosp", "target_deceased"], return_full_df=True):
    print("\nExtracting Target Variable(s)")
    copy_df = orig_df.copy()
    # handle censored
    # copy_df = copy_df.replace("censored", None)
    # define target variables and print some stats
    # define target_deceased
    if "Reference Event-Decease date-Days from Reference" in copy_df.columns:
        copy_df["target_deceased"] = copy_df["Reference Event-Decease date-Days from Reference"].notnull()
        print("\ndeceased: %d (%d %%)" % (copy_df["target_deceased"].sum(), copy_df["target_deceased"].mean() * 100))
    # define target_corona_hosp - based on EVER (and not at/after) corona hosp event
    if "target - first corona hosp ever-Hospital entry date-Days from Reference" in copy_df.columns:
        copy_df["target_corona_hosp"] = copy_df["target - first corona hosp ever-Hospital entry date-Days from Reference"].notnull()
        print("corona hosp: %d (%d %%)" % (copy_df["target_corona_hosp"].sum(), copy_df["target_corona_hosp"].mean() * 100))
    # define target_worst_severity - based on EVER (and not at/after) corona hosp events by severity
    copy_df["target_worst_severity"] = copy_df.apply(lambda row: get_worst_severity(row), axis=1)
    print("\nworst severity (for corona hosp patients):")
    print(copy_df["target_worst_severity"].value_counts())
    # define target_should_hosp - a unified target, based on deceased OR any corona hosp
    copy_df["target_should_hosp"] = copy_df.apply(lambda row: get_should_hosp(row), axis=1)
    # print("# df["target_corona_hosp"].sum())
    print("\nshould hosp: %d (%d %%)" % (copy_df["target_should_hosp"].sum(), copy_df["target_should_hosp"].mean() * 100))
    if return_full_df:
        return copy_df
    else:
        return copy_df[target_vars]


def load_target_variable(csv_path):
    df = pd.read_csv(csv_path)
    # df = df[df['Reference Event-Age at event'] > 60]
    return get_target_variable(df)

