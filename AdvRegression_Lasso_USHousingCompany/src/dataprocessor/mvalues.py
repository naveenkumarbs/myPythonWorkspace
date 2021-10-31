


def missing_values(housing):
    return housing.isna().sum()*100/len(housing)


def column_analyse(housing, col):
    return housing[col].value_counts(normalize=True, dropna=False)


def continuous_var(housing):
    return housing._get_numeric_data().columns