from collections import Counter

def label_site_error_analysis(y_true, y_pred, sites):
    error_pairs = [(true, pred, site) for true, pred, site in zip(y_true, y_pred, sites) if true != pred]
    error_analysis = Counter(error_pairs)
    return error_analysis

def label_site_all_analysis(y_true, y_pred, sites):
    error_pairs = [(true, pred, site) for true, pred, site in zip(y_true, y_pred, sites)]
    error_analysis = Counter(error_pairs)
    return error_analysis