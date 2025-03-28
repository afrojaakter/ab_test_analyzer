import numpy as np
import scipy.stats as stats
import math
import pandas as pd
import io
import base64

def calculate_significance(control_conversions, control_visitors, 
                          variant_conversions, variant_visitors, 
                          alpha=0.05):
    """
    Calculate statistical significance using a two-tailed Z-test for proportions.
    
    Parameters:
    -----------
    control_conversions: int
        Number of conversions in the control group
    control_visitors: int
        Number of visitors in the control group
    variant_conversions: int
        Number of conversions in the variant group
    variant_visitors: int
        Number of visitors in the variant group
    alpha: float
        Significance level (default: 0.05)
        
    Returns:
    --------
    tuple: (p_value, is_significant, z_score)
    """
    # Calculate conversion rates
    p1 = control_conversions / control_visitors if control_visitors > 0 else 0
    p2 = variant_conversions / variant_visitors if variant_visitors > 0 else 0
    
    # Calculate pooled probability
    p_pooled = (control_conversions + variant_conversions) / (control_visitors + variant_visitors)
    
    # Calculate standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/control_visitors + 1/variant_visitors))
    
    # Handle division by zero
    if se == 0:
        return 1.0, False, 0
    
    # Calculate z-score
    z_score = (p2 - p1) / se
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Check if the result is statistically significant
    is_significant = p_value < alpha
    
    return p_value, is_significant, z_score

def calculate_confidence_interval(conversions, visitors, confidence=0.95):
    """
    Calculate the confidence interval for a proportion using the Wilson score interval.
    
    Parameters:
    -----------
    conversions: int
        Number of conversions
    visitors: int
        Number of visitors
    confidence: float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    tuple: (lower_bound, upper_bound)
    """
    if visitors == 0:
        return (0, 0)
    
    # Calculate conversion rate
    p = conversions / visitors
    
    # Calculate Wilson score interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    # Calculate components
    denominator = 1 + (z**2 / visitors)
    centre_adjusted_probability = (p + z**2 / (2 * visitors)) / denominator
    adjusted_standard_error = z * math.sqrt((p * (1 - p) + z**2 / (4 * visitors)) / visitors) / denominator
    
    lower = max(0, centre_adjusted_probability - adjusted_standard_error)
    upper = min(1, centre_adjusted_probability + adjusted_standard_error)
    
    return (lower, upper)

def calculate_sample_size(baseline_rate, mde, confidence=0.95, power=0.8):
    """
    Calculate required sample size for an A/B test.
    
    Parameters:
    -----------
    baseline_rate: float
        The conversion rate of the control group
    mde: float
        Minimum detectable effect (relative change)
    confidence: float
        Confidence level (default: 0.95)
    power: float
        Statistical power (default: 0.8)
        
    Returns:
    --------
    int: Required sample size per variant
    """
    # Calculate variant rate based on baseline and MDE
    variant_rate = baseline_rate * (1 + mde)
    
    # Get z-scores for confidence and power
    z_alpha = stats.norm.ppf(1 - (1 - confidence) / 2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate pooled standard error
    p_pooled = (baseline_rate + variant_rate) / 2
    se_pooled = math.sqrt(2 * p_pooled * (1 - p_pooled))
    
    # Calculate effect size
    effect_size = abs(variant_rate - baseline_rate)
    
    # Calculate sample size
    n = ((z_alpha + z_beta) / effect_size * se_pooled) ** 2
    
    # Return rounded up sample size
    return math.ceil(n)
    
def chi_square_test(control_conversions, control_visitors, variant_conversions, variant_visitors, alpha=0.05):
    """
    Perform a chi-square test for independence to determine if there is a significant difference 
    between the expected frequencies and the observed frequencies in categories (control vs variant).
    
    Parameters:
    -----------
    control_conversions: int
        Number of conversions in the control group
    control_visitors: int
        Number of visitors in the control group
    variant_conversions: int
        Number of conversions in the variant group
    variant_visitors: int
        Number of visitors in the variant group
    alpha: float
        Significance level (default: 0.05)
        
    Returns:
    --------
    tuple: (p_value, is_significant, chi2_statistic)
    """
    # Create contingency table
    # [conversions, non-conversions]
    control_non_conversions = control_visitors - control_conversions
    variant_non_conversions = variant_visitors - variant_conversions
    
    contingency_table = [
        [control_conversions, control_non_conversions],
        [variant_conversions, variant_non_conversions]
    ]
    
    # Handle zero values
    if control_visitors == 0 or variant_visitors == 0:
        return 1.0, False, 0
    
    # Perform chi-square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Determine if result is significant
    is_significant = p_value < alpha
    
    return p_value, is_significant, chi2_stat

def t_test_for_means(control_mean, control_std, control_size, 
                     variant_mean, variant_std, variant_size, alpha=0.05):
    """
    Perform a two-sample t-test for means of continuous metrics.
    
    Parameters:
    -----------
    control_mean: float
        Mean value of the control group
    control_std: float
        Standard deviation of the control group
    control_size: int
        Sample size of the control group
    variant_mean: float
        Mean value of the variant group
    variant_std: float
        Standard deviation of the variant group
    variant_size: int
        Sample size of the variant group
    alpha: float
        Significance level (default: 0.05)
        
    Returns:
    --------
    tuple: (p_value, is_significant, t_statistic)
    """
    # Calculate t-statistic and p-value
    # Welch's t-test (does not assume equal variances)
    if control_size < 2 or variant_size < 2:
        return 1.0, False, 0
    
    # Calculate degrees of freedom
    dof = ((control_std**2 / control_size + variant_std**2 / variant_size)**2) / \
          ((control_std**2 / control_size)**2 / (control_size - 1) + 
           (variant_std**2 / variant_size)**2 / (variant_size - 1))
    
    # Calculate t-statistic
    t_stat = (control_mean - variant_mean) / \
             math.sqrt(control_std**2 / control_size + variant_std**2 / variant_size)
    
    # Calculate p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
    
    # Determine if result is significant
    is_significant = p_value < alpha
    
    return p_value, is_significant, t_stat

def generate_download_link(df, filename, text):
    """
    Generate a download link for a dataframe.
    
    Parameters:
    -----------
    df: pandas.DataFrame
        DataFrame to download
    filename: str
        Name of the file to download
    text: str
        Text to display on download link
        
    Returns:
    --------
    str: HTML link for downloading the data
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def parse_uploaded_csv(uploaded_file):
    """
    Parse an uploaded CSV file.
    
    Parameters:
    -----------
    uploaded_file: BytesIO
        Uploaded file object from Streamlit
        
    Returns:
    --------
    dict: Dictionary with extracted values or None if parsing failed
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check for valid columns
        required_columns = ['control_visitors', 'control_conversions', 
                           'variant_visitors', 'variant_conversions']
        
        if all(col in df.columns for col in required_columns):
            # Get the latest data (last row)
            latest = df.iloc[-1]
            
            return {
                'control_visitors': int(latest['control_visitors']),
                'control_conversions': int(latest['control_conversions']),
                'variant_visitors': int(latest['variant_visitors']),
                'variant_conversions': int(latest['variant_conversions'])
            }
        
        # Alternative format (for time series data)
        if 'date' in df.columns and 'group' in df.columns and 'visitors' in df.columns and 'conversions' in df.columns:
            control_data = df[df['group'] == 'control']
            variant_data = df[df['group'] == 'variant']
            
            if not control_data.empty and not variant_data.empty:
                return {
                    'control_visitors': int(control_data['visitors'].sum()),
                    'control_conversions': int(control_data['conversions'].sum()),
                    'variant_visitors': int(variant_data['visitors'].sum()),
                    'variant_conversions': int(variant_data['conversions'].sum())
                }
        
        return None
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return None
        
def multi_comparison_anova(data):
    """
    Perform ANOVA test to compare multiple variants.
    
    Parameters:
    -----------
    data: dict
        Dictionary with variant names as keys and lists of values as values
    
    Returns:
    --------
    tuple: (f_statistic, p_value, is_significant)
    """
    groups = list(data.values())
    
    # Check if we have at least 2 groups
    if len(groups) < 2:
        return 0, 1.0, False
    
    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Determine if result is significant
    is_significant = p_value < 0.05
    
    return f_stat, p_value, is_significant

def pairwise_tukey_hsd(data, labels=None, alpha=0.05):
    """
    Perform pairwise Tukey HSD test to compare all pairs of variants.
    
    Parameters:
    -----------
    data: dict
        Dictionary with variant names as keys and lists of values as values
    labels: list
        List of group labels (optional)
    alpha: float
        Significance level (default: 0.05)
        
    Returns:
    --------
    tuple: (results_df, all_pairs_df)
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import pandas as pd
    
    # Prepare data for Tukey's test
    all_values = []
    group_labels = []
    
    if labels is None:
        labels = list(data.keys())
    
    for i, (group, values) in enumerate(data.items()):
        all_values.extend(values)
        group_labels.extend([labels[i]] * len(values))
    
    # Perform Tukey's HSD test
    tukey_results = pairwise_tukeyhsd(all_values, group_labels, alpha=alpha)
    
    # Create dataframe with all pairs results
    all_pairs_df = pd.DataFrame({
        'Group 1': tukey_results.groupsunique[tukey_results.pairindices[:,0]],
        'Group 2': tukey_results.groupsunique[tukey_results.pairindices[:,1]],
        'Mean Diff': tukey_results.meandiffs,
        'P-Value': tukey_results.pvalues,
        'Significant': tukey_results.reject,
        'Lower CI': tukey_results.confint[:,0],
        'Upper CI': tukey_results.confint[:,1]
    })
    
    # Create summary dataframe
    results_df = pd.DataFrame({
        'Group': labels,
        'Mean': [np.mean(data[label] if label in data else []) for label in labels],
        'Count': [len(data[label] if label in data else []) for label in labels],
        'Std Dev': [np.std(data[label] if label in data else [], ddof=1) for label in labels]
    })
    
    return results_df, all_pairs_df

def multi_proportion_test(conversions, visitors, labels=None, alpha=0.05):
    """
    Perform chi-square test of independence for multiple proportions.
    
    Parameters:
    -----------
    conversions: list
        List of conversion counts for each variant
    visitors: list
        List of visitor counts for each variant
    labels: list
        List of group labels (optional)
    alpha: float
        Significance level (default: 0.05)
        
    Returns:
    --------
    tuple: (chi2_stat, p_value, is_significant, df)
    """
    import pandas as pd
    
    # Create contingency table
    contingency_table = np.array([
        conversions,
        np.array(visitors) - np.array(conversions)
    ])
    
    # Perform chi-square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Determine if result is significant
    is_significant = p_value < alpha
    
    # If labels not provided, create default ones
    if labels is None:
        labels = [f"Variant {chr(65+i)}" for i in range(len(conversions))]
    
    # Create dataframe with results
    df = pd.DataFrame({
        'Group': labels,
        'Conversions': conversions,
        'Visitors': visitors,
        'Conversion Rate': np.array(conversions) / np.array(visitors),
    })
    
    return chi2_stat, p_value, is_significant, df

def pairwise_proportion_test(conversions, visitors, labels=None, alpha=0.05):
    """
    Perform pairwise comparisons of proportions with Bonferroni correction.
    
    Parameters:
    -----------
    conversions: list
        List of conversion counts for each variant
    visitors: list
        List of visitor counts for each variant
    labels: list
        List of group labels (optional)
    alpha: float
        Significance level (default: 0.05)
        
    Returns:
    --------
    DataFrame: Results of pairwise comparisons
    """
    import pandas as pd
    from itertools import combinations
    
    # If labels not provided, create default ones
    if labels is None:
        labels = [f"Variant {chr(65+i)}" for i in range(len(conversions))]
    
    # Get all pairwise combinations
    n_groups = len(conversions)
    n_comparisons = n_groups * (n_groups - 1) // 2
    
    # Apply Bonferroni correction
    alpha_corrected = alpha / n_comparisons
    
    results = []
    
    for i, j in combinations(range(n_groups), 2):
        # For each pair, perform z-test for proportions
        p1 = conversions[i] / visitors[i]
        p2 = conversions[j] / visitors[j]
        
        # Calculate pooled probability
        p_pooled = (conversions[i] + conversions[j]) / (visitors[i] + visitors[j])
        
        # Calculate standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/visitors[i] + 1/visitors[j]))
        
        # Handle division by zero
        if se == 0:
            z_score = 0
            p_value = 1.0
        else:
            # Calculate z-score
            z_score = (p2 - p1) / se
            
            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Determine if significant with Bonferroni correction
        is_significant = p_value < alpha_corrected
        
        # Add result to list
        results.append({
            'Group 1': labels[i],
            'Group 2': labels[j],
            'Rate 1': p1,
            'Rate 2': p2,
            'Difference': p2 - p1,
            'Relative Diff': (p2 - p1) / p1 if p1 > 0 else np.inf,
            'Z-Score': z_score,
            'P-Value': p_value,
            'Significant': is_significant,
            'Alpha (corrected)': alpha_corrected
        })
    
    return pd.DataFrame(results)
