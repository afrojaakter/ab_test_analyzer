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


def calculate_multiple_variations_significance(conversions_list, visitors_list, alpha=0.05):
    """
    Calculate statistical significance for multiple variations compared to control.
    Uses Bonferroni correction for multiple comparisons.
    
    Parameters:
    -----------
    conversions_list: list
        List of conversion counts for each variation (first item is control)
    visitors_list: list
        List of visitor counts for each variation (first item is control)
    alpha: float
        Significance level (default: 0.05)
        
    Returns:
    --------
    list: List of tuples for each variation vs control comparison 
          [(p_value, is_significant, z_score, corrected_p_value, is_significant_corrected)]
    """
    if len(conversions_list) != len(visitors_list):
        raise ValueError("Conversions list and visitors list must have the same length")
    
    if len(conversions_list) < 2:
        raise ValueError("Must have at least two variations (control + variant)")
    
    # Control values (first items in the lists)
    control_conversions = conversions_list[0]
    control_visitors = visitors_list[0]
    
    # Number of comparisons (for Bonferroni correction)
    num_comparisons = len(conversions_list) - 1
    
    # Adjusted alpha for Bonferroni correction
    adjusted_alpha = alpha / num_comparisons
    
    results = []
    
    # Compare each variant to control
    for i in range(1, len(conversions_list)):
        p_value, is_significant, z_score = calculate_significance(
            control_conversions, control_visitors,
            conversions_list[i], visitors_list[i],
            alpha
        )
        
        # Apply Bonferroni correction
        corrected_p_value = min(p_value * num_comparisons, 1.0)
        is_significant_corrected = corrected_p_value < alpha
        
        results.append((
            p_value, 
            is_significant, 
            z_score, 
            corrected_p_value, 
            is_significant_corrected
        ))
    
    return results

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
                'variant_conversions': int(latest['variant_conversions']),
                'time_series_data': df if 'date' in df.columns else None
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
                    'variant_conversions': int(variant_data['conversions'].sum()),
                    'time_series_data': df
                }
        
        return None
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return None
        
def analyze_time_series(df):
    """
    Analyze time series data to see how significance evolved over time.
    
    Parameters:
    -----------
    df: pandas.DataFrame
        DataFrame with time series data with columns:
        - date: Date of the data point
        - group: 'control' or 'variant'
        - visitors: Number of visitors on that date for that group
        - conversions: Number of conversions on that date for that group
        
    Returns:
    --------
    tuple: (dates, p_values, is_significant_list, control_rates, variant_rates)
    """
    if 'date' not in df.columns or 'group' not in df.columns or 'visitors' not in df.columns or 'conversions' not in df.columns:
        raise ValueError("DataFrame must have 'date', 'group', 'visitors', and 'conversions' columns")
    
    # Convert date to datetime if it's not already
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Get unique dates
    dates = df['date'].unique()
    
    p_values = []
    is_significant_list = []
    control_rates = []
    variant_rates = []
    
    # Calculate cumulative metrics for each date
    for i, date in enumerate(dates):
        # Get data up to and including this date
        date_df = df[df['date'] <= date]
        
        # Split into control and variant
        control_data = date_df[date_df['group'] == 'control']
        variant_data = date_df[date_df['group'] == 'variant']
        
        # Calculate cumulative metrics
        control_visitors = control_data['visitors'].sum()
        control_conversions = control_data['conversions'].sum()
        variant_visitors = variant_data['visitors'].sum()
        variant_conversions = variant_data['conversions'].sum()
        
        # Calculate conversion rates
        control_rate = control_conversions / control_visitors if control_visitors > 0 else 0
        variant_rate = variant_conversions / variant_visitors if variant_visitors > 0 else 0
        
        control_rates.append(control_rate)
        variant_rates.append(variant_rate)
        
        # Calculate significance
        if i > 0 and control_visitors > 0 and variant_visitors > 0:
            p_value, is_significant, _ = calculate_significance(
                control_conversions, control_visitors,
                variant_conversions, variant_visitors
            )
            p_values.append(p_value)
            is_significant_list.append(is_significant)
        else:
            p_values.append(1.0)
            is_significant_list.append(False)
    
    return dates, p_values, is_significant_list, control_rates, variant_rates

def calculate_test_duration(baseline_rate, mde, daily_visitors_per_variant, confidence=0.95, power=0.8):
    """
    Calculate how long an A/B test should run based on traffic and desired MDE.
    
    Parameters:
    -----------
    baseline_rate: float
        The conversion rate of the control group
    mde: float
        Minimum detectable effect (relative change)
    daily_visitors_per_variant: int
        Number of visitors per variant per day
    confidence: float
        Confidence level (default: 0.95)
    power: float
        Statistical power (default: 0.8)
        
    Returns:
    --------
    int: Number of days the test should run
    """
    # Calculate required sample size per variant
    sample_size = calculate_sample_size(baseline_rate, mde, confidence, power)
    
    # Calculate number of days needed
    days = math.ceil(sample_size / daily_visitors_per_variant)
    
    return days
