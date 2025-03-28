import numpy as np
import scipy.stats as stats
import math

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
