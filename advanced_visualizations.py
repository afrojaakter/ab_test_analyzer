import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import math

def generate_time_series_simulation(control_rate, variant_rate, visitors_per_day=100, days=14, 
                                    confidence_level=0.95, random_seed=42):
    """
    Simulate a time series of A/B test results to show how significance evolves over time.
    
    Parameters:
    -----------
    control_rate: float
        The true conversion rate of the control group
    variant_rate: float
        The true conversion rate of the variant group
    visitors_per_day: int
        Number of visitors per day for each variant
    days: int
        Number of days to simulate
    confidence_level: float
        Confidence level (default: 0.95)
    random_seed: int
        Random seed for reproducibility
    
    Returns:
    --------
    dict: Dictionary with time series data and plot
    """
    np.random.seed(random_seed)
    
    # Initialize data structures
    dates = pd.date_range(start='2023-01-01', periods=days)
    cumulative_data = {
        'date': dates,
        'control_visitors': [],
        'control_conversions': [],
        'control_rate': [],
        'variant_visitors': [],
        'variant_conversions': [],
        'variant_rate': [],
        'p_value': [],
        'significant': [],
        'relative_lift': [],
        'cum_relative_lift': []
    }
    
    total_control_visitors = 0
    total_control_conversions = 0
    total_variant_visitors = 0
    total_variant_conversions = 0
    
    # Get Z critical value for the two-tailed test
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    # Calculate daily results and cumulative metrics
    for day in range(days):
        # Generate daily conversions based on the true rates
        control_daily_visitors = visitors_per_day
        variant_daily_visitors = visitors_per_day
        
        # Add some randomness to visitors (±10%)
        control_daily_visitors = int(control_daily_visitors * np.random.uniform(0.9, 1.1))
        variant_daily_visitors = int(variant_daily_visitors * np.random.uniform(0.9, 1.1))
        
        # Generate conversions
        control_daily_conversions = np.random.binomial(control_daily_visitors, control_rate)
        variant_daily_conversions = np.random.binomial(variant_daily_visitors, variant_rate)
        
        # Update cumulative totals
        total_control_visitors += control_daily_visitors
        total_control_conversions += control_daily_conversions
        total_variant_visitors += variant_daily_visitors
        total_variant_conversions += variant_daily_conversions
        
        # Calculate rates
        daily_control_rate = control_daily_conversions / control_daily_visitors
        daily_variant_rate = variant_daily_conversions / variant_daily_visitors
        
        cumulative_control_rate = total_control_conversions / total_control_visitors
        cumulative_variant_rate = total_variant_conversions / total_variant_visitors
        
        # Calculate daily relative lift
        daily_relative_lift = ((daily_variant_rate - daily_control_rate) / daily_control_rate 
                              if daily_control_rate > 0 else 0)
        
        # Calculate cumulative relative lift
        cum_relative_lift = ((cumulative_variant_rate - cumulative_control_rate) / cumulative_control_rate 
                            if cumulative_control_rate > 0 else 0)
        
        # Calculate p-value for cumulative data
        # Pooled conversion rate
        pooled_rate = (total_control_conversions + total_variant_conversions) / (total_control_visitors + total_variant_visitors)
        
        # Standard error
        se = math.sqrt(pooled_rate * (1 - pooled_rate) * (1/total_control_visitors + 1/total_variant_visitors))
        
        # Z-score
        z_score = (cumulative_variant_rate - cumulative_control_rate) / se if se > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Determine if significant
        significant = p_value < alpha
        
        # Update data
        cumulative_data['control_visitors'].append(total_control_visitors)
        cumulative_data['control_conversions'].append(total_control_conversions)
        cumulative_data['control_rate'].append(cumulative_control_rate)
        cumulative_data['variant_visitors'].append(total_variant_visitors)
        cumulative_data['variant_conversions'].append(total_variant_conversions)
        cumulative_data['variant_rate'].append(cumulative_variant_rate)
        cumulative_data['p_value'].append(p_value)
        cumulative_data['significant'].append(significant)
        cumulative_data['relative_lift'].append(daily_relative_lift)
        cumulative_data['cum_relative_lift'].append(cum_relative_lift)
    
    # Create DataFrame
    df = pd.DataFrame(cumulative_data)
    
    # Create plot
    fig = go.Figure()
    
    # Add conversion rates
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['control_rate'],
        mode='lines+markers',
        name='Control Rate',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['variant_rate'],
        mode='lines+markers',
        name='Variant Rate',
        line=dict(color='green')
    ))
    
    # Add significance threshold line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=[alpha] * len(df),
        mode='lines',
        name=f'Significance Threshold (p={alpha:.2f})',
        line=dict(color='red', dash='dash')
    ))
    
    # Add p-value line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['p_value'],
        mode='lines+markers',
        name='p-value',
        line=dict(color='orange'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Time Series View of A/B Test Significance',
        xaxis_title='Date',
        yaxis=dict(
            title='Conversion Rate',
            tickformat='.2%'
        ),
        yaxis2=dict(
            title=dict(text='p-value', font=dict(color='orange')),
            tickfont=dict(color='orange'),
            anchor='x',
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500
    )
    
    # Add markers for when significance is achieved
    significance_index = None
    for i, row in df.iterrows():
        if row['significant']:
            significance_index = i
            break
    
    # Instead of adding a vline for significance, add a custom annotation and shape
    if significance_index is not None:
        # Add a shape for the significance line - using a line shape instead of vline
        significance_day = significance_index + 1  # +1 for day number (1-indexed)
        
        # Add a band to highlight the significant day
        fig.add_shape(
            type="rect",
            x0=significance_day - 0.5,
            x1=significance_day + 0.5,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            fillcolor="rgba(0, 255, 0, 0.2)",
            line=dict(width=0),
        )
        
        # Add annotation for the significance
        fig.add_annotation(
            x=significance_day,
            y=1,
            xref="x",
            yref="paper",
            text="Significance Achieved",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color="green", size=12),
            bgcolor="white",
            bordercolor="green",
            borderwidth=1,
        )
    
    return {'df': df, 'fig': fig, 'reached_significance': significance_index is not None}

def create_distribution_comparison(control_mean, control_std, variant_mean, variant_std, 
                                   sample_points=1000, random_seed=42):
    """
    Create a visualization comparing the distributions of the control and variant groups.
    
    Parameters:
    -----------
    control_mean: float
        Mean of the control group
    control_std: float
        Standard deviation of the control group
    variant_mean: float
        Mean of the variant group
    variant_std: float
        Standard deviation of the variant group
    sample_points: int
        Number of points to sample for the distribution
    random_seed: int
        Random seed for reproducibility
    
    Returns:
    --------
    plotly.graph_objects.Figure: Distribution comparison plot
    """
    np.random.seed(random_seed)
    
    # Generate sample distributions
    x = np.linspace(min(control_mean - 3*control_std, variant_mean - 3*variant_std),
                   max(control_mean + 3*control_std, variant_mean + 3*variant_std),
                   sample_points)
    
    control_dist = stats.norm.pdf(x, control_mean, control_std)
    variant_dist = stats.norm.pdf(x, variant_mean, variant_std)
    
    # Calculate overlap and separation
    overlap = np.minimum(control_dist, variant_dist).sum() / max(control_dist.sum(), variant_dist.sum())
    effect_size = abs(control_mean - variant_mean) / ((control_std + variant_std) / 2)
    
    # Create plot
    fig = go.Figure()
    
    # Add distributions
    fig.add_trace(go.Scatter(
        x=x,
        y=control_dist,
        mode='lines',
        name='Control Distribution',
        line=dict(color='blue'),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 255, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=variant_dist,
        mode='lines',
        name='Variant Distribution',
        line=dict(color='green'),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.2)'
    ))
    
    # Add mean lines
    fig.add_vline(x=control_mean, line_dash="dash", line_color="blue",
                 annotation_text="Control Mean", annotation_position="top right")
    fig.add_vline(x=variant_mean, line_dash="dash", line_color="green",
                 annotation_text="Variant Mean", annotation_position="top right")
    
    # Add overlap text
    fig.add_annotation(
        x=0.5,
        y=0.9,
        xref="paper",
        yref="paper",
        text=f"Distribution Overlap: {overlap:.1%}<br>Effect Size: {effect_size:.2f}",
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig.update_layout(
        title='Distributions Comparison',
        xaxis_title='Value',
        yaxis_title='Probability Density',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500
    )
    
    return fig

def create_distribution_comparison_categorical(control_rate, variant_rate, control_size=1000, variant_size=1000, random_seed=42):
    """
    Create a visualization comparing the binomial distributions of the control and variant groups for conversion rates.
    
    Parameters:
    -----------
    control_rate: float
        Conversion rate of the control group
    variant_rate: float
        Conversion rate of the variant group
    control_size: int
        Sample size of the control group
    variant_size: int
        Sample size of the variant group
    random_seed: int
        Random seed for reproducibility
    
    Returns:
    --------
    plotly.graph_objects.Figure: Distribution comparison plot
    """
    np.random.seed(random_seed)
    
    # Generate samples for visualization
    control_samples = np.random.binomial(control_size, control_rate, size=1000) / control_size
    variant_samples = np.random.binomial(variant_size, variant_rate, size=1000) / variant_size
    
    # Create histogram plot
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=control_samples,
        name='Control',
        opacity=0.7,
        marker_color='blue',
        histnorm='probability density'
    ))
    
    fig.add_trace(go.Histogram(
        x=variant_samples,
        name='Variant',
        opacity=0.7,
        marker_color='green',
        histnorm='probability density'
    ))
    
    # Calculate and visualize confidence intervals
    control_ci_lower, control_ci_upper = proportion_confidence_interval(control_rate, control_size)
    variant_ci_lower, variant_ci_upper = proportion_confidence_interval(variant_rate, variant_size)
    
    # Add vertical lines for observed rates
    fig.add_vline(x=control_rate, line_dash="solid", line_color="blue",
                  annotation_text="Control Rate", annotation_position="top right")
    fig.add_vline(x=variant_rate, line_dash="solid", line_color="green",
                  annotation_text="Variant Rate", annotation_position="top right")
    
    # Add confidence intervals
    fig.add_vrect(
        x0=control_ci_lower, x1=control_ci_upper,
        line_width=0, fillcolor="blue", opacity=0.1,
        annotation_text="Control 95% CI", annotation_position="bottom right"
    )
    
    fig.add_vrect(
        x0=variant_ci_lower, x1=variant_ci_upper,
        line_width=0, fillcolor="green", opacity=0.1,
        annotation_text="Variant 95% CI", annotation_position="bottom left"
    )
    
    # Calculate overlap between confidence intervals
    if control_ci_upper < variant_ci_lower or variant_ci_upper < control_ci_lower:
        overlap = "No overlap (significant difference)"
    else:
        overlap = "Confidence intervals overlap (may not be significant)"
    
    # Add overlap text
    fig.add_annotation(
        x=0.5,
        y=0.9,
        xref="paper",
        yref="paper",
        text=overlap,
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig.update_layout(
        title='Conversion Rate Distributions Comparison',
        xaxis_title='Conversion Rate',
        yaxis_title='Probability Density',
        barmode='overlay',
        xaxis=dict(tickformat='.2%'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500
    )
    
    return fig

def proportion_confidence_interval(p, n, confidence=0.95):
    """
    Calculate Wilson score interval for a proportion.
    
    Parameters:
    -----------
    p: float
        The observed proportion
    n: int
        The sample size
    confidence: float
        Confidence level (default: 0.95)
    
    Returns:
    --------
    tuple: (lower_bound, upper_bound)
    """
    # Get Z critical value
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    # Calculate Wilson score interval
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    ci_half_width = z * math.sqrt(p * (1 - p) / n + z**2/(4*n**2)) / denominator
    
    lower_bound = max(0, center - ci_half_width)
    upper_bound = min(1, center + ci_half_width)
    
    return lower_bound, upper_bound

def create_power_analysis_curve(baseline_rate, mde_range=None, sample_sizes=None, confidence=0.95, power=0.8):
    """
    Create a power analysis curve to visualize the relationship between 
    minimum detectable effect, sample size, and statistical power.
    
    Parameters:
    -----------
    baseline_rate: float
        The baseline conversion rate
    mde_range: list
        Range of minimum detectable effects to visualize (relative changes)
    sample_sizes: list
        List of sample sizes to consider
    confidence: float
        Confidence level (default: 0.95)
    power: float
        Statistical power (default: 0.8)
    
    Returns:
    --------
    plotly.graph_objects.Figure: Power analysis curve plot
    """
    # Default values if not provided
    if mde_range is None:
        mde_range = np.linspace(0.01, 0.3, 30)  # 1% to 30% relative change
    
    if sample_sizes is None:
        sample_sizes = [1000, 2000, 5000, 10000, 20000]
    
    # Calculate required sample sizes for each MDE
    mde_sample_sizes = []
    for mde in mde_range:
        required_n = calculate_sample_size_for_proportion(baseline_rate, mde, confidence, power)
        mde_sample_sizes.append(required_n)
    
    # Create primary plot: MDE vs Sample Size
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=mde_range,
        y=mde_sample_sizes,
        mode='lines+markers',
        name=f'Required Sample Size (per variant)',
        line=dict(color='blue')
    ))
    
    fig1.update_layout(
        title='Sample Size vs Minimum Detectable Effect',
        xaxis=dict(
            title='Minimum Detectable Effect (Relative Change)',
            tickformat='.0%'
        ),
        yaxis=dict(
            title='Required Sample Size (per variant)',
            type='log'
        ),
        height=500
    )
    
    # Create secondary plot: Power curves for different sample sizes
    mde_values = np.linspace(0.01, 0.3, 100)
    fig2 = go.Figure()
    
    for n in sample_sizes:
        power_values = []
        for mde in mde_values:
            # Calculate power for this sample size and MDE
            effect_size = mde * baseline_rate  # Absolute effect
            power_val = calculate_power_for_proportion(baseline_rate, baseline_rate + effect_size, n, n, confidence)
            power_values.append(power_val)
        
        fig2.add_trace(go.Scatter(
            x=mde_values,
            y=power_values,
            mode='lines',
            name=f'n = {n:,}'
        ))
    
    # Add horizontal line at the specified power
    fig2.add_hline(y=power, line_dash="dash", line_color="red",
                   annotation_text=f"Target Power ({power:.0%})", annotation_position="right")
    
    fig2.update_layout(
        title='Statistical Power vs Minimum Detectable Effect',
        xaxis=dict(
            title='Minimum Detectable Effect (Relative Change)',
            tickformat='.0%'
        ),
        yaxis=dict(
            title='Statistical Power',
            tickformat='.0%',
            range=[0, 1]
        ),
        height=500,
        legend=dict(
            title="Sample Size (per variant)"
        )
    )
    
    return {'sample_size_curve': fig1, 'power_curves': fig2}

def calculate_sample_size_for_proportion(p1, mde, confidence=0.95, power=0.8):
    """
    Calculate required sample size for an A/B test with conversion rates.
    
    Parameters:
    -----------
    p1: float
        The baseline conversion rate
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
    # Calculate absolute effect size
    p2 = p1 * (1 + mde)
    
    # Get Z values for confidence and power
    alpha = 1 - confidence
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate pooled proportion
    p_pooled = (p1 + p2) / 2
    
    # Calculate required sample size
    numerator = (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
    denominator = (p2 - p1)**2
    
    n = numerator / denominator
    
    return math.ceil(n)

def calculate_power_for_proportion(p1, p2, n1, n2, confidence=0.95):
    """
    Calculate the statistical power for a given sample size and effect size.
    
    Parameters:
    -----------
    p1: float
        Conversion rate of the control group
    p2: float
        Conversion rate of the variant group
    n1: int
        Sample size of the control group
    n2: int
        Sample size of the variant group
    confidence: float
        Confidence level (default: 0.95)
    
    Returns:
    --------
    float: Statistical power (0 to 1)
    """
    # Get Z critical value for confidence
    alpha = 1 - confidence
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Pooled proportion for null hypothesis
    p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    
    # Standard error under null hypothesis
    se_null = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Standard error under alternative hypothesis
    se_alt = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    
    # Non-centrality parameter
    delta = abs(p2 - p1) / se_null
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_alpha - delta) + stats.norm.cdf(-z_alpha - delta)
    
    return power