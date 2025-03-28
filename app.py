import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import utils

# Set page config
st.set_page_config(
    page_title="A/B Test Significance Calculator",
    page_icon="📊",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .significance-result {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .significant {
        background-color: #d4edda;
        color: #155724;
    }
    .not-significant {
        background-color: #f8d7da;
        color: #721c24;
    }
    .highlight {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 A/B Test Significance Calculator")
st.markdown("""
This app helps you determine if your A/B test results are statistically significant or merely due to random chance.
Simply input your test data below and get an instant significance analysis.
""")

# Create tabs for different features
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Basic A/B Test", 
    "Multiple Variations", 
    "Continuous Metrics", 
    "Time Series Analysis", 
    "Test Planning"
])

# ----------------------
# Basic A/B Test Tab
# ----------------------
with tab1:
    st.header("Basic A/B Test Analysis")
    st.markdown("Compare two variations (A and B) to determine if there's a statistically significant difference.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Control Group (A)")
        control_visitors = st.number_input("Visitors A", min_value=1, value=1000, step=1, key="cv_basic")
        control_conversions = st.number_input("Conversions A", min_value=0, value=150, step=1, key="cc_basic")
        
    with col2:
        st.subheader("Variant Group (B)")
        variant_visitors = st.number_input("Visitors B", min_value=1, value=1000, step=1, key="vv_basic")
        variant_conversions = st.number_input("Conversions B", min_value=0, value=180, step=1, key="vc_basic")
    
    # Add upload CSV option
    st.markdown("### Or upload your data")
    uploaded_file = st.file_uploader("Upload CSV with test data", type=["csv"], key="uploader_basic")
    
    if uploaded_file is not None:
        data = utils.parse_uploaded_csv(uploaded_file)
        if data:
            control_visitors = data["control_visitors"]
            control_conversions = data["control_conversions"]
            variant_visitors = data["variant_visitors"]
            variant_conversions = data["variant_conversions"]
            
            st.success("Data successfully loaded from CSV!")
        else:
            st.error("Could not parse CSV. Make sure it has the correct format.")
    
    # Calculate conversion rates and differences
    control_rate = control_conversions / control_visitors if control_visitors > 0 else 0
    variant_rate = variant_conversions / variant_visitors if variant_visitors > 0 else 0
    relative_difference = (variant_rate - control_rate) / control_rate if control_rate > 0 else 0
    absolute_difference = variant_rate - control_rate
    
    # Display metrics
    st.markdown("### Conversion Rates")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Control Conversion Rate", f"{control_rate:.2%}")
    
    with metric_col2:
        st.metric("Variant Conversion Rate", f"{variant_rate:.2%}")
    
    with metric_col3:
        st.metric("Relative Difference", f"{relative_difference:.2%}", 
                 delta=f"{absolute_difference:.2%}" if absolute_difference != 0 else None)
    
    # Statistical Test Selection
    test_method = st.selectbox(
        "Select Statistical Test",
        ["Z-test (Default)", "Chi-square Test"],
        index=0
    )
    
    significance_level = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    
    # Calculate significance based on selected test
    if test_method == "Z-test (Default)":
        p_value, is_significant, test_stat = utils.calculate_significance(
            control_conversions, control_visitors,
            variant_conversions, variant_visitors,
            significance_level
        )
        test_stat_name = "Z-score"
    else:  # Chi-square
        p_value, is_significant, test_stat = utils.chi_square_test(
            control_conversions, control_visitors,
            variant_conversions, variant_visitors,
            significance_level
        )
        test_stat_name = "Chi-square"
    
    # Calculate confidence intervals
    control_ci = utils.calculate_confidence_interval(control_conversions, control_visitors, 1 - significance_level)
    variant_ci = utils.calculate_confidence_interval(variant_conversions, variant_visitors, 1 - significance_level)
    
    # Display results
    st.markdown("### Results")
    
    # Significance result box
    if is_significant:
        st.markdown(f"""
        <div class="significance-result significant">
            ✅ Statistically Significant Result (p-value: {p_value:.4f})
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="significance-result not-significant">
            ❌ Not Statistically Significant (p-value: {p_value:.4f})
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results
    with st.expander("Detailed Results", expanded=True):
        st.markdown(f"""
        - **P-value**: {p_value:.6f}
        - **{test_stat_name}**: {test_stat:.4f}
        - **Significance Level (α)**: {significance_level}
        - **Control Confidence Interval**: [{control_ci[0]:.2%}, {control_ci[1]:.2%}]
        - **Variant Confidence Interval**: [{variant_ci[0]:.2%}, {variant_ci[1]:.2%}]
        """)
    
    # Visualization
    st.markdown("### Visualization")
    
    # Bar chart with confidence intervals
    fig = go.Figure()
    
    # Add bars for conversion rates
    fig.add_trace(go.Bar(
        x=['Control', 'Variant'],
        y=[control_rate, variant_rate],
        text=[f"{control_rate:.2%}", f"{variant_rate:.2%}"],
        textposition='auto',
        name='Conversion Rate',
        marker_color=['#636EFA', '#EF553B']
    ))
    
    # Add error bars for confidence intervals
    fig.add_trace(go.Scatter(
        x=['Control', 'Variant'],
        y=[control_rate, variant_rate],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[control_ci[1] - control_rate, variant_ci[1] - variant_rate],
            arrayminus=[control_rate - control_ci[0], variant_rate - variant_ci[0]],
            visible=True
        ),
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title='Conversion Rates with Confidence Intervals',
        xaxis_title='Variation',
        yaxis_title='Conversion Rate',
        yaxis_tickformat='.2%',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Opportunity to download the results
    results_df = pd.DataFrame({
        'Metric': ['Visitors', 'Conversions', 'Conversion Rate', 'Confidence Interval (Lower)', 'Confidence Interval (Upper)'],
        'Control': [control_visitors, control_conversions, control_rate, control_ci[0], control_ci[1]],
        'Variant': [variant_visitors, variant_conversions, variant_rate, variant_ci[0], variant_ci[1]]
    })
    
    st.markdown("### Export Results")
    st.markdown(utils.generate_download_link(results_df, 'ab_test_results.csv', '📥 Download Results as CSV'), unsafe_allow_html=True)

# ----------------------
# Multiple Variations Tab
# ----------------------
with tab2:
    st.header("Multiple Variations Analysis (A/B/C/n)")
    st.markdown("Compare multiple variations against a control to determine if there are statistically significant differences.")
    
    # Get number of variations
    num_variations = st.number_input("Number of Variations (Including Control)", min_value=2, max_value=10, value=3, step=1)
    
    variations = []
    cols = st.columns(num_variations)
    
    visitors_list = []
    conversions_list = []
    
    # Create input fields for each variation
    for i in range(num_variations):
        with cols[i]:
            if i == 0:
                st.subheader("Control (A)")
            else:
                st.subheader(f"Variant {chr(66+i-1)}")
            
            visitors = st.number_input(f"Visitors", min_value=1, value=1000, key=f"v{i}_multi")
            conversions = st.number_input(f"Conversions", min_value=0, value=150 + (i * 10), key=f"c{i}_multi")
            
            visitors_list.append(visitors)
            conversions_list.append(conversions)
    
    # Calculate conversion rates
    conversion_rates = [conversions / visitors if visitors > 0 else 0 for conversions, visitors in zip(conversions_list, visitors_list)]
    
    # Calculate significance
    try:
        results = utils.calculate_multiple_variations_significance(conversions_list, visitors_list, alpha=0.05)
        
        # Display results
        st.markdown("### Results")
        
        # Create a dataframe for results
        results_df = pd.DataFrame({
            'Variation': [f"Variant {chr(66+i)}" for i in range(num_variations-1)],
            'Conversion Rate': [f"{rate:.2%}" for rate in conversion_rates[1:]],
            'Lift vs Control': [f"{(rate-conversion_rates[0])/conversion_rates[0]:.2%}" if conversion_rates[0] > 0 else "N/A" for rate in conversion_rates[1:]],
            'P-value': [f"{r[0]:.6f}" for r in results],
            'Z-score': [f"{r[2]:.4f}" for r in results],
            'Significant': ["✅" if r[1] else "❌" for r in results],
            'Bonferroni-corrected P-value': [f"{r[3]:.6f}" for r in results],
            'Significant (corrected)': ["✅" if r[4] else "❌" for r in results]
        })
        
        st.dataframe(results_df)
        
        # Visualization
        st.markdown("### Visualization")
        
        # Bar chart with confidence intervals
        fig = go.Figure()
        
        # Add bars for conversion rates
        variation_names = ['Control'] + [f"Variant {chr(66+i)}" for i in range(num_variations-1)]
        colors = px.colors.qualitative.Plotly
        
        fig.add_trace(go.Bar(
            x=variation_names,
            y=conversion_rates,
            text=[f"{cr:.2%}" for cr in conversion_rates],
            textposition='auto',
            marker_color=colors[:num_variations]
        ))
        
        fig.update_layout(
            title='Conversion Rates Across Variations',
            xaxis_title='Variation',
            yaxis_title='Conversion Rate',
            yaxis_tickformat='.2%'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.markdown("### Export Results")
        st.markdown(utils.generate_download_link(results_df, 'multiple_variations_results.csv', '📥 Download Results as CSV'), unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error calculating significance: {e}")

# ----------------------
# Continuous Metrics Tab
# ----------------------
with tab3:
    st.header("Continuous Metrics Analysis")
    st.markdown("Compare two variations using t-test for continuous metrics like average order value, time on page, etc.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Control Group (A)")
        control_mean = st.number_input("Mean A", min_value=0.0, value=50.0, step=0.1, format="%.2f")
        control_std = st.number_input("Standard Deviation A", min_value=0.1, value=10.0, step=0.1, format="%.2f")
        control_size = st.number_input("Sample Size A", min_value=2, value=100, step=1)
        
    with col2:
        st.subheader("Variant Group (B)")
        variant_mean = st.number_input("Mean B", min_value=0.0, value=52.5, step=0.1, format="%.2f")
        variant_std = st.number_input("Standard Deviation B", min_value=0.1, value=10.0, step=0.1, format="%.2f")
        variant_size = st.number_input("Sample Size B", min_value=2, value=100, step=1)
    
    significance_level_t = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01, key="alpha_t")
    
    # Calculate significance
    p_value, is_significant, t_stat = utils.t_test_for_means(
        control_mean, control_std, control_size,
        variant_mean, variant_std, variant_size,
        significance_level_t
    )
    
    # Calculate relative difference
    relative_diff = (variant_mean - control_mean) / control_mean if control_mean != 0 else float('inf')
    
    # Display results
    st.markdown("### Results")
    
    # Significance result box
    if is_significant:
        st.markdown(f"""
        <div class="significance-result significant">
            ✅ Statistically Significant Result (p-value: {p_value:.4f})
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="significance-result not-significant">
            ❌ Not Statistically Significant (p-value: {p_value:.4f})
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results
    with st.expander("Detailed Results", expanded=True):
        st.markdown(f"""
        - **P-value**: {p_value:.6f}
        - **T-statistic**: {t_stat:.4f}
        - **Significance Level (α)**: {significance_level_t}
        - **Absolute Difference**: {variant_mean - control_mean:.4f}
        - **Relative Difference**: {relative_diff:.2%}
        """)
    
    # Visualization
    st.markdown("### Visualization")
    
    # Create normal distributions for visualization
    x_control = np.linspace(control_mean - 3*control_std, control_mean + 3*control_std, 100)
    y_control = stats.norm.pdf(x_control, control_mean, control_std)
    
    x_variant = np.linspace(variant_mean - 3*variant_std, variant_mean + 3*variant_std, 100)
    y_variant = stats.norm.pdf(x_variant, variant_mean, variant_std)
    
    # Plot distributions
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_control,
        y=y_control,
        mode='lines',
        name='Control',
        fill='tozeroy',
        line=dict(color='#636EFA')
    ))
    
    fig.add_trace(go.Scatter(
        x=x_variant,
        y=y_variant,
        mode='lines',
        name='Variant',
        fill='tozeroy',
        line=dict(color='#EF553B')
    ))
    
    # Add mean lines
    fig.add_vline(x=control_mean, line_dash="dash", line_color="#636EFA")
    fig.add_vline(x=variant_mean, line_dash="dash", line_color="#EF553B")
    
    fig.update_layout(
        title='Distribution Comparison',
        xaxis_title='Value',
        yaxis_title='Density',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Opportunity to download the results
    results_df = pd.DataFrame({
        'Metric': ['Mean', 'Standard Deviation', 'Sample Size', 'T-statistic', 'P-value', 'Significant'],
        'Control': [control_mean, control_std, control_size, '', '', ''],
        'Variant': [variant_mean, variant_std, variant_size, t_stat, p_value, is_significant]
    })
    
    st.markdown("### Export Results")
    st.markdown(utils.generate_download_link(results_df, 'continuous_metrics_results.csv', '📥 Download Results as CSV'), unsafe_allow_html=True)

# ----------------------
# Time Series Analysis Tab
# ----------------------
with tab4:
    st.header("Time Series Analysis")
    st.markdown("See how statistical significance evolves over time in your A/B test.")
    
    # Two ways to provide time series data
    data_option = st.radio(
        "How would you like to provide time series data?",
        ["Upload CSV", "Generate Sample Data"]
    )
    
    time_series_data = None
    
    if data_option == "Upload CSV":
        uploaded_ts_file = st.file_uploader(
            "Upload CSV with daily data",
            type=["csv"],
            help="CSV should have columns: date, group (control/variant), visitors, conversions"
        )
        
        if uploaded_ts_file:
            try:
                df = pd.read_csv(uploaded_ts_file)
                if all(col in df.columns for col in ['date', 'group', 'visitors', 'conversions']):
                    time_series_data = df
                    st.success("Data successfully loaded!")
                else:
                    st.error("CSV must have columns: date, group, visitors, conversions")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
    
    else:  # Generate sample data
        st.markdown("### Generate Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            days = st.slider("Number of Days", 7, 60, 30)
            control_cr = st.slider("Control Conversion Rate", 0.01, 0.50, 0.15, 0.01)
            daily_visitors = st.slider("Daily Visitors per Variation", 100, 5000, 1000, 100)
        
        with col2:
            effect_start_day = st.slider("Effect Start Day", 1, days, days // 2)
            variant_lift = st.slider("Variant Lift", -0.5, 2.0, 0.15, 0.05)
            random_seed = st.slider("Random Seed", 1, 999, 42)
        
        np.random.seed(random_seed)
        
        # Generate dates
        start_date = datetime.now() - timedelta(days=days)
        date_list = [start_date + timedelta(days=i) for i in range(days)]
        date_strs = [date.strftime('%Y-%m-%d') for date in date_list]
        
        # Generate data
        data_rows = []
        
        for i, date in enumerate(date_strs):
            # Control group
            control_visitors = np.random.randint(int(daily_visitors * 0.9), int(daily_visitors * 1.1))
            control_conversions = np.random.binomial(control_visitors, control_cr)
            
            data_rows.append({
                'date': date,
                'group': 'control',
                'visitors': control_visitors,
                'conversions': control_conversions
            })
            
            # Variant group with effect starting from effect_start_day
            variant_visitors = np.random.randint(int(daily_visitors * 0.9), int(daily_visitors * 1.1))
            if i >= effect_start_day - 1:
                variant_cr = control_cr * (1 + variant_lift)
            else:
                variant_cr = control_cr
                
            variant_conversions = np.random.binomial(variant_visitors, variant_cr)
            
            data_rows.append({
                'date': date,
                'group': 'variant',
                'visitors': variant_visitors,
                'conversions': variant_conversions
            })
        
        time_series_data = pd.DataFrame(data_rows)
        
        st.markdown("### Preview of Generated Data")
        st.dataframe(time_series_data.head(6))
    
    # If time series data is available, analyze it
    if time_series_data is not None:
        try:
            dates, p_values, is_significant_list, control_rates, variant_rates = utils.analyze_time_series(time_series_data)
            
            # Display results
            st.markdown("### Time Series Results")
            
            # Convert to strings for display
            date_strs = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in dates]
            
            # Create dataframe for results
            results_df = pd.DataFrame({
                'Date': date_strs,
                'Control Rate': [f"{cr:.2%}" for cr in control_rates],
                'Variant Rate': [f"{vr:.2%}" for vr in variant_rates],
                'Lift': [f"{(vr-cr)/cr:.2%}" if cr > 0 else "N/A" for vr, cr in zip(variant_rates, control_rates)],
                'P-Value': [f"{pv:.4f}" for pv in p_values],
                'Significant': ["✅" if sig else "❌" for sig in is_significant_list]
            })
            
            # Show tabular data
            with st.expander("Show Tabular Data", expanded=False):
                st.dataframe(results_df)
            
            # Visualization
            st.markdown("### Visualization")
            
            # Create line chart for conversion rates
            fig1 = go.Figure()
            
            fig1.add_trace(go.Scatter(
                x=date_strs,
                y=control_rates,
                mode='lines+markers',
                name='Control Rate',
                line=dict(color='#636EFA')
            ))
            
            fig1.add_trace(go.Scatter(
                x=date_strs,
                y=variant_rates,
                mode='lines+markers',
                name='Variant Rate',
                line=dict(color='#EF553B')
            ))
            
            fig1.update_layout(
                title='Conversion Rates Over Time',
                xaxis_title='Date',
                yaxis_title='Conversion Rate',
                yaxis_tickformat='.2%',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            # Create line chart for p-values
            fig2 = go.Figure()
            
            # Add significance threshold line
            fig2.add_hline(y=0.05, line_dash="dash", line_color="red")
            
            fig2.add_trace(go.Scatter(
                x=date_strs,
                y=p_values,
                mode='lines+markers',
                name='P-Value',
                line=dict(color='#00CC96')
            ))
            
            # Color regions based on significance
            for i in range(len(date_strs)):
                if is_significant_list[i]:
                    fig2.add_vrect(
                        x0=i-0.5,
                        x1=i+0.5,
                        fillcolor="green",
                        opacity=0.15,
                        layer="below",
                        line_width=0
                    )
            
            fig2.update_layout(
                title='P-Values Over Time (Red Line = 0.05 Threshold)',
                xaxis_title='Date',
                yaxis_title='P-Value',
                yaxis_type="log",
                showlegend=False
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
            
            # Download time series data
            st.markdown("### Export Results")
            st.markdown(utils.generate_download_link(results_df, 'time_series_results.csv', '📥 Download Results as CSV'), unsafe_allow_html=True)
            
            # Current test status
            st.markdown("### Current Test Status")
            final_p_value = p_values[-1]
            final_significance = is_significant_list[-1]
            
            if final_significance:
                st.markdown(f"""
                <div class="significance-result significant">
                    ✅ Currently Statistically Significant (p-value: {final_p_value:.4f})
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="significance-result not-significant">
                    ❌ Not Yet Statistically Significant (p-value: {final_p_value:.4f})
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error analyzing time series data: {e}")

# ----------------------
# Test Planning Tab
# ----------------------
with tab5:
    st.header("Test Planning")
    st.markdown("Plan your A/B test: Calculate required sample size and test duration based on expected metrics.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Test Parameters")
        baseline_cr = st.slider("Baseline Conversion Rate", 0.001, 0.50, 0.05, 0.001, format="%.3f")
        mde = st.slider("Minimum Detectable Effect (Relative Lift)", 0.01, 1.00, 0.15, 0.01)
        
        st.markdown(f"### Goals")
        st.markdown(f"""
        You want to detect a change of at least **{mde:.0%}** from the baseline conversion rate of **{baseline_cr:.2%}**.
        
        This means detecting when the variant rate is ≥ **{baseline_cr * (1 + mde):.2%}** or ≤ **{baseline_cr * (1 - mde):.2%}**.
        """)
    
    with col2:
        st.subheader("Statistical Parameters")
        confidence = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
        power = st.slider("Statistical Power", 0.7, 0.99, 0.8, 0.01)
        
        st.markdown(f"### What this means")
        st.markdown(f"""
        - **{confidence:.0%} Confidence**: You want to be {confidence:.0%} certain that a detected effect is real
        - **{power:.0%} Power**: You want to be {power:.0%} certain that you can detect the minimum effect if it exists
        """)
    
    # Calculate sample size
    sample_size = utils.calculate_sample_size(baseline_cr, mde, confidence, power)
    
    st.markdown("### Required Sample Size")
    st.markdown(f"""
    <div class="highlight">
        <h4>You need <span style="color:#FF4B4B;">{sample_size:,}</span> visitors per variation</h4>
        <p>Total visitors needed: <strong>{sample_size * 2:,}</strong> (for both control and variant)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Test duration calculator
    st.subheader("Calculate Test Duration")
    
    daily_traffic = st.number_input("Average Daily Visitors (to your entire website)", min_value=10, value=5000, step=100)
    traffic_allocation = st.slider("Traffic Allocation to Test (%)", 10, 100, 50, 5) / 100
    
    daily_test_traffic = daily_traffic * traffic_allocation
    daily_visitors_per_variant = daily_test_traffic / 2
    
    test_days = utils.calculate_test_duration(baseline_cr, mde, daily_visitors_per_variant, confidence, power)
    
    st.markdown("### Test Duration Estimate")
    st.markdown(f"""
    <div class="highlight">
        <h4>Your test will take approximately <span style="color:#FF4B4B;">{test_days}</span> days</h4>
        <p>Based on:</p>
        <ul>
            <li>Daily traffic: {daily_traffic:,} visitors</li>
            <li>Test allocation: {traffic_allocation:.0%}</li>
            <li>Each variation gets: {int(daily_visitors_per_variant):,} visitors per day</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional information
    with st.expander("Tips for Successful A/B Testing"):
        st.markdown("""
        ### Best Practices
        
        1. **Always decide on your success metrics before starting the test**
           - Avoid changing your success metrics during or after the test
        
        2. **Run the test for the pre-calculated duration**
           - Don't stop early just because you see significance
           - Don't extend indefinitely if you don't see the results you want
        
        3. **Consider seasonal factors**
           - Make sure your test period doesn't include unusual events or holidays
           - For highly seasonal businesses, plan tests during representative periods
        
        4. **Start with high-impact changes**
           - Test changes that have the potential for larger impact first
           - Small changes often require larger sample sizes to detect significance
        
        5. **Test only one change at a time**
           - Multivariate testing requires exponentially larger sample sizes
           - Simple A/B tests make it clear what caused the observed effect
        
        6. **Ensure random assignment**
           - Make sure users are randomly assigned to variations
           - Check that your test groups have similar characteristics
        """)

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This tool is designed to help you analyze and plan A/B tests with statistical rigor. 
It implements industry-standard statistical methods to ensure your test results are reliable.

For binary metrics (like conversion rates), we use the Z-test for proportions or Chi-Square test.
For continuous metrics (like revenue, time on page), we use Welch's t-test which doesn't assume equal variances.

Multiple comparison corrections (like Bonferroni) are applied when testing multiple variations to control the family-wise error rate.
""")