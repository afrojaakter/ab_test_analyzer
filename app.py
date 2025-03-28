import streamlit as st
import numpy as np
import pandas as pd
import math
from utils import (
    calculate_significance, calculate_confidence_interval, calculate_sample_size,
    chi_square_test, t_test_for_means, generate_download_link, parse_uploaded_csv,
    multi_proportion_test, pairwise_proportion_test, multi_comparison_anova, pairwise_tukey_hsd
)
import plotly.graph_objects as go
import io

# Set page configuration
st.set_page_config(
    page_title="A/B Test Significance Calculator",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("A/B Test Statistical Significance Calculator")
st.markdown("""
This app helps you determine if your A/B test results are statistically significant.
Enter the data from your experiment below to analyze the results.
""")

# Sidebar with information about statistical methods
with st.sidebar:
    st.header("About Statistical Significance")
    st.markdown("""
    ### What is Statistical Significance?
    Statistical significance helps determine if the difference between variant A and B is due to chance or a real effect.
    
    ### Methods Used
    This calculator uses:
    - **Z-test for proportions**: Compares conversion rates between two groups
    - **P-value**: Probability that the observed difference could occur by random chance
    - **Confidence Intervals**: Range of plausible values for the true conversion rate
    
    ### Interpreting Results
    - **p < 0.05**: Conventionally considered statistically significant (95% confidence)
    - **p < 0.01**: Strong statistical significance (99% confidence)
    - **Confidence Interval**: If the intervals don't overlap, the difference is likely significant
    """)
    
    st.header("Sample Size Calculator")
    st.markdown("Determine the sample size needed for your next test:")
    
    baseline_cr = st.number_input("Baseline conversion rate (%)", 
                                  min_value=0.1, max_value=100.0, value=5.0, step=0.1) / 100
    mde = st.number_input("Minimum detectable effect (%)", 
                          min_value=1.0, max_value=100.0, value=10.0, step=0.5) / 100
    confidence = st.slider("Confidence level", 
                          min_value=80, max_value=99, value=95, step=1)
    power = st.slider("Statistical power", 
                      min_value=80, max_value=99, value=80, step=1)
    
    if st.button("Calculate Required Sample Size"):
        sample_size = calculate_sample_size(baseline_cr, mde, confidence/100, power/100)
        st.success(f"Required sample size per variant: **{sample_size:,}**")
        st.markdown(f"Total sample size needed: **{sample_size*2:,}**")

# Main form for A/B test data entry
st.header("Enter Your A/B Test Data")

# Add data import/export section
st.markdown("### Data Import/Export")
data_tabs = st.tabs(["Import Data", "Export Data"])

with data_tabs[0]:
    st.markdown("Upload your CSV file with A/B test data to automatically fill the form below.")
    st.markdown("The CSV should have columns: `control_visitors`, `control_conversions`, `variant_visitors`, `variant_conversions`")
    st.markdown("Alternatively, it can have columns: `date`, `group` (control or variant), `visitors`, `conversions`")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        test_data = parse_uploaded_csv(uploaded_file)
        if test_data:
            st.success("Data loaded successfully!")
            # We'll use these values later in the form
            control_visitors_input = test_data['control_visitors']
            control_conversions_input = test_data['control_conversions']
            variant_visitors_input = test_data['variant_visitors']
            variant_conversions_input = test_data['variant_conversions']
        else:
            st.error("Could not parse the CSV file. Make sure it has the correct format.")
            control_visitors_input = 1000
            control_conversions_input = 100
            variant_visitors_input = 1000
            variant_conversions_input = 120
    else:
        # Default values
        control_visitors_input = 1000
        control_conversions_input = 100
        variant_visitors_input = 1000
        variant_conversions_input = 120

with data_tabs[1]:
    st.markdown("Export your test data and results to a CSV file.")
    
    if 'results_generated' not in st.session_state:
        st.session_state.results_generated = False
        
    if not st.session_state.results_generated:
        st.info("Results will be available for export after you calculate significance.")
    else:
        # Create DataFrame with all the test data and results
        export_data = pd.DataFrame({
            'Metric': ['Visitors', 'Conversions', 'Conversion Rate', 'P-value', 'Significant', 'Confidence'],
            'Control': [
                st.session_state.control_visitors,
                st.session_state.control_conversions,
                f"{st.session_state.control_rate:.4f}",
                f"{st.session_state.p_value:.4f}",
                "Yes" if st.session_state.is_significant else "No",
                f"{st.session_state.confidence_level:.1f}%"
            ],
            'Variant': [
                st.session_state.variant_visitors,
                st.session_state.variant_conversions,
                f"{st.session_state.variant_rate:.4f}",
                "",
                "",
                ""
            ]
        })
        
        st.dataframe(export_data)
        st.markdown(generate_download_link(export_data, "ab_test_results.csv", "Download Results as CSV"), unsafe_allow_html=True)

# Test type selection
st.markdown("### Test Type")
test_type = st.radio(
    "Select Statistical Test",
    ["Conversion Rate (Z-test)", "Conversion Rate (Chi-square)", "Continuous Metric (T-test)", "Multiple Variants (A/B/C/n)"],
    horizontal=True
)

if test_type == "Conversion Rate (Z-test)" or test_type == "Conversion Rate (Chi-square)":
    # Original conversion rate test form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Control Group (A)")
        control_visitors = st.number_input("Visitors (A)", min_value=1, value=control_visitors_input, help="Total number of users in control group")
        control_conversions = st.number_input("Conversions (A)", min_value=0, value=control_conversions_input, help="Number of conversions in control group")
        
        # Add validation for control conversions
        if control_conversions > control_visitors:
            st.error("Conversions cannot exceed visitors!")
            control_conversions = control_visitors
    
    with col2:
        st.subheader("Variant Group (B)")
        variant_visitors = st.number_input("Visitors (B)", min_value=1, value=variant_visitors_input, help="Total number of users in variant group")
        variant_conversions = st.number_input("Conversions (B)", min_value=0, value=variant_conversions_input, help="Number of conversions in variant group")
        
        # Add validation for variant conversions
        if variant_conversions > variant_visitors:
            st.error("Conversions cannot exceed visitors!")
            variant_conversions = variant_visitors
    
    # Calculate conversion rates
    if control_visitors > 0:
        control_rate = control_conversions / control_visitors
    else:
        control_rate = 0
    
    if variant_visitors > 0:
        variant_rate = variant_conversions / variant_visitors
    else:
        variant_rate = 0
    
    # Display conversion rates
    st.markdown("### Conversion Rates")
    rate_cols = st.columns(3)
    with rate_cols[0]:
        st.metric("Control Conversion Rate", f"{control_rate:.2%}")
    with rate_cols[1]:
        st.metric("Variant Conversion Rate", f"{variant_rate:.2%}", 
                delta=f"{(variant_rate - control_rate):.2%}", delta_color="normal")
    with rate_cols[2]:
        relative_change = ((variant_rate - control_rate) / control_rate) if control_rate > 0 else 0
        st.metric("Relative Improvement", f"{relative_change:.2%}")

elif test_type == "Continuous Metric (T-test)":
    # Continuous metric form (T-test)
    st.markdown("Enter mean, standard deviation, and sample size for continuous metrics like revenue or time spent.")
    
    cont_col1, cont_col2 = st.columns(2)
    
    with cont_col1:
        st.subheader("Control Group (A)")
        control_mean = st.number_input("Mean (A)", value=100.0, step=0.1, help="Average value in control group")
        control_std = st.number_input("Standard Deviation (A)", min_value=0.1, value=20.0, step=0.1, help="Standard deviation in control group")
        control_size = st.number_input("Sample Size (A)", min_value=2, value=1000, help="Number of samples in control group")
    
    with cont_col2:
        st.subheader("Variant Group (B)")
        variant_mean = st.number_input("Mean (B)", value=105.0, step=0.1, help="Average value in variant group")
        variant_std = st.number_input("Standard Deviation (B)", min_value=0.1, value=20.0, step=0.1, help="Standard deviation in variant group")
        variant_size = st.number_input("Sample Size (B)", min_value=2, value=1000, help="Number of samples in variant group")
    
    # Display metrics
    st.markdown("### Metrics")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Control Mean", f"{control_mean:.2f}")
    with metric_cols[1]:
        st.metric("Variant Mean", f"{variant_mean:.2f}", 
                delta=f"{(variant_mean - control_mean):.2f}", delta_color="normal")
    with metric_cols[2]:
        relative_change = ((variant_mean - control_mean) / control_mean) if control_mean != 0 else 0
        st.metric("Relative Improvement", f"{relative_change:.2%}")
        
elif test_type == "Multiple Variants (A/B/C/n)":
    # Multiple variants testing (A/B/C/n)
    st.markdown("### Compare Multiple Variants")
    metric_type = st.radio(
        "Select Metric Type",
        ["Conversion Rate", "Continuous Metric (Mean)"],
        horizontal=True
    )
    
    # Number of variants selection
    num_variants = st.number_input("Number of Variants", min_value=2, max_value=10, value=3, 
                                  help="Total number of variants to compare (including control)")
    
    if metric_type == "Conversion Rate":
        # Create a form for conversion rate data
        st.markdown("### Enter Conversion Data")
        
        conversion_data = []
        labels = []
        
        for i in range(num_variants):
            variant_letter = chr(65 + i)
            variant_name = "Control" if i == 0 else f"Variant {variant_letter}"
            labels.append(variant_name)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(variant_name)
                visitors = st.number_input(f"Visitors ({variant_letter})", min_value=1, value=1000, 
                                         key=f"visitors_{i}")
            
            with col2:
                st.write(" ")  # Spacer for alignment
                conversions = st.number_input(f"Conversions ({variant_letter})", min_value=0, value=100 + (i * 10),
                                           max_value=visitors, key=f"conversions_{i}")
                
                # Calculate and show conversion rate
                rate = conversions / visitors
                st.metric(f"Conversion Rate ({variant_letter})", f"{rate:.2%}")
            
            # Add to our data collection
            conversion_data.append({"visitors": visitors, "conversions": conversions, "rate": rate})
        
        # Display data summary
        st.markdown("### Conversion Rates Summary")
        summary_data = pd.DataFrame({
            'Variant': labels,
            'Visitors': [data["visitors"] for data in conversion_data],
            'Conversions': [data["conversions"] for data in conversion_data],
            'Conversion Rate': [data["rate"] for data in conversion_data]
        })
        
        st.dataframe(summary_data)
        
        # Create a bar chart to visualize the rates
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=[data["rate"] for data in conversion_data],
                text=[f"{data['rate']:.2%}" for data in conversion_data],
                textposition='auto',
                marker_color=['blue'] + ['green'] * (num_variants - 1)
            )
        ])
        
        fig.update_layout(
            title='Conversion Rates Comparison',
            xaxis_title='Variant',
            yaxis_title='Conversion Rate',
            yaxis_tickformat='.2%',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Continuous Metric
        # Create a form for continuous metric data
        st.markdown("### Enter Continuous Metric Data")
        
        continuous_data = []
        labels = []
        
        for i in range(num_variants):
            variant_letter = chr(65 + i)
            variant_name = "Control" if i == 0 else f"Variant {variant_letter}"
            labels.append(variant_name)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(variant_name)
                mean = st.number_input(f"Mean ({variant_letter})", value=100.0 + (i * 5), step=0.1,
                                     key=f"mean_{i}")
                std = st.number_input(f"Standard Deviation ({variant_letter})", min_value=0.1, value=20.0, 
                                    step=0.1, key=f"std_{i}")
            
            with col2:
                st.write(" ")  # Spacer for alignment
                sample_size = st.number_input(f"Sample Size ({variant_letter})", min_value=2, value=1000,
                                           key=f"size_{i}")
                
                # Calculate standard error
                se = std / math.sqrt(sample_size)
                st.metric(f"Standard Error ({variant_letter})", f"{se:.2f}")
            
            # Add to our data collection
            continuous_data.append({
                "mean": mean, 
                "std": std, 
                "sample_size": sample_size,
                "se": se
            })
        
        # Display data summary
        st.markdown("### Metrics Summary")
        summary_data = pd.DataFrame({
            'Variant': labels,
            'Mean': [data["mean"] for data in continuous_data],
            'Standard Deviation': [data["std"] for data in continuous_data],
            'Sample Size': [data["sample_size"] for data in continuous_data],
            'Standard Error': [data["se"] for data in continuous_data]
        })
        
        st.dataframe(summary_data)
        
        # Create a bar chart with error bars to visualize the means
        fig = go.Figure()
        
        # Calculate confidence intervals (95%)
        ci_lower = [data["mean"] - 1.96 * data["se"] for data in continuous_data]
        ci_upper = [data["mean"] + 1.96 * data["se"] for data in continuous_data]
        
        # Add bars with error bars
        fig.add_trace(go.Bar(
            x=labels,
            y=[data["mean"] for data in continuous_data],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[u - m for u, m in zip(ci_upper, [data["mean"] for data in continuous_data])],
                arrayminus=[m - l for l, m in zip(ci_lower, [data["mean"] for data in continuous_data])]
            ),
            text=[f"{data['mean']:.2f}" for data in continuous_data],
            textposition='auto',
            marker_color=['blue'] + ['green'] * (num_variants - 1)
        ))
        
        fig.update_layout(
            title='Mean Values with 95% Confidence Intervals',
            xaxis_title='Variant',
            yaxis_title='Mean Value',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


# Calculate statistical significance when button is clicked
if st.button("Calculate Significance"):
    # Determine which test to run based on the selected test type
    if test_type == "Continuous Metric (T-test)":
        # T-test for continuous metrics
        if control_size < 30 or variant_size < 30:
            st.warning("⚠️ Warning: Sample size is small. Results may not be reliable.")
        
        # Calculate significance using t-test
        p_value, is_significant, t_stat = t_test_for_means(
            control_mean, control_std, control_size,
            variant_mean, variant_std, variant_size
        )
        
        # Display significance results
        st.markdown("## Results Analysis")
        
        # Create columns for the results
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown("### Statistical Metrics")
            st.markdown(f"**P-value:** {p_value:.4f}")
            st.markdown(f"**T-statistic:** {t_stat:.4f}")
            
            if is_significant:
                st.markdown(f"**Confidence:** >95%")
            else:
                confidence_level = (1 - p_value) * 100
                if confidence_level > 0:
                    st.markdown(f"**Confidence:** {confidence_level:.1f}%")
                else:
                    st.markdown(f"**Confidence:** Very low")
        
        with col2:
            # Visual indicator of significance
            if is_significant:
                st.success("SIGNIFICANT")
            else:
                st.warning("NOT SIGNIFICANT")
                
        with col3:
            # Calculate standard errors
            control_se = control_std / math.sqrt(control_size)
            variant_se = variant_std / math.sqrt(variant_size)
            
            # Calculate approximate 95% confidence intervals
            z = 1.96  # z-score for 95% confidence
            control_ci_lower = control_mean - z * control_se
            control_ci_upper = control_mean + z * control_se
            variant_ci_lower = variant_mean - z * variant_se
            variant_ci_upper = variant_mean + z * variant_se
            
            st.markdown("### Confidence Intervals (95%)")
            st.markdown(f"**Control:** [{control_ci_lower:.2f} - {control_ci_upper:.2f}]")
            st.markdown(f"**Variant:** [{variant_ci_lower:.2f} - {variant_ci_upper:.2f}]")
            
            # Check if confidence intervals overlap
            if control_ci_upper < variant_ci_lower or variant_ci_upper < control_ci_lower:
                st.markdown("**Confidence intervals do not overlap**")
            else:
                st.markdown("**Confidence intervals overlap**")
        
        # Visualization of means with confidence intervals
        st.markdown("### Visualization")
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'Group': ['Control', 'Variant'],
            'Mean': [control_mean, variant_mean],
            'Lower CI': [control_ci_lower, variant_ci_lower],
            'Upper CI': [control_ci_upper, variant_ci_upper]
        })
        
        # Create a bar chart with error bars using Plotly
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=df['Group'],
            y=df['Mean'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=df['Upper CI'] - df['Mean'],
                arrayminus=df['Mean'] - df['Lower CI']
            ),
            marker_color=['blue', 'green'],
            width=0.5
        ))
        
        # Update layout
        fig.update_layout(
            title='Mean Values with 95% Confidence Intervals',
            xaxis_title='Group',
            yaxis_title='Mean Value',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Save results to session state for export
        st.session_state.results_generated = True
        st.session_state.control_visitors = control_size
        st.session_state.control_conversions = 0  # Not applicable for t-test
        st.session_state.variant_visitors = variant_size
        st.session_state.variant_conversions = 0  # Not applicable for t-test
        st.session_state.control_rate = control_mean
        st.session_state.variant_rate = variant_mean
        st.session_state.p_value = p_value
        st.session_state.is_significant = is_significant
        st.session_state.confidence_level = (1 - p_value) * 100 if not is_significant else 95
        
        # Display interpretation
        st.markdown("## Interpretation")
        
        if is_significant:
            st.success("""
            ### ✅ Statistically Significant Result
            
            The difference between the control and variant means is statistically significant. 
            This means the observed difference is likely not due to random chance.
            """)
            
            if variant_mean > control_mean:
                st.markdown(f"""
                The variant outperformed the control by {(variant_mean - control_mean):.2f} absolute difference
                ({relative_change:.2%} relative improvement).
                
                **Recommendation:** Consider implementing the variant.
                """)
            else:
                st.markdown(f"""
                The variant underperformed compared to the control by {(control_mean - variant_mean):.2f} absolute difference
                ({-relative_change:.2%} relative decrease).
                
                **Recommendation:** Stick with the control or test a new variant.
                """)
        else:
            st.warning("""
            ### ⚠️ Not Statistically Significant
            
            The difference between the control and variant means is not statistically significant.
            This means the observed difference might be due to random chance.
            """)
            
            if p_value < 0.1:
                st.markdown("""
                The result is trending towards significance. You might consider:
                - Continuing the test to collect more data
                - Analyzing specific segments where the effect might be stronger
                """)
            else:
                st.markdown("""
                There's little evidence of a meaningful difference between the variants. You might consider:
                - Ending the test and trying a different approach
                - Checking if there might be implementation issues with your test
                """)
    
    elif test_type == "Multiple Variants (A/B/C/n)":
        if metric_type == "Conversion Rate":
            # Check for valid sample sizes
            if any(data["visitors"] < 30 for data in conversion_data):
                st.warning("⚠️ Warning: Some variants have small sample sizes. Results may not be reliable.")
            
            # Check for zero conversions in all groups
            if all(data["conversions"] == 0 for data in conversion_data):
                st.error("Cannot calculate significance when all groups have zero conversions.")
            else:
                # Prepare data for the chi-square test
                conversions_list = [data["conversions"] for data in conversion_data]
                visitors_list = [data["visitors"] for data in conversion_data]
                
                # Run chi-square test for multiple proportions
                chi2_stat, p_value, is_significant, result_df = multi_proportion_test(
                    conversions_list, visitors_list, labels
                )
                
                # Run pairwise comparisons
                pairwise_df = pairwise_proportion_test(conversions_list, visitors_list, labels)
                
                # Add visual indicators for significance
                pairwise_df['Significant'] = pairwise_df['Significant'].apply(
                    lambda x: '✓' if x else '✗'
                )
                
                # Display results
                st.markdown("## Results Analysis")
                
                # Create columns for the results
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("### Statistical Metrics")
                    st.markdown(f"**Chi-square statistic:** {chi2_stat:.4f}")
                    st.markdown(f"**P-value:** {p_value:.4f}")
                    st.markdown(f"**Degrees of freedom:** {len(labels) - 1}")
                    
                    if is_significant:
                        st.markdown(f"**Confidence:** >95%")
                    else:
                        confidence_level = (1 - p_value) * 100
                        if confidence_level > 0:
                            st.markdown(f"**Confidence:** {confidence_level:.1f}%")
                        else:
                            st.markdown(f"**Confidence:** Very low")
                
                with col2:
                    # Visual indicator of significance
                    if is_significant:
                        st.success("SIGNIFICANT")
                    else:
                        st.warning("NOT SIGNIFICANT")
                
                # Display pairwise comparison results
                st.markdown("### Pairwise Comparisons")
                st.markdown("Results adjusted with Bonferroni correction for multiple comparisons.")
                st.dataframe(pairwise_df)
                
                # Calculate confidence intervals for each variant
                confidence_intervals = []
                for i, data in enumerate(conversion_data):
                    ci = calculate_confidence_interval(data["conversions"], data["visitors"])
                    confidence_intervals.append({
                        "variant": labels[i],
                        "lower": ci[0],
                        "upper": ci[1]
                    })
                
                # Create visualization with confidence intervals
                st.markdown("### Visualization with Confidence Intervals")
                
                # Create a DataFrame for plotting
                ci_df = pd.DataFrame(confidence_intervals)
                
                # Create a bar chart with error bars using Plotly
                fig = go.Figure()
                
                # Add bars
                fig.add_trace(go.Bar(
                    x=[data["variant"] for data in confidence_intervals],
                    y=[data["rate"] for data in conversion_data],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[ci["upper"] - data["rate"] for ci, data in zip(confidence_intervals, conversion_data)],
                        arrayminus=[data["rate"] - ci["lower"] for ci, data in zip(confidence_intervals, conversion_data)]
                    ),
                    marker_color=['blue'] + ['green'] * (len(labels) - 1),
                    width=0.6
                ))
                
                # Update layout
                fig.update_layout(
                    title='Conversion Rates with 95% Confidence Intervals',
                    xaxis_title='Variant',
                    yaxis_title='Conversion Rate',
                    yaxis_tickformat='.2%',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display interpretation
                st.markdown("## Interpretation")
                
                if is_significant:
                    st.success("""
                    ### ✅ Statistically Significant Differences Detected
                    
                    There are statistically significant differences among the variants. This means at least one variant 
                    performs differently than the others, and the observed differences are likely not due to random chance.
                    """)
                    
                    # Find best performer
                    best_idx = np.argmax([data["rate"] for data in conversion_data])
                    best_variant = labels[best_idx]
                    best_rate = conversion_data[best_idx]["rate"]
                    
                    control_idx = 0  # Control is always the first variant
                    control_rate = conversion_data[control_idx]["rate"]
                    
                    if best_idx != control_idx:
                        improvement = (best_rate - control_rate) / control_rate if control_rate > 0 else float('inf')
                        st.markdown(f"""
                        **Best performer:** {best_variant} with {best_rate:.2%} conversion rate
                        ({improvement:.2%} improvement over Control)
                        
                        **Recommendation:** Look at the pairwise comparisons to confirm which specific variants differ 
                        significantly from each other. Consider implementing the best-performing variant.
                        """)
                    else:
                        st.markdown(f"""
                        **Best performer:** The Control with {control_rate:.2%} conversion rate
                        
                        **Recommendation:** Keep the current implementation (Control) since it outperforms 
                        the variants. Consider testing new variants with different approaches.
                        """)
                else:
                    st.warning("""
                    ### ⚠️ No Statistically Significant Differences
                    
                    There are no statistically significant differences among the variants. This means the observed 
                    differences might be due to random chance.
                    """)
                    
                    if p_value < 0.1:
                        st.markdown("""
                        The result is trending towards significance. You might consider:
                        - Continuing the test to collect more data
                        - Analyzing specific segments where differences might be stronger
                        """)
                    else:
                        st.markdown("""
                        There's little evidence of meaningful differences between the variants. You might consider:
                        - Ending the test and trying different approaches
                        - Testing more distinct variations
                        - Checking if there might be implementation issues with your test
                        """)
        
        else:  # Continuous metric
            # Check for valid sample sizes
            if any(data["sample_size"] < 30 for data in continuous_data):
                st.warning("⚠️ Warning: Some variants have small sample sizes. Results may not be reliable.")
            
            # Prepare data for ANOVA test
            anova_data = {}
            for i, data in enumerate(continuous_data):
                # Create a normal distribution for each variant based on mean and std
                np.random.seed(42 + i)  # For reproducibility, different for each variant
                anova_data[labels[i]] = np.random.normal(
                    data["mean"], data["std"], size=min(data["sample_size"], 1000)
                )
            
            # Run ANOVA test
            f_stat, p_value, is_significant = multi_comparison_anova(anova_data)
            
            # Run Tukey's HSD test for pairwise comparisons
            results_df, all_pairs_df = pairwise_tukey_hsd(anova_data, labels)
            
            # Add visual indicators for significance
            all_pairs_df['Significant'] = all_pairs_df['Significant'].apply(
                lambda x: '✓' if x else '✗'
            )
            
            # Display results
            st.markdown("## Results Analysis")
            
            # Create columns for the results
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Statistical Metrics")
                st.markdown(f"**F-statistic:** {f_stat:.4f}")
                st.markdown(f"**P-value:** {p_value:.4f}")
                st.markdown(f"**Degrees of freedom:** {len(labels) - 1}, {sum([len(vals) for vals in anova_data.values()]) - len(labels)}")
                
                if is_significant:
                    st.markdown(f"**Confidence:** >95%")
                else:
                    confidence_level = (1 - p_value) * 100
                    if confidence_level > 0:
                        st.markdown(f"**Confidence:** {confidence_level:.1f}%")
                    else:
                        st.markdown(f"**Confidence:** Very low")
            
            with col2:
                # Visual indicator of significance
                if is_significant:
                    st.success("SIGNIFICANT")
                else:
                    st.warning("NOT SIGNIFICANT")
            
            # Display group statistics
            st.markdown("### Group Statistics")
            st.dataframe(results_df)
            
            # Display pairwise comparison results
            st.markdown("### Pairwise Comparisons (Tukey HSD)")
            st.markdown("Tukey's Honestly Significant Difference test for all pairs of variants.")
            st.dataframe(all_pairs_df)
            
            # Create visualization with confidence intervals
            st.markdown("### Visualization with Confidence Intervals")
            
            # Create a bar chart with error bars using Plotly
            fig = go.Figure()
            
            # Calculate 95% confidence intervals
            ci_lower = [data["mean"] - 1.96 * data["se"] for data in continuous_data]
            ci_upper = [data["mean"] + 1.96 * data["se"] for data in continuous_data]
            
            # Add bars with error bars
            fig.add_trace(go.Bar(
                x=labels,
                y=[data["mean"] for data in continuous_data],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[u - m for u, m in zip(ci_upper, [data["mean"] for data in continuous_data])],
                    arrayminus=[m - l for l, m in zip([data["mean"] for data in continuous_data], ci_lower)]
                ),
                text=[f"{data['mean']:.2f}" for data in continuous_data],
                textposition='auto',
                marker_color=['blue'] + ['green'] * (len(labels) - 1),
                width=0.6
            ))
            
            # Update layout
            fig.update_layout(
                title='Mean Values with 95% Confidence Intervals',
                xaxis_title='Variant',
                yaxis_title='Mean Value',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display interpretation
            st.markdown("## Interpretation")
            
            if is_significant:
                st.success("""
                ### ✅ Statistically Significant Differences Detected
                
                There are statistically significant differences among the variants. This means at least one variant 
                performs differently than the others, and the observed differences are likely not due to random chance.
                """)
                
                # Find best performer
                best_idx = np.argmax([data["mean"] for data in continuous_data])
                best_variant = labels[best_idx]
                best_mean = continuous_data[best_idx]["mean"]
                
                control_idx = 0  # Control is always the first variant
                control_mean = continuous_data[control_idx]["mean"]
                
                if best_idx != control_idx:
                    improvement = (best_mean - control_mean) / control_mean if control_mean != 0 else float('inf')
                    st.markdown(f"""
                    **Best performer:** {best_variant} with mean value of {best_mean:.2f}
                    ({improvement:.2%} improvement over Control)
                    
                    **Recommendation:** Look at the pairwise comparisons to confirm which specific variants differ 
                    significantly from each other. Consider implementing the best-performing variant.
                    """)
                else:
                    st.markdown(f"""
                    **Best performer:** The Control with mean value of {control_mean:.2f}
                    
                    **Recommendation:** Keep the current implementation (Control) since it outperforms 
                    the variants. Consider testing new variants with different approaches.
                    """)
            else:
                st.warning("""
                ### ⚠️ No Statistically Significant Differences
                
                There are no statistically significant differences among the variants. This means the observed 
                differences might be due to random chance.
                """)
                
                if p_value < 0.1:
                    st.markdown("""
                    The result is trending towards significance. You might consider:
                    - Continuing the test to collect more data
                    - Analyzing specific segments where differences might be stronger
                    """)
                else:
                    st.markdown("""
                    There's little evidence of meaningful differences between the variants. You might consider:
                    - Ending the test and trying different approaches
                    - Testing more distinct variations
                    - Checking if there might be implementation issues with your test
                    """)
    
    elif test_type == "Conversion Rate (Chi-square)":
        # Chi-square test for conversion rates
        if control_visitors < 30 or variant_visitors < 30:
            st.warning("⚠️ Warning: Sample size is small. Results may not be reliable.")
        
        # Check for zero conversions
        if control_conversions == 0 and variant_conversions == 0:
            st.error("Cannot calculate significance when both groups have zero conversions.")
        else:
            # Calculate significance using chi-square test
            p_value, is_significant, chi2_stat = chi_square_test(
                control_conversions, control_visitors,
                variant_conversions, variant_visitors
            )
            
            # Calculate confidence intervals
            control_ci = calculate_confidence_interval(control_conversions, control_visitors)
            variant_ci = calculate_confidence_interval(variant_conversions, variant_visitors)
            
            # Display significance results
            st.markdown("## Results Analysis")
            
            # Create columns for the results
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown("### Statistical Metrics")
                st.markdown(f"**P-value:** {p_value:.4f}")
                st.markdown(f"**Chi-square statistic:** {chi2_stat:.4f}")
                
                if is_significant:
                    st.markdown(f"**Confidence:** >95%")
                else:
                    confidence_level = (1 - p_value) * 100
                    if confidence_level > 0:
                        st.markdown(f"**Confidence:** {confidence_level:.1f}%")
                    else:
                        st.markdown(f"**Confidence:** Very low")
            
            with col2:
                # Visual indicator of significance
                if is_significant:
                    st.success("SIGNIFICANT")
                else:
                    st.warning("NOT SIGNIFICANT")
                    
            with col3:
                st.markdown("### Confidence Intervals (95%)")
                st.markdown(f"**Control:** [{control_ci[0]:.2%} - {control_ci[1]:.2%}]")
                st.markdown(f"**Variant:** [{variant_ci[0]:.2%} - {variant_ci[1]:.2%}]")
                
                # Check if confidence intervals overlap
                if control_ci[1] < variant_ci[0] or variant_ci[1] < control_ci[0]:
                    st.markdown("**Confidence intervals do not overlap**")
                else:
                    st.markdown("**Confidence intervals overlap**")
            
            # Visualization of conversion rates with confidence intervals
            st.markdown("### Visualization")
            
            # Create a DataFrame for plotting
            df = pd.DataFrame({
                'Group': ['Control', 'Variant'],
                'Conversion Rate': [control_rate, variant_rate],
                'Lower CI': [control_ci[0], variant_ci[0]],
                'Upper CI': [control_ci[1], variant_ci[1]]
            })
            
            # Create a bar chart with error bars using Plotly
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=df['Group'],
                y=df['Conversion Rate'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=df['Upper CI'] - df['Conversion Rate'],
                    arrayminus=df['Conversion Rate'] - df['Lower CI']
                ),
                marker_color=['blue', 'green'],
                width=0.5
            ))
            
            # Update layout
            fig.update_layout(
                title='Conversion Rates with 95% Confidence Intervals',
                xaxis_title='Group',
                yaxis_title='Conversion Rate',
                yaxis_tickformat='.2%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Save results to session state for export
            st.session_state.results_generated = True
            st.session_state.control_visitors = control_visitors
            st.session_state.control_conversions = control_conversions
            st.session_state.variant_visitors = variant_visitors
            st.session_state.variant_conversions = variant_conversions
            st.session_state.control_rate = control_rate
            st.session_state.variant_rate = variant_rate
            st.session_state.p_value = p_value
            st.session_state.is_significant = is_significant
            st.session_state.confidence_level = (1 - p_value) * 100 if not is_significant else 95
            
            # Display interpretation
            st.markdown("## Interpretation")
            
            if is_significant:
                st.success("""
                ### ✅ Statistically Significant Result
                
                The difference between the control and variant groups is statistically significant. 
                This means the observed difference is likely not due to random chance.
                """)
                
                if variant_rate > control_rate:
                    st.markdown(f"""
                    The variant outperformed the control by {(variant_rate - control_rate):.2%} absolute difference
                    ({relative_change:.2%} relative improvement).
                    
                    **Recommendation:** Consider implementing the variant.
                    """)
                else:
                    st.markdown(f"""
                    The variant underperformed compared to the control by {(control_rate - variant_rate):.2%} absolute difference
                    ({-relative_change:.2%} relative decrease).
                    
                    **Recommendation:** Stick with the control or test a new variant.
                    """)
            else:
                st.warning("""
                ### ⚠️ Not Statistically Significant
                
                The difference between the control and variant groups is not statistically significant.
                This means the observed difference might be due to random chance.
                """)
                
                if p_value < 0.1:
                    st.markdown("""
                    The result is trending towards significance. You might consider:
                    - Continuing the test to collect more data
                    - Analyzing specific segments where the effect might be stronger
                    """)
                else:
                    st.markdown("""
                    There's little evidence of a meaningful difference between the variants. You might consider:
                    - Ending the test and trying a different approach
                    - Checking if there might be implementation issues with your test
                    """)
    
    else:  # Z-test (default)
        if control_visitors < 30 or variant_visitors < 30:
            st.warning("⚠️ Warning: Sample size is small. Results may not be reliable.")
        
        # Check for zero conversions
        if control_conversions == 0 and variant_conversions == 0:
            st.error("Cannot calculate significance when both groups have zero conversions.")
        else:
            # Calculate significance using z-test
            p_value, is_significant, z_score = calculate_significance(
                control_conversions, control_visitors, 
                variant_conversions, variant_visitors
            )
            
            # Calculate confidence intervals
            control_ci = calculate_confidence_interval(control_conversions, control_visitors)
            variant_ci = calculate_confidence_interval(variant_conversions, variant_visitors)
            
            # Display significance results
            st.markdown("## Results Analysis")
            
            # Create columns for the results
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown("### Statistical Metrics")
                st.markdown(f"**P-value:** {p_value:.4f}")
                st.markdown(f"**Z-score:** {z_score:.4f}")
                
                if is_significant:
                    st.markdown(f"**Confidence:** >95%")
                else:
                    confidence_level = (1 - p_value) * 100
                    if confidence_level > 0:
                        st.markdown(f"**Confidence:** {confidence_level:.1f}%")
                    else:
                        st.markdown(f"**Confidence:** Very low")
            
            with col2:
                # Visual indicator of significance
                if is_significant:
                    st.success("SIGNIFICANT")
                else:
                    st.warning("NOT SIGNIFICANT")
                    
            with col3:
                st.markdown("### Confidence Intervals (95%)")
                st.markdown(f"**Control:** [{control_ci[0]:.2%} - {control_ci[1]:.2%}]")
                st.markdown(f"**Variant:** [{variant_ci[0]:.2%} - {variant_ci[1]:.2%}]")
                
                # Check if confidence intervals overlap
                if control_ci[1] < variant_ci[0] or variant_ci[1] < control_ci[0]:
                    st.markdown("**Confidence intervals do not overlap**")
                else:
                    st.markdown("**Confidence intervals overlap**")
            
            # Visualization of conversion rates with confidence intervals
            st.markdown("### Visualization")
            
            # Create a DataFrame for plotting
            df = pd.DataFrame({
                'Group': ['Control', 'Variant'],
                'Conversion Rate': [control_rate, variant_rate],
                'Lower CI': [control_ci[0], variant_ci[0]],
                'Upper CI': [control_ci[1], variant_ci[1]]
            })
            
            # Create a bar chart with error bars using Plotly
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=df['Group'],
                y=df['Conversion Rate'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=df['Upper CI'] - df['Conversion Rate'],
                    arrayminus=df['Conversion Rate'] - df['Lower CI']
                ),
                marker_color=['blue', 'green'],
                width=0.5
            ))
            
            # Update layout
            fig.update_layout(
                title='Conversion Rates with 95% Confidence Intervals',
                xaxis_title='Group',
                yaxis_title='Conversion Rate',
                yaxis_tickformat='.2%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Save results to session state for export
            st.session_state.results_generated = True
            st.session_state.control_visitors = control_visitors
            st.session_state.control_conversions = control_conversions
            st.session_state.variant_visitors = variant_visitors
            st.session_state.variant_conversions = variant_conversions
            st.session_state.control_rate = control_rate
            st.session_state.variant_rate = variant_rate
            st.session_state.p_value = p_value
            st.session_state.is_significant = is_significant
            st.session_state.confidence_level = (1 - p_value) * 100 if not is_significant else 95
            
            # Display interpretation
            st.markdown("## Interpretation")
            
            if is_significant:
                st.success("""
                ### ✅ Statistically Significant Result
                
                The difference between the control and variant groups is statistically significant. 
                This means the observed difference is likely not due to random chance.
                """)
                
                if variant_rate > control_rate:
                    st.markdown(f"""
                    The variant outperformed the control by {(variant_rate - control_rate):.2%} absolute difference
                    ({relative_change:.2%} relative improvement).
                    
                    **Recommendation:** Consider implementing the variant.
                    """)
                else:
                    st.markdown(f"""
                    The variant underperformed compared to the control by {(control_rate - variant_rate):.2%} absolute difference
                    ({-relative_change:.2%} relative decrease).
                    
                    **Recommendation:** Stick with the control or test a new variant.
                    """)
            else:
                st.warning("""
                ### ⚠️ Not Statistically Significant
                
                The difference between the control and variant groups is not statistically significant.
                This means the observed difference might be due to random chance.
                """)
                
                if p_value < 0.1:
                    st.markdown("""
                    The result is trending towards significance. You might consider:
                    - Continuing the test to collect more data
                    - Analyzing specific segments where the effect might be stronger
                    """)
                else:
                    st.markdown("""
                    There's little evidence of a meaningful difference between the variants. You might consider:
                    - Ending the test and trying a different approach
                    - Checking if there might be implementation issues with your test
                    """)
    
    # Additional considerations for all test types
    st.markdown("""
    ### Additional Considerations
    
    - **Practical Significance:** Even with statistical significance, consider if the improvement is practically meaningful
    - **Test Duration:** Ensure your test ran for at least 1-2 full business cycles
    - **Segment Analysis:** Consider analyzing specific user segments for deeper insights
    - **Multiple Testing:** If you're running multiple tests, adjust your significance threshold
    """)

# Advanced visualizations section
from advanced_visualizations import (
    generate_time_series_simulation, create_distribution_comparison,
    create_distribution_comparison_categorical, create_power_analysis_curve
)

st.markdown("---")
st.header("Advanced Visualizations")
st.markdown("""
These visualizations help you better understand your test results and plan future experiments.
Choose from the options below to explore different aspects of your A/B test.
""")

# Create tabs for different visualization types
viz_tabs = st.tabs(["Time Series View", "Distribution Comparison", "Power Analysis"])

with viz_tabs[0]:
    st.subheader("Time Series View")
    st.markdown("""
    This simulation shows how statistical significance typically evolves over time as more data is collected.
    It can help you understand how long to run your test before reaching a conclusion.
    """)
    
    # Input parameters for time series simulation
    ts_cols = st.columns(3)
    with ts_cols[0]:
        ts_control_rate = st.number_input("Control Conversion Rate", 
                                         min_value=0.001, max_value=1.0, value=0.1, step=0.001, 
                                         format="%.3f", key="ts_control_rate")
    with ts_cols[1]:
        ts_variant_rate = st.number_input("Variant Conversion Rate", 
                                         min_value=0.001, max_value=1.0, value=0.12, step=0.001, 
                                         format="%.3f", key="ts_variant_rate")
    with ts_cols[2]:
        ts_visitors_per_day = st.number_input("Daily Visitors Per Variant", 
                                             min_value=10, max_value=10000, value=500, step=10, 
                                             key="ts_visitors")
    
    ts_cols2 = st.columns(2)
    with ts_cols2[0]:
        ts_days = st.slider("Test Duration (Days)", 
                          min_value=7, max_value=60, value=21, step=1, 
                          key="ts_days")
    with ts_cols2[1]:
        ts_confidence = st.slider("Confidence Level", 
                                min_value=0.8, max_value=0.99, value=0.95, step=0.01, 
                                format="%.0f%%", key="ts_confidence")
    
    if st.button("Generate Time Series Simulation", key="btn_ts_sim"):
        # Run the simulation
        ts_results = generate_time_series_simulation(
            control_rate=ts_control_rate,
            variant_rate=ts_variant_rate,
            visitors_per_day=ts_visitors_per_day,
            days=ts_days,
            confidence_level=ts_confidence
        )
        
        # Display the plot
        st.plotly_chart(ts_results['fig'], use_container_width=True)
        
        # Show when significance was reached (if it was)
        if ts_results['reached_significance']:
            # Get the first date where significance was reached
            first_sig_day = ts_results['df'][ts_results['df']['significant']].iloc[0]
            st.success(f"Statistical significance was reached on day {first_sig_day.name + 1} with a p-value of {first_sig_day['p_value']:.4f}")
            
            # Calculate visitors needed
            visitors_needed = first_sig_day['control_visitors'] + first_sig_day['variant_visitors']
            st.info(f"Total visitors required: {visitors_needed:,}")
        else:
            st.warning("Statistical significance was not reached within the simulated time period. Consider running a longer test or expecting a larger effect.")

with viz_tabs[1]:
    st.subheader("Distribution Comparison")
    st.markdown("""
    This visualization shows the distribution of values for the control and variant groups, 
    helping you understand the overlap between them and the effect size.
    """)
    
    # Choose distribution type
    dist_type = st.radio(
        "Select Distribution Type",
        ["Conversion Rate (Binomial)", "Continuous Metric (Normal)"],
        horizontal=True,
        key="dist_type"
    )
    
    if dist_type == "Continuous Metric (Normal)":
        # Input for continuous metric distributions
        dist_cols = st.columns(2)
        
        with dist_cols[0]:
            st.subheader("Control Group")
            dist_control_mean = st.number_input("Control Mean", 
                                              value=100.0, step=1.0, key="dist_control_mean")
            dist_control_std = st.number_input("Control Standard Deviation", 
                                             min_value=0.1, value=20.0, step=1.0, key="dist_control_std")
        
        with dist_cols[1]:
            st.subheader("Variant Group")
            dist_variant_mean = st.number_input("Variant Mean", 
                                              value=110.0, step=1.0, key="dist_variant_mean")
            dist_variant_std = st.number_input("Variant Standard Deviation", 
                                             min_value=0.1, value=20.0, step=1.0, key="dist_variant_std")
        
        if st.button("Generate Distribution Comparison", key="btn_dist_cont"):
            # Create the plot
            dist_fig = create_distribution_comparison(
                control_mean=dist_control_mean,
                control_std=dist_control_std,
                variant_mean=dist_variant_mean,
                variant_std=dist_variant_std
            )
            
            # Display the plot
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Calculate t-test statistics
            t_stat, p_value, df = stats.ttest_ind_from_stats(
                dist_control_mean, dist_control_std, 1000,  # Assuming n=1000 for significance calculation
                dist_variant_mean, dist_variant_std, 1000,
                equal_var=False  # Using Welch's t-test
            )
            
            # Display statistics
            st.markdown(f"**t-statistic:** {t_stat:.4f}")
            st.markdown(f"**p-value:** {p_value:.4f}")
            
            # Significance statement
            if p_value < 0.05:
                st.success("The difference between means is statistically significant (p < 0.05).")
            else:
                st.warning("The difference between means is not statistically significant (p ≥ 0.05).")
            
            # Effect size
            effect_size = abs(dist_control_mean - dist_variant_mean) / ((dist_control_std + dist_variant_std) / 2)
            
            if effect_size < 0.2:
                effect_interpretation = "small"
            elif effect_size < 0.5:
                effect_interpretation = "medium"
            elif effect_size < 0.8:
                effect_interpretation = "large"
            else:
                effect_interpretation = "very large"
            
            st.info(f"Effect size (Cohen's d): {effect_size:.2f} ({effect_interpretation})")
    
    else:  # Conversion Rate distribution
        # Input for conversion rate distributions
        cr_cols = st.columns(2)
        
        with cr_cols[0]:
            st.subheader("Control Group")
            cr_control_rate = st.number_input("Control Conversion Rate", 
                                            min_value=0.001, max_value=1.0, value=0.1, 
                                            step=0.001, format="%.3f", key="cr_control_rate")
            cr_control_size = st.number_input("Control Sample Size", 
                                            min_value=100, value=1000, step=100, key="cr_control_size")
        
        with cr_cols[1]:
            st.subheader("Variant Group")
            cr_variant_rate = st.number_input("Variant Conversion Rate", 
                                            min_value=0.001, max_value=1.0, value=0.12, 
                                            step=0.001, format="%.3f", key="cr_variant_rate")
            cr_variant_size = st.number_input("Variant Sample Size", 
                                            min_value=100, value=1000, step=100, key="cr_variant_size")
        
        if st.button("Generate Distribution Comparison", key="btn_dist_cr"):
            # Create the plot
            cr_fig = create_distribution_comparison_categorical(
                control_rate=cr_control_rate,
                variant_rate=cr_variant_rate,
                control_size=cr_control_size,
                variant_size=cr_variant_size
            )
            
            # Display the plot
            st.plotly_chart(cr_fig, use_container_width=True)
            
            # Calculate Z-test statistics
            # Pooled proportion under null hypothesis
            p_pooled = (cr_control_rate * cr_control_size + cr_variant_rate * cr_variant_size) / (cr_control_size + cr_variant_size)
            
            # Standard error
            se = math.sqrt(p_pooled * (1 - p_pooled) * (1/cr_control_size + 1/cr_variant_size))
            
            # Z-score
            z_score = (cr_variant_rate - cr_control_rate) / se if se > 0 else 0
            
            # P-value (two-tailed)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            # Display statistics
            st.markdown(f"**Z-score:** {z_score:.4f}")
            st.markdown(f"**p-value:** {p_value:.4f}")
            
            # Significance statement
            if p_value < 0.05:
                st.success("The difference between conversion rates is statistically significant (p < 0.05).")
            else:
                st.warning("The difference between conversion rates is not statistically significant (p ≥ 0.05).")
            
            # Relative improvement
            relative_diff = (cr_variant_rate - cr_control_rate) / cr_control_rate
            st.info(f"Relative improvement: {relative_diff:.2%}")

with viz_tabs[2]:
    st.subheader("Power Analysis")
    st.markdown("""
    Power analysis helps you determine the sample size needed to detect a specific effect, or
    understand the minimum effect you can detect with your current sample size.
    """)
    
    # Input parameters for power analysis
    power_cols = st.columns(3)
    with power_cols[0]:
        power_baseline = st.number_input("Baseline Conversion Rate", 
                                        min_value=0.001, max_value=1.0, value=0.1, 
                                        step=0.001, format="%.3f", key="power_baseline")
    with power_cols[1]:
        power_confidence = st.slider("Confidence Level", 
                                   min_value=0.8, max_value=0.99, value=0.95, step=0.01, 
                                   format="%.0f%%", key="power_confidence")
    with power_cols[2]:
        power_level = st.slider("Statistical Power", 
                              min_value=0.7, max_value=0.99, value=0.8, step=0.01, 
                              format="%.0f%%", key="power_level")
    
    if st.button("Generate Power Analysis", key="btn_power"):
        # Create power analysis curves
        power_results = create_power_analysis_curve(
            baseline_rate=power_baseline,
            confidence=power_confidence,
            power=power_level
        )
        
        # Display sample size curve
        st.subheader("Sample Size Requirements")
        st.markdown("""
        This chart shows the required sample size per variant to detect different minimum detectable effects (MDEs).
        As the MDE gets smaller, the required sample size increases exponentially.
        """)
        st.plotly_chart(power_results['sample_size_curve'], use_container_width=True)
        
        # Display power curves
        st.subheader("Power Curves")
        st.markdown("""
        These curves show the statistical power for different effect sizes and sample sizes.
        The horizontal line represents the target power level you specified.
        """)
        st.plotly_chart(power_results['power_curves'], use_container_width=True)
        
        # Example calculations
        st.subheader("Sample Size Estimates")
        
        mde_examples = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20% relative improvements
        
        # Create table
        mde_data = []
        for mde in mde_examples:
            sample_size = calculate_sample_size_for_proportion(power_baseline, mde, power_confidence, power_level)
            absolute_effect = power_baseline * mde
            new_rate = power_baseline * (1 + mde)
            
            mde_data.append({
                "MDE (Relative)": f"{mde:.0%}",
                "Effect (Absolute)": f"{absolute_effect:.3f}",
                "New Rate": f"{new_rate:.3f}",
                "Sample Size (per variant)": f"{sample_size:,}",
                "Total Sample": f"{sample_size*2:,}"
            })
        
        mde_df = pd.DataFrame(mde_data)
        st.table(mde_df)

# Footer with additional info
st.markdown("---")

# Update footer based on selected test type
if test_type == "Continuous Metric (T-test)":
    st.markdown("""
    **Note:** This calculator uses Welch's t-test for comparing means of continuous metrics,
    which is appropriate for revenue, time spent, or other continuous variables. 
    It doesn't assume equal variances between groups.
    """)
elif test_type == "Conversion Rate (Chi-square)":
    st.markdown("""
    **Note:** This calculator uses a Chi-square test for comparing proportions,
    which is appropriate for most A/B testing scenarios with binary (conversion/no-conversion) outcomes.
    """)
else:  # Z-test
    st.markdown("""
    **Note:** This calculator uses a two-tailed Z-test for proportions to calculate statistical significance, 
    which is appropriate for most A/B testing scenarios with binary (conversion/no-conversion) outcomes.
    """)
