import streamlit as st
import numpy as np
import pandas as pd
from utils import calculate_significance, calculate_confidence_interval, calculate_sample_size
import plotly.graph_objects as go

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

col1, col2 = st.columns(2)

with col1:
    st.subheader("Control Group (A)")
    control_visitors = st.number_input("Visitors (A)", min_value=1, value=1000, help="Total number of users in control group")
    control_conversions = st.number_input("Conversions (A)", min_value=0, value=100, help="Number of conversions in control group")
    
    # Add validation for control conversions
    if control_conversions > control_visitors:
        st.error("Conversions cannot exceed visitors!")
        control_conversions = control_visitors

with col2:
    st.subheader("Variant Group (B)")
    variant_visitors = st.number_input("Visitors (B)", min_value=1, value=1000, help="Total number of users in variant group")
    variant_conversions = st.number_input("Conversions (B)", min_value=0, value=120, help="Number of conversions in variant group")
    
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
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Control Conversion Rate", f"{control_rate:.2%}")
with col2:
    st.metric("Variant Conversion Rate", f"{variant_rate:.2%}", 
             delta=f"{(variant_rate - control_rate):.2%}", delta_color="normal")
with col3:
    relative_change = ((variant_rate - control_rate) / control_rate) if control_rate > 0 else 0
    st.metric("Relative Improvement", f"{relative_change:.2%}")

# Calculate statistical significance when button is clicked
if st.button("Calculate Significance"):
    if control_visitors < 30 or variant_visitors < 30:
        st.warning("⚠️ Warning: Sample size is small. Results may not be reliable.")
    
    # Check for zero conversions
    if control_conversions == 0 and variant_conversions == 0:
        st.error("Cannot calculate significance when both groups have zero conversions.")
    else:
        # Calculate significance
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
        
        # Additional considerations
        st.markdown("""
        ### Additional Considerations
        
        - **Practical Significance:** Even with statistical significance, consider if the improvement is practically meaningful
        - **Test Duration:** Ensure your test ran for at least 1-2 full business cycles
        - **Segment Analysis:** Consider analyzing specific user segments for deeper insights
        - **Multiple Testing:** If you're running multiple tests, adjust your significance threshold
        """)

# Footer with additional info
st.markdown("---")
st.markdown("""
**Note:** This calculator uses a two-tailed Z-test for proportions to calculate statistical significance, 
which is appropriate for most A/B testing scenarios with binary (conversion/no-conversion) outcomes.
""")
