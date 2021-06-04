import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import src.variables as v
import src.training as tr
import src.markdown as md
import src.examples as ex

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="ECG - Anomaly Detection",
                       page_icon="assets/ecg_logo.png",
                       layout='wide',
                       initial_sidebar_state='expanded')

    # GIF
    col1, col2, col3 = st.beta_columns([3,.5,3])
    col2.markdown(md.image, unsafe_allow_html=True)

    col1, title_col, col3 = st.beta_columns([1.2,2,1.2])
    title_col.title('Anomaly Detection in ECG Data')

    # Dataset
    st.markdown("## The ECG Dataset")
    dataset_expander = st.beta_expander("Read more")
    with dataset_expander:
        st.write(md.dataset)

    # Model
    st.markdown("## The Model")
    model_expander = st.beta_expander("Read more")
    with model_expander:
        st.write(md.model)
        col1, col2, col3 = st.beta_columns([1,2,1])
        col2.image('assets/autoencoder.png', caption='Autoencoder Architecture')

    # Example ECGs
    col1, col2, col3, col4, col5 = st.beta_columns([0.1, 1, 0.2, 1, 0.1])
    # Normal
    with col2:
        st.subheader('Normal Data vs. Reconstructed Data')
        plt.plot(tr.normal_test_data[1], '#7193ff', label='Normal')
        plt.plot(v.decoder_out[1], '#ff6666', label='Reconstructed')
        plt.legend(loc='upper right')
        st.pyplot()

        explanation = st.beta_expander("Explanation")
        with explanation:
            st.write(md.normal_explain)
    # Anamoly
    with col4:
        st.subheader('Anomaly Data vs. Reconstructed Data')
        plt.plot(tr.anomaly_test_data[1], '#7193ff', label='Anomaly')
        plt.plot(v.decoder_out_a[1], '#ff6666', label='Reconstructed')
        plt.legend(loc='upper right')
        st.pyplot()

        explanation = st.beta_expander("Explanation")
        with explanation:
            st.write(md.anomaly_explain)

    st.markdown("## Predicting Normal vs. Anomaly ECG Data")
    st.write(md.predictions_text)
    with st.beta_expander("Threshold Calculation"):
        with st.echo():
            threshold = np.mean(v.reconstruction_loss) + 2*np.std(v.reconstruction_loss)
            st.write(threshold)
        
    col1, col2, col3, col4 = st.beta_columns([1,2,2.5,1])
    with col2:
        st.write(md.threshold)
    with col3:
        # Threshold Graph
        plt.hist(v.reconstruction_loss.numpy(), bins=50, color='#559e83', label='Normal')
        plt.hist(v.reconstruction_loss_a.numpy(), bins=50, color='#c79dd7', label='Anomaly')
        plt.axvline(threshold, color='#ff9a00', linewidth=3, linestyle='dashdot', label='{:0.3f}'.format(threshold))
        plt.legend(loc='upper right')
        st.pyplot()

    st.markdown("## Model Evaluation")
    dataframe = pd.DataFrame(v.eval_data, columns=['Normal ECG Data', 'Anomaly ECG Data'])
    dataframe = dataframe.rename(index={0: "Predicted Count", 1: "Actual Count", 2:'Accuracy Score'})
    st.markdown(md.evaluation)
    st.write(dataframe)

    st.markdown("## Try it out! 👇🏼")
    st.markdown("Can you guess which ECG is normal and which is not?")

    st.subheader('Example 1')
    with st.beta_expander('Open'):
        ex.example_1()

    st.subheader('Example 2')
    with st.beta_expander('Open'):
        ex.example_2()
    
    st.subheader('Example 3')
    with st.beta_expander('Open'):
        ex.example_3()


if __name__ == "__main__":
    main()