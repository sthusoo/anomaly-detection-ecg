import streamlit as st

import matplotlib.pyplot as plt

import src.variables as v
import src.training as tr


def example_1():
    acol1, acol2, acol3, acol4, acol5 = st.beta_columns([2, 0.2, 1.5, 0.3, 1])
    with acol1:
        plt.plot(tr.normal_test_data[5], '#7193ff')
        st.pyplot()

    with acol3:
        options = ['Select one', 'Normal', 'Anomaly', 'I don\'t know']
        guess_1 = st.selectbox('Take a Guess', options, index=0, key="1")
        if st.button('Submit Guess', key="1"):
            submitted_1 = True
        else:
            submitted_1 = False

        if submitted_1:
            if guess_1 == options[0]:
                return
            else:
                st.markdown('#### Model Prediction')
                if v.is_normal_1:
                    st.write('Normal ECG')
                else:
                    st.write('Anomaly ECG')

            st.markdown('#### Ground Truth')
            st.markdown('Normal ECG')
    
    with acol5:
        if not submitted_1:
            return
        else: 
            if v.is_normal_1 and guess_1 == "Normal":
                st.subheader('You are correct! 🥳')
                st.balloons()
            else:
                st.subheader('Wrong Answer 😅')

def example_2():
    bcol1, bcol2, bcol3, bcol4, bcol5 = st.beta_columns([2, 0.2, 1.5, 0.3, 1])
    with bcol1:
        plt.plot(tr.anomaly_test_data[20], '#7193ff')
        st.pyplot()

    with bcol3:
        options_2 = ['Select one', 'Normal', 'Anomaly', 'I don\'t know']
        guess_2 = st.selectbox('Take a Guess', options_2, index=0, key="2")
        if st.button('Submit Guess', key="2"):
            submitted_2 = True
        else:
            submitted_2 = False

        if submitted_2:
            if guess_2 == options_2[0]:
                return
            else:
                st.markdown('#### Model Prediction')
                if v.is_normal_2:
                    st.write('Normal ECG')
                else:
                    st.write('Anomaly ECG')

            st.markdown('#### Ground Truth')
            st.markdown('Anomaly ECG')
    
    with bcol5:
        if not submitted_2:
            return
        else: 
            if not v.is_normal_2 and guess_2 == "Anomaly":
                st.subheader('You are correct! 🥳')
                st.balloons()
            else:
                st.subheader('Wrong Answer 😅')

def example_3():
    ccol1, ccol2, ccol3, ccol4, ccol5 = st.beta_columns([2, 0.2, 1.5, 0.3, 1])
    with ccol1:
        plt.plot(tr.anomaly_test_data[20], '#7193ff')
        st.pyplot()

    with ccol3:
        options_3 = ['Select one', 'Normal', 'Anomaly', 'I don\'t know']
        guess_3 = st.selectbox('Take a Guess', options_3, index=0, key="3")
        if st.button('Submit Guess', key="3"):
            submitted_3 = True
        else:
            submitted_3 = False

        if submitted_3:
            if guess_3 == options_3[0]:
                return
            else:
                st.markdown('#### Model Prediction')
                if v.is_normal_3:
                    st.write('Normal ECG')
                else:
                    st.write('Anomaly ECG')

            st.markdown('#### Ground Truth')
            st.markdown('Anomaly ECG')
    
    with ccol5:
        if not submitted_3:
            return
        else: 
            if not v.is_normal_3 and guess_3 == "Anomaly":
                st.subheader('You are correct! 🥳')
                st.balloons()
            else:
                st.subheader('Wrong Answer 😅')