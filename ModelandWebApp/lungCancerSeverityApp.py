import streamlit as st 
import numpy as np
import pandas as pd 
import joblib
##Introduction 
st.sidebar.title("Explore the options")
st.sidebar.markdown("<br>", unsafe_allow_html=True) ## To create the gaps between
page = st.sidebar.selectbox("Features:", ["🏠 Home", " 📊 Prediction", " ℹ️ About the Model"," 👨‍💻 Developer"]) ## Facilites 

if page == "🏠 Home": ## This is the home page 
    ##This is to be displayed in the sidebar
    st.sidebar.markdown("""
    ---  
    ### ⚠️ Disclaimer

    This application was developed as part of a data science project and is based on a relatively small and synthetic dataset. While the model demonstrates how lifestyle and clinical features can be used to predict lung cancer severity levels, it is **not trained on real clinical-scale data** and should **not be used for any actual medical decision-making**.

    Please explore and learn from this tool, but do not interpret its output as medical advice or a diagnostic reference. Always consult a qualified healthcare professional for real-world health concerns.
    """)

    ## This is to be displayed in the main screen 
    st.title("🎗️ Lung Cancer Severity Checker")
    st.text("")
    st.text("")
    st.header("""
    This application uses real-world clinical and lifestyle data to predict the severity level of lung cancer in patients. Based on factors such as age, smoking habits, alcohol use, diet, and reported symptoms, the model evaluates the likelihood of the condition being categorized as Low, Medium, or High severity.""")
    st.text("")
    st.text("")
    st.subheader("To explore and predict please go to 📊 Prediction in the Navigation at the top right ")


##This is for the prediction page
elif page == " 📊 Prediction":
    #This is for the markdown text
    st.sidebar.markdown("""
    ---  
    ### ⚠️ Disclaimer

    This application was developed as part of a data science project and is based on a relatively small and synthetic dataset. While the model demonstrates how lifestyle and clinical features can be used to predict lung cancer severity levels, it is **not trained on real clinical-scale data** and should **not be used for any actual medical decision-making**.

    Please explore and learn from this tool, but do not interpret its output as medical advice or a diagnostic reference. Always consult a qualified healthcare professional for real-world health concerns.
    """)
    #This is for the main screen 
    st.header("Predict the severity of cancer based on various factors. Just use the sidebar sliders to adjust the values, and the model will do the rest!")
    st.text("")
    ###Sliders for the prediction 
    age = st.text_input("Enter your age")
    if age:
        try:
            age = int(age)
        except ValueError:
            st.error("Please enter a valid number for age.")
    else:
        st.warning("Please enter your age to continue.")
    st.text("")
    gender = st.selectbox("Choose your Gender:",["Male","Female"])
    # User puts their smoking Details
    st.markdown("### How often do you smoke? *(scale of 0 to 10)*")
    smoking = st.slider("", 0, 10, 0, key="smoke")
    st.divider()

    # Alcohol Consumption Details
    st.markdown("### How often do you consume alcohol? *(scale of 0 to 10)*")
    alcoholUse = st.slider("", 0, 10, 0, key="alcohol")
    st.divider()

    # How obese users are?
    st.markdown("### How would you rate your obesity level? *(scale of 0 to 10)*")
    obesity = st.slider("", 0, 10, 0, key="obesity")
    st.divider()

    # How Balanced is the Diet of the users?
    st.markdown("### How balanced is your daily diet? *(scale of 0 to 10)*")
    balancedDiet = st.slider("", 0, 10, 0, key="diet")
    st.divider()

    # How fatigued do the user feels?
    st.markdown("### How often do you feel fatigued? *(scale of 0 to 10)*")
    fatigue = st.slider("", 0, 10, 0, key="fatigue")
    st.divider()

    # Is there presence of the blood while coughing ?
    st.markdown("### How often do you experience coughing of blood? *(scale of 0 to 10)*")
    coughingBlood = st.slider("", 0, 10, 0, key="coughing_blood")
    st.divider()

    # How is the chest pain like?!
    st.markdown("### How severe is your chest pain? *(scale of 0 to 10)*")
    chestPain = st.slider("", 0, 10, 0, key="chest_pain")
    st.divider()

    # How polluted the environment is in their area?
    st.markdown("### How much are you exposed to air pollution? *(scale of 0 to 10)*")
    airPollution = st.slider("", 0, 10, 0, key="air_pollution")
    st.divider()

    # Has anyone been victim of Lung Cancer in their family?!
    st.markdown("### What do you consider your genetic risk to be? *(scale of 0 to 10)*")
    geneticRisk = st.slider("", 0, 10, 0, key="genetic_risk")
    st.divider()

    # How severe is the user's lung disease
    st.markdown("### How severe is your chronic lung disease? *(scale of 0 to 10)*")
    chronicLungDisease = st.slider("", 0, 10, 0, key="lung_disease")
    st.divider()

    ## Model's Prediction 
    genderBinary = 0 if gender.lower() == 'female' else 1
    tobePredicted = np.array([age, genderBinary,smoking,alcoholUse,obesity,balancedDiet,fatigue, coughingBlood,chestPain,airPollution,geneticRisk,chronicLungDisease]).reshape(1,-1)
    print(tobePredicted)
    ##Model 
    model = joblib.load('lungCancerPredictor.pkl')
    if st.button("Predict"):
            prediction = model.predict(tobePredicted)
            st.subheader("Based on the details you have predicted")
            if prediction == 0:
                st.success(" Based on the input provided, the predicted severity level is **Low**. This indicates a minimal likelihood of severe lung cancer symptoms at this stage.")
            elif prediction == 1:
                st.info("Based on the input provided, the predicted severity level is **Moderate**. There may be a developing risk that should be monitored more closely.")
            else:
                st.error("Based on the input provided, the predicted severity level is **High**. This suggests a greater likelihood of severe lung cancer indicators. Please consider consulting a medical professional for further evaluation.")
            st.warning("⚠️ This is a demo. Do not rely on this prediction for medical advice.")


elif page == ' ℹ️ About the Model':
    st.sidebar.markdown("""
    ---  
    ### ⚠️ Disclaimer

    This application was developed as part of a data science project and is based on a relatively small and synthetic dataset. While the model demonstrates how lifestyle and clinical features can be used to predict lung cancer severity levels, it is **not trained on real clinical-scale data** and should **not be used for any actual medical decision-making**.

    Please explore and learn from this tool, but do not interpret its output as medical advice or a diagnostic reference. Always consult a qualified healthcare professional for real-world health concerns.
    """)
    st.title("About the App - 🎗️ Lung Cancer Severity Checker")
    st.text("")
    st.subheader("""
    This lung cancer severity prediction model uses a Logistic Regression classifier built with a scikit-learn Pipeline. It processes clinical and lifestyle features like age, gender, smoking, alcohol use, obesity, diet, symptoms (fatigue, chest pain, blood cough), pollution exposure, genetic risk, and chronic lung disease.""")

    st.subheader("This educational model, built for project purposes, achieved over 94% in precision, recall, and accuracy. It is intended for those already affected to assess severity.")

    st.warning("This model is based on a small dataset and should not be relied upon. It is intended for educational purposes only.")


else:
    st.sidebar.title("Meet the developer")
    st.sidebar.header("Sujal Adhikari")
    st.sidebar.write("Caldwell, New Jersey")
    st.sidebar.write("Data Science | Data Analysis | Machine Learning")
    st.text("")
    st.sidebar.text("Thank you for trying out my first web app built with scikit-learn! This journey has had its challenges, but I'm proud to share something with real-life meaning. It's just the beginning—and purely educational for now. I’ll keep building and sharing more. Feel free to connect with me and follow my journey below!")
    st.sidebar.markdown("[Github](https://github.com/suzaladhikari)", unsafe_allow_html=True)
    st.sidebar.markdown("[Twitter](https://twitter.com/LifeOfSujal)", unsafe_allow_html=True)
    st.sidebar.markdown("sujal.adhikari.ds@gmail.com")

    st.title("Greetings 👋 ¡Hola! 👋 Bonjour")
    st.text("")
    st.subheader("Hi, I'm **Sujal Adhikari**, a sophomore at Caldwell University and an aspiring Data Scientist. This project means a lot to me—it’s my very first step into real-world machine learning, built from the ground up through hard work, persistence, and a deep desire to learn. There were challenges along the way, but every late night and every debugging session was worth it. I’m incredibly grateful to **Dr. Vlad Veksler** for his guidance and for helping me understand what it truly takes to grow in this field. Creating something meaningful using what I’ve learned so far has been both humbling and empowering. This is just the beginning, and I’m excited to keep learning, growing, and building. Thank you for being a part of this journey.")
    st.text("")
    st.subheader("Thank You 🙏 Gracias! 🙏 Merci")
