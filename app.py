import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib  # assuming you're using a saved model
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt

#Loading pre-trained model and data 
model = joblib.load('decision_tree_model.pkl')
model_fs = joblib.load('decision_tree_model_fs.pkl')

#loading the dataset
df = pd.read_csv('cleanedCKDdataset.csv')
# Extracting categorical and numerical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

# Sample feature ranking function (replace with your actual feature selection)
def get_feature_ranking():
    # Assuming you have your feature data and target variable
    data = pd.read_csv('cleanedCKDdataset.csv')  # replace with your actual dataset
    X = data.drop('class', axis=1)  # replace 'class' with your target column
    y = data['class']
    
    # Using SelectKBest with ANOVA F-value score function
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    
    scores = selector.scores_
    ranking = pd.DataFrame({'Feature': X.columns, 'Score': scores})
    ranking = ranking.sort_values(by='Score', ascending=False)
    
    return ranking


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset Overview", "Feature Selection", "Predict CKD without feature selection",\
                                   "Predict CKD (Top 5 Features)" ])

# Home page
if page == "Home":
    # Welcome section with image and updated writeup
    st.title("Welcome to the Chronic Kidney Disease Prediction App")

    #Image for CKD
    st.image('ckd.jpeg')  

    # Updated introduction writeup
    st.write("""
    This app allows users to predict the likelihood of Chronic Kidney Disease (CKD) based on symptoms.

    Chronic kidney disease (CKD) is a significant health concern globally. 
             The Global Burden of Disease (GBD) study ranked CKD the 19th leading cause of morbidity and death in 2013 [1]. 
             Worldwide, the age-standardized CKD prevalence is 10.4% in men and 10% in women and is higher in low- and middle-income 
             countries than high-income countries [2]. For sub-Saharan Africa (SSA), recent systematic reviews reported a
              prevalence of 13.9% [3], and 10.1% respectively [4]; The pooled prevalence for CKD was 16% in West Africa, 
             the highest in the continent. CKD is characterized by young age of patients in Africa, huge morbidity and premature deaths.
              About 90% of patients with CKD die within 90 days of starting dialysis.
             
    Navigate to the 'Predict CKD' page to input your symptoms and get a prediction ().

    You can also view the importance of various features in predicting CKD on the 'Feature Selection' page.
    """)
    # Adding a hyperlink using markdown
    st.markdown("""
    Statistics gotten from this research work, visit [Timothy et al., (2021)](https://bmcnephrol.biomedcentral.com/articles/10.1186/s12882-020-02126-8).
    """)
# Prediction page
elif page == "Predict CKD without feature selection":
    st.title("CKD Prediction")
    st.write("Enter the following features to predict the presence or absence of Chronic Kidney Disease. Here all the 25 features are used to predict chronic kidney disease CKD")

    # Input fields for each of the 26 features
    age = st.number_input("Age", min_value=0.0, step=0.1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
    specific_gravity = st.number_input("Specific Gravity", min_value=0.0, step=0.01)
    albumin = st.number_input("Albumin", min_value=0.0, step=0.1)
    sugar = st.number_input("Sugar", min_value=0.0, step=0.1)
    red_blood_cells = st.selectbox("Red Blood Cells (1 for normal, 0 for abnormal)", options=[0, 1])
    pus_cell  = st.selectbox("Pus Cell (1 for normal, 0 for abnormal)", options=[0, 1])
    pus_cell_clumps  = st.selectbox("Pus Cell Clumps (1 for present, 0 for absent)", options=[0, 1])

    bacteria = st.selectbox("Bacteria (1 for present, 0 for absent)", options=[0, 1])
    blood_glucose_random = st.number_input("Blood Glucose Random", min_value=0.0, step=0.1)
    
    blood_urea = st.number_input("Blood Urea", min_value=0.0, step=0.1)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, step=0.1)
    sodium = st.number_input("Sodium", min_value=0.0, step=0.1)
    potassium = st.number_input("Potassium", min_value=0.0, step=0.1)
    haemoglobin = st.number_input("Haemoglobin", min_value=0.0, step=0.1)
    packed_cell_volume = st.number_input("Packed Cell Volume", min_value=0.0, step=0.1)
    white_blood_cell_count = st.number_input("White Blood Cell Count", min_value=0.0, step=0.1)
    red_blood_cell_count = st.number_input("Red Blood Cell Count", min_value=0.0, step=0.1)

    hypertension = st.selectbox("Hypertension (1 for yes, 0 for no)", options=[0, 1])

    diabetes_mellitus = st.selectbox("Diabetes Mellitus (1 for yes, 0 for no)", options=[0, 1])
    coronary_artery_disease = st.selectbox("Coronary Artery Disease (1 for yes, 0 for no)", options=[0, 1])
    appetite = st.selectbox("Appetite (1 for good, 0 for poor)", options=[0, 1])
    peda_edema = st.selectbox("Pedal Edema (1 for yes, 0 for no)", options=[0, 1])
    aanemia = st.selectbox("Anemia (1 for yes, 0 for no)", options=[0, 1])

    # Collect all the inputs into a list
    input_data = [[
        age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell,
        pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, serum_creatinine, sodium,
        potassium, haemoglobin, packed_cell_volume, white_blood_cell_count, red_blood_cell_count,
        hypertension, diabetes_mellitus, coronary_artery_disease, appetite, peda_edema, aanemia
    ]]

    # Display predict button
    if st.button("Predict"):
        # Placeholder for model prediction, replace with actual model prediction logic
        # prediction = model.predict(input_data)
        
        prediction = "Positive" if np.random.rand() > 0.5 else "Negative"  # Dummy prediction for now
        
        st.write(f"The predicted result for CKD is: **{prediction}**")


# Feature Selection page
elif page == "Feature Selection":
    st.title("Feature Importance Ranking")
    
    # Get and display the feature ranking
    st.write("To ease the stress on inputing all the entire 25 features, feature selection has \
             been carried out to view the most relevant features in predicting CKD.\
              Figure 1 shows the top 20 features while figure 2 shows the top 5 features.")
    ranking = get_feature_ranking()
    #st.dataframe(ranking)

    # Create subplots for two charts on the same row (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot the entire top features with their scores in the first column
    ax1.barh(ranking['Feature'], ranking['Score'], color='skyblue')
    ax1.set_xlabel('Scores')
    ax1.set_ylabel('Features')
    ax1.set_title('Feature Scores')
    ax1.invert_yaxis()  # To show the highest scores at the top

    # Plot only the top 5 features in the second column
    top_5 = ranking.head(5)
    ax2.barh(top_5['Feature'], top_5['Score'], color='lightcoral')
    ax2.set_xlabel('Scores')
    ax2.set_ylabel('Features')
    ax2.set_title('Top 5 Feature Scores')
    ax2.invert_yaxis()

    # Adjust layout and display both plots in Streamlit
    plt.tight_layout()
    st.pyplot(fig)  # Use st.pyplot to display the plot with both charts




# Data Preparation page
elif page == "Dataset Overview":
    st.title("Dataset Overview")
    
    # Checking numerical features distribution
    st.subheader("Numerical Features Distribution")
    
    plt.figure(figsize=(20, 15))
    plotnumber = 1
    
    for column in num_cols:  # Make sure num_cols is defined
        if plotnumber <= 14:
            ax = plt.subplot(3, 5, plotnumber)
            sns.histplot(df[column], kde=True)  # Plot distribution of numerical columns
            plt.xlabel(column)
        plotnumber += 1
    
    plt.tight_layout()
    st.pyplot(plt.gcf())  # Display the plot in Streamlit

# Prediction using top 5 features page
elif page == "Predict CKD (Top 5 Features)":
    st.title("Predict CKD Using Top 5 Features")
    
    # Displaying feature importance for user understanding
    st.write("""
        This page predicts Chronic Kidney Disease (CKD) based on the top 5 most important features:
        1. Specific Gravity
        2. Haemoglobin
        3. Packed Cell Volume
        4. Hypertension
        5. Albumin
    """)
    
    # Input fields for the top 5 features
    specific_gravity = st.number_input("Specific Gravity", min_value=0.000, max_value=1.030, step=0.001, format="%.3f")
    haemoglobin = st.number_input("Haemoglobin (g/dl)", min_value=0.0, max_value=20.0, step=0.1)
    packed_cell_volume = st.number_input("Packed Cell Volume", min_value=0.0, max_value=60.0, step=0.1)
    hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", options=[0, 1])
    albumin = st.number_input("Albumin (g/dl)", min_value=0.0, max_value=5.0, step=0.1)

    # Button to trigger prediction
    if st.button("Predict"):
        # Create input array for model
        input_data = np.array([[specific_gravity, haemoglobin, packed_cell_volume, hypertension, albumin]])
        
        # Perform prediction using your trained model (replace this with your model)
        # For example: prediction = model.predict(input_data)
        prediction = model_fs.predict(input_data)  # Replace with your CKD model for prediction
        
        # Display the prediction result
        if prediction == 1:
            st.write("The predicted result is: **Positive for CKD**")
        else:
            st.write("The predicted result is: **Negative for CKD**")
