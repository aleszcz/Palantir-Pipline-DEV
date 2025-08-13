# Example file

#data cleaning
# The following code performs data cleaning on the dataset All_patients_summary_fact_table_de_id. First, it selects relevant features needed for model building. Then, it filters out patients with missing or zero age and those with unspecified or other genders. Finally, it filters the dataset to include only races that are well represented.#


def data_cleaning(All_patients_summary_fact_table_de_id):
    # Select features for model bulding
    df = All_patients_summary_fact_table_de_id.select("age","BMI_max_observed_or_calculated","sex","race","race_ethnicity","OBESITY_indicator","PREGNANCY_indicator","TOBACCOSMOKER_indicator","CARDIOMYOPATHIES_indicator","CEREBROVASCULARDISEASE_indicator","CHRONICLUNGDISEASE_indicator","CONGESTIVEHEARTFAILURE_indicator","CORONARYARTERYDISEASE_indicator",
"DEMENTIA_indicator","DEPRESSION_indicator","DIABETESCOMPLICATED_indicator","DIABETESUNCOMPLICATED_indicator","DOWNSYNDROME_indicator","HEARTFAILURE_indicator","HEMIPLEGIAORPARAPLEGIA_indicator","HIVINFECTION_indicator","HYPERTENSION_indicator","KIDNEYDISEASE_indicator","MALIGNANTCANCER_indicator","METASTATICSOLIDTUMORCANCERS_indicator","MILDLIVERDISEASE_indicator","MODERATESEVERELIVERDISEASE_indicator","MYOCARDIALINFARCTION_indicator","OTHERIMMUNOCOMPROMISED_indicator","PEPTICULCER_indicator","PERIPHERALVASCULARDISEASE_indicator","PSYCHOSIS_indicator","PULMONARYEMBOLISM_indicator","RHEUMATOLOGICDISEASE_indicator","SICKLECELLDISEASE_indicator","SUBSTANCEUSEDISORDER_indicator","THALASSEMIA_indicator","TUBERCULOSIS_indicator","SYSTEMICCORTICOSTEROIDS_indicator","patient_death_indicator")

    # Select patients age if it is null or zero dropped it
    df_age = df.where(df.age>=1)
    print ('# of patients age>1 between dates are', df_age.count())

    # Dropped Sex if it is missing or others (No matching concepts)
    df_gender = df_age.where((df_age.sex == 'MALE') | (df_age.sex == 'FEMALE'))
    print ('# of patients with gender Male or Female', df_gender.count())

    # Select race with good representation in the data
    df_race = df_gender.where((df_gender.race == 'White') | (df_gender.race == 'Asian') |(df_gender.race == 'Black or African American') | (df_gender.race == 'Unknown'))
    print ('# of patients with race (white, Asian,Black and Unknown', df_race.count())

    return df_race

#Feature enginerring

### 🧪 Feature Engineering for Group Comparison

# This node prepares the dataset for downstream analysis by selecting relevant clinical and demographic features, then creating binary group indicators to compare model behavior across populations.

# Specifically, it:
# - Selects key variables (e.g., age, comorbidities, outcomes).
# - Converts race and sex into simplified group indicators for demonstration: `1` for individuals identified as White or male (privileged), and `0` otherwise.
# - Converts the resulting Spark DataFrame into a Pandas DataFrame for use in fairness evaluation and visualization steps later in the pipeline.

# This setup allows us to explore how model predictions may vary across groups, using a basic and replicable example for teaching purposes.

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

# Initialize Spark session
spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

def feature_engineering_privilege(data_cleaning):

    # Select relevant columns
    df = data_cleaning.select(
        "age", "sex", "race", "OBESITY_indicator", "PREGNANCY_indicator", "TOBACCOSMOKER_indicator",
        "CARDIOMYOPATHIES_indicator", "CEREBROVASCULARDISEASE_indicator", "CHRONICLUNGDISEASE_indicator",
        "CONGESTIVEHEARTFAILURE_indicator", "CORONARYARTERYDISEASE_indicator", "DEMENTIA_indicator",
        "DEPRESSION_indicator", "DIABETESCOMPLICATED_indicator", "DIABETESUNCOMPLICATED_indicator",
        "DOWNSYNDROME_indicator", "HEARTFAILURE_indicator", "HEMIPLEGIAORPARAPLEGIA_indicator",
        "HIVINFECTION_indicator", "HYPERTENSION_indicator", "KIDNEYDISEASE_indicator",
        "MALIGNANTCANCER_indicator", "METASTATICSOLIDTUMORCANCERS_indicator", "MILDLIVERDISEASE_indicator",
        "MODERATESEVERELIVERDISEASE_indicator", "MYOCARDIALINFARCTION_indicator",
        "OTHERIMMUNOCOMPROMISED_indicator", "PEPTICULCER_indicator", "PERIPHERALVASCULARDISEASE_indicator",
        "PSYCHOSIS_indicator", "PULMONARYEMBOLISM_indicator", "RHEUMATOLOGICDISEASE_indicator",
        "SICKLECELLDISEASE_indicator", "SUBSTANCEUSEDISORDER_indicator", "THALASSEMIA_indicator",
        "TUBERCULOSIS_indicator", "SYSTEMICCORTICOSTEROIDS_indicator", "patient_death_indicator",

    )

    # Define a dictionary of values to replace in Race
    replace_dict = {'Black or African American': 'Black_or_African_American'}
    df = df.replace(replace_dict, subset=['race'])

    # Convert race and sex to privileged and unprivileged groups
    df = df.withColumn("sex_privilege", F.when(F.col("sex") == "MALE", 1).otherwise(0))
    df = df.withColumn("race_privilege", F.when(F.col("race") == "White", 1).otherwise(0))

    # Select the necessary columns and convert to Pandas DataFrame
    df = df.select([col for col in df.columns if col not in ["sex", "race"]]).toPandas()

    return df

#feature scaling
#The following code performs feature scaling and removes quasi-constant features from the dataset. It first separates the continuous features and applies standard scaling to ensure they #are on a comparable scale. Then, it concatenates the scaled continuous features back with the categorical features. Finally, it removes quasi-constant features, which have 99% of their #values the same, to ensure the dataset is suitable for model building.


def feature_scaling(feature_engineering_privilege):
    df = feature_engineering_privilege
    # continous feature
    continous_features = ["age"]
    df_cont = df[continous_features]

    # Create a StandardScaler object
    scaler = StandardScaler()
    continous_features_ss = pd.DataFrame(scaler.fit_transform(df_cont), columns = df_cont.columns.tolist())

    # drop continous features from dataframe to select cotegorical features
    columns = df_cont.columns.tolist()
    df_cate = df.drop(columns, axis = 1)

    # categorical_features = df_cate.columns.tolist()
    df_scaled = pd.concat([continous_features_ss , df_cate], axis = 1 )

    # remove constant features  (since summary table shows that there is no constant features)
#    constant_filter = VarianceThreshold(threshold = 0)
#    constant_filter.fit(df)
#    df = df[df.columns[constant_filter.get_support(indices = True)]]

    # Remove quasi_constant features ( here we exclude feature which have 99% same values in the features)
    quasi_constant_filter = VarianceThreshold(threshold = 0.01)
    quasi_constant_filter.fit(df_scaled)
    df_scaled = df_scaled[df_scaled.columns[quasi_constant_filter.get_support(indices = True)]]

    return df_scaled

#predict

# Train a logistic regression model to predict chronic lung disease.
# The target is 'CHRONICLUNGDISEASE_indicator'. We split the dataset,
# fit the model, generate predictions, and evaluate performance using
# a confusion matrix and classification report.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

def predict_lung(feature_scaling):
    # Convert to Pandas DataFrame if necessary
    if isinstance(feature_scaling, pd.DataFrame):
        df = feature_scaling
    else:
        df = feature_scaling.toPandas()

    # Define features and target
    y = df['CHRONICLUNGDISEASE_indicator']
    X = df.drop(['CHRONICLUNGDISEASE_indicator'], axis=1)
    # print(y)
    # print('-----------------------------------------------------')
    # print(X)
    # Split the data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(train_x, train_y)

    # Predict total_visits on the test set
    test_y_pred = logreg.predict(test_x)

    # Print the confusion matrix and classification report
    cm = confusion_matrix(test_y, test_y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(test_y, test_y_pred))

    # Add predictions to the original dataframe for the entire dataset
    df['predicted_CHRONICLUNGDISEASE_indicator'] = logreg.predict(X)



    return df

#Race fairness

# This function calculates accuracy, TPR, FPR, FNR, and PPP for each group
# defined by the `race_privilege` indicator (1 = White, 0 = Non-White).
# It then computes the ratio of unprivileged to privileged group metrics
# to observe any large performance gaps. This step helps identify whether
# the model behaves consistently across these two race-based groups.

# Output: Group-Level Performance Ratios (Non-White / White)
# This output shows how the model's performance compares across race-defined groups.
# A TPR ratio > 1 means the unprivileged group had a higher true positive rate.
# Similar interpretations apply for FPR, FNR, accuracy, and PPP.
# Large differences across these ratios suggest inconsistent model behavior across groups.

from sklearn.metrics import confusion_matrix
import numpy as np

def calculate_race_fairness_metrics(predict_lung):

    # Calculate fairness for a subgroup of population
    def fairness_metrics(df):
        cm = confusion_matrix(df['y'], df['y_pred'])
        TN, FP, FN, TP = cm.ravel()

        N = TP + FP + FN + TN  # Total population
        ACC = (TP + TN) / N  # Accuracy
        TPR = TP / (TP + FN)  # True positive rate
        FPR = FP / (FP + TN)  # False positive rate
        FNR = FN / (TP + FN)  # False negative rate
        PPP = (TP + FP) / N  # % portion of predicted as positive

        return np.array([ACC, TPR, FPR, FNR, PPP])

    predict_lung['y'] = predict_lung['CHRONICLUNGDISEASE_indicator']
    predict_lung['y_pred'] = predict_lung['predicted_CHRONICLUNGDISEASE_indicator']

    # Calculate fairness metrics for race
    fm_race_1 = fairness_metrics(predict_lung[predict_lung['race_privilege'] == 1])
    fm_race_0 = fairness_metrics(predict_lung[predict_lung['race_privilege'] == 0])
    fm_race = fm_race_0 / fm_race_1

    print("Fairness metrics for race (Non-White/White):")
    print(f"Accuracy Ratio: {fm_race[0]:.4f}")
    print(f"TPR Ratio: {fm_race[1]:.4f}")
    print(f"FPR Ratio: {fm_race[2]:.4f}")
    print(f"FNR Ratio: {fm_race[3]:.4f}")
    print(f"PPP Ratio: {fm_race[4]:.4f}\n")

#intersectional fairness matrix

# This function calculates key performance metrics (accuracy, TPR, FPR, FNR, PPP)
# for groups defined by race, sex, and their intersection (e.g., Female & Non-White).
# Ratios are calculated to compare group-level differences.
# This helps assess whether the model behaves consistently across combined group identities.

# Output: Group Performance Metrics
# The model performs very differently across subgroups.
# - TPR and PPP ratios are particularly high for Non-White and Female groups,suggesting over-prediction or more positive classifications.
# - Intersectional metrics show that some subgroups (e.g., Female & White, Male & Non-White) receive almost no positive predictions, with a TPR near 0 and FNR near 1.
# These results highlight that even if overall accuracy looks acceptable,
# model behavior may be highly inconsistent across intersectional identities.



from sklearn.metrics import confusion_matrix
import numpy as np

def calculate_intersectional_fairness_metrics_1(predict_lung):

    # Calculate fairness for a subgroup of population
    def fairness_metrics(df):
        cm = confusion_matrix(df['y'], df['y_pred'])
        TN, FP, FN, TP = cm.ravel()

        N = TP + FP + FN + TN  # Total population
        ACC = (TP + TN) / N  # Accuracy
        TPR = TP / (TP + FN)  # True positive rate
        FPR = FP / (FP + TN)  # False positive rate
        FNR = FN / (TP + FN)  # False negative rate
        PPP = (TP + FP) / N  # % predicted as positive

        return np.array([ACC, TPR, FPR, FNR, PPP])

    predict_lung['y'] = predict_lung['CHRONICLUNGDISEASE_indicator']
    predict_lung['y_pred'] = predict_lung['predicted_CHRONICLUNGDISEASE_indicator']

    # Calculate fairness metrics for race
    fm_race_1 = fairness_metrics(predict_lung[predict_lung['race_privilege'] == 1])
    fm_race_0 = fairness_metrics(predict_lung[predict_lung['race_privilege'] == 0])
    fm_race = fm_race_0 / fm_race_1
    print("Fairness metrics for race (Non-White/White):")
    print(f"Accuracy Ratio: {fm_race[0]:.4f}, TPR Ratio: {fm_race[1]:.4f}, FPR Ratio: {fm_race[2]:.4f}, FNR Ratio: {fm_race[3]:.4f}, PPP Ratio: {fm_race[4]:.4f}\n")

    # Calculate fairness metrics for sex
    fm_sex_1 = fairness_metrics(predict_lung[predict_lung['sex_privilege'] == 1])
    fm_sex_0 = fairness_metrics(predict_lung[predict_lung['sex_privilege'] == 0])
    fm_sex = fm_sex_0 / fm_sex_1
    print("Fairness metrics for sex (Female/Male):")
    print(f"Accuracy Ratio: {fm_sex[0]:.4f}, TPR Ratio: {fm_sex[1]:.4f}, FPR Ratio: {fm_sex[2]:.4f}, FNR Ratio: {fm_sex[3]:.4f}, PPP Ratio: {fm_sex[4]:.4f}\n")

    # Calculate fairness metrics for intersection of race and sex
    predict_lung['intersection_group'] = predict_lung.apply(
        lambda row: f"{'Male' if row['sex_privilege'] == 1 else 'Female'}_"
                    f"{'White' if row['race_privilege'] == 1 else 'Non-White'}", axis=1)

    intersection_groups = predict_lung['intersection_group'].unique()
    for group in intersection_groups:
        subset = predict_lung[predict_lung['intersection_group'] == group]
        metrics = fairness_metrics(subset)
        print(f"Fairness metrics for {group.replace('_', ' & ')}:")
        print(f"Accuracy: {metrics[0]:.4f}, TPR: {metrics[1]:.4f}, FPR: {metrics[2]:.4f}, FNR: {metrics[3]:.4f}, PPP: {metrics[4]:.4f}\n")

"""** Additional Methods for bias mitigation"""

#SMOTE


# In this step, we use SMOTE (Synthetic Minority Oversampling Technique) to address the class imbalance problem
# by generating synthetic examples for the minority class (chronic lung disease cases).
# This balances the training data before fitting a logistic regression model.

# Output: The confusion matrix and classification report show how the model performs after applying SMOTE:
# - Class 1 recall improved dramatically (from 0% to 65%), meaning the model is now catching many more positive cases.
# - However, precision for class 1 is still very low, which means the model is also producing many false positives.
# - Overall accuracy dropped compared to the imbalanced model, but this is expected since the model is now more focused
#   on detecting the minority class instead of only optimizing for the majority.
# This trade-off is common when addressing imbalance and highlights the need to look beyond accuracy alone.

###########################################################
#Note: When applying SMOTE (Synthetic Minority Over-sampling Technique), the standard method does not consider privileged and unprivileged groups during the oversampling process. SMOTE generates synthetic samples for the minority class without accounting for any specific fairness constraints related to these groups.However, there are fairness-aware variants of SMOTE, such as Fair-SMOTE, which aim to balance the representation of privileged and unprivileged groups. These methods specifically ####modify the oversampling process to address fairness issues by ensuring that the generated synthetic samples do not exacerbate existing biases
##############################################################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

def predict_lung_SMOTE(feature_scaling):
    # Convert to Pandas DataFrame if necessary
    if isinstance(feature_scaling, pd.DataFrame):
        df = feature_scaling
    else:
        df = feature_scaling.toPandas()

    # Define features and target
    y = df['CHRONICLUNGDISEASE_indicator']
    X = df.drop(['CHRONICLUNGDISEASE_indicator'], axis=1)

    # Split the data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    train_x_resampled, train_y_resampled = smote.fit_resample(train_x, train_y)

    # Train the logistic regression model on resampled data
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(train_x_resampled, train_y_resampled)

    # Predict chronic lung disease on the test set
    test_y_pred = logreg.predict(test_x)

    # Print the confusion matrix and classification report
    cm = confusion_matrix(test_y, test_y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(test_y, test_y_pred))

    # Add predictions to the original dataframe for the entire dataset
    df['predicted_CHRONICLUNGDISEASE_indicator'] = logreg.predict(X)

    return df

#REWEIGTING

# Reweighting with Fairlearn GridSearch (Demographic Parity Constraint)
#
# In this code, we apply a fairness-aware training method using Fairlearn’s GridSearch.
# The goal is to train a logistic regression model that meets a fairness constraint:
# specifically, *demographic parity* — meaning the model should predict positive outcomes
# at similar rates across groups defined by sex and race privilege.
#
# To achieve this, Fairlearn adjusts the importance (weights) of different training samples
# during model training. These weights guide the model to treat each group more equally,
# especially when one group is underrepresented or harder to classify.
#
# Is this Pre-processing or In-processing?
# - **Statistically**, reweighting is sometimes introduced as a *pre-processing* technique
#   because it adjusts how data contributes to training without modifying the model itself.
# - **Practically**, when using Fairlearn’s `GridSearch`, reweighting happens *during*
#   model optimization — so it functions as an *in-processing* method.
#
# Output Interpretation:
# - The model now identifies more positive cases in the minority class (recall for class 1 is 28%),
#   but precision remains low, and overall accuracy dropped to 68%.
# - This is a typical trade-off when enforcing fairness constraints: group-level balance improves,
#   but performance metrics like precision and accuracy may decline.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from fairlearn.reductions import GridSearch, DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, true_negative_rate

def predict_lung_with_reweighting(feature_scaling):
    # Convert to Pandas DataFrame if necessary
    if isinstance(feature_scaling, pd.DataFrame):
        df = feature_scaling
    else:
        df = feature_scaling.toPandas()

    # Define features and target
    y = df['CHRONICLUNGDISEASE_indicator']
    X = df.drop(['CHRONICLUNGDISEASE_indicator'], axis=1)
    sensitive_features = df[['sex_privilege', 'race_privilege']]

    # Split the data into training and testing sets
    train_x, test_x, train_y, test_y, train_s, test_s = train_test_split(X, y, sensitive_features, test_size=0.2, random_state=42)

    # Define logistic regression model
    estimator = LogisticRegression(max_iter=1000)

    # Apply GridSearch for reweighting
    mitigator = GridSearch(estimator, constraints=DemographicParity(), grid_size=10)
    mitigator.fit(train_x, train_y, sensitive_features=train_s)

    # Predict on the test set
    test_y_pred = mitigator.predict(test_x)

    # Print the confusion matrix and classification report
    cm = confusion_matrix(test_y, test_y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(test_y, test_y_pred))

    # Add predictions to the original dataframe for the entire dataset
    df['predicted_CHRONICLUNGDISEASE_indicator'] = mitigator.predict(X)

    return df

#Adversal nn
# Adversarial Debiasing with Neural Networks
# This node demonstrates a simple implementation of adversarial debiasing, a fairness-aware machine learning technique that tries to reduce the model's reliance on sensitive attributes (like race) during prediction.

# How It Works:
# Primary Task:
# We first train a neural network to predict whether a patient has chronic lung disease. This is the main prediction task (the "primary model").

# Adversarial Task:
# After the primary model is trained, we use its predicted outputs to train a second model (the "adversary") that tries to predict the sensitive attribute — in this case, race_privilege.
# If the adversary performs well, it suggests that the primary model is leaking information about race — even if race wasn't explicitly used in the model input.

# Goal:
# In a more advanced setup, the two models would be trained jointly so that the primary model learns to make accurate predictions while minimizing the adversary’s ability to detect group membership. Here, we demonstrate just the core logic for educational purposes.

# Outputs:
# The confusion matrix and classification report for the primary task show how well the model predicts lung disease.

# The adversarial results show how much sensitive group information remains in the model output — if performance is high, bias is still present; if it drops, the model may be fairer.



import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def predict_lung_with_adversarial_nn(feature_scaling):
    # Convert to Pandas DataFrame if necessary
    df = feature_scaling if isinstance(feature_scaling, pd.DataFrame) else feature_scaling.toPandas()

    # Define features and target for the primary task
    y_primary = df['CHRONICLUNGDISEASE_indicator']
    X = df.drop(['CHRONICLUNGDISEASE_indicator', 'race_privilege'], axis=1)

    # Split the data into training and testing sets for the primary task
    train_x, test_x, train_y, test_y = train_test_split(X, y_primary, test_size=0.2, random_state=42)

    # Build the primary model (simplified neural network)
    input_layer = Input(shape=(train_x.shape[1],))
    dense_layer = Dense(32, activation='relu')(input_layer)
    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    primary_model = Model(inputs=input_layer, outputs=output_layer)
    primary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the primary model
    primary_model.fit(train_x, train_y, epochs=2, batch_size=32, validation_split=0.1, verbose=0)

    # Predict on the test set for the primary task
    test_y_pred_primary = (primary_model.predict(test_x) > 0.5).astype("int32")

    # Print the confusion matrix and classification report for the primary task
    cm_primary = confusion_matrix(test_y, test_y_pred_primary)
    cr_primary = classification_report(test_y, test_y_pred_primary, target_names=['No Chronic Lung Disease', 'Chronic Lung Disease'])

    print("Confusion Matrix for Primary Task:")
    print(cm_primary)
    print("\nClassification Report for Primary Task:")
    print(cr_primary)

    # Add primary task predictions to the original dataframe
    df['predicted_CHRONICLUNGDISEASE_indicator'] = (primary_model.predict(X) > 0.5).astype("int32")

    # Define features and target for the adversarial task (predicting sensitive attribute 'race_privilege')
    y_adversary = df['race_privilege']
    X_adversary = df[['predicted_CHRONICLUNGDISEASE_indicator']]  # Using the predictions from the primary task

    # Split the data into training and testing sets for the adversarial task
    train_x_adv, test_x_adv, train_y_adv, test_y_adv = train_test_split(X_adversary, y_adversary, test_size=0.2, random_state=42)

    # Build the adversarial model (simplified neural network)
    input_layer_adv = Input(shape=(train_x_adv.shape[1],))
    dense_layer_adv = Dense(32, activation='relu')(input_layer_adv)
    output_layer_adv = Dense(1, activation='sigmoid')(dense_layer_adv)

    adversary_model = Model(inputs=input_layer_adv, outputs=output_layer_adv)
    adversary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the adversarial model
    adversary_model.fit(train_x_adv, train_y_adv, epochs=2, batch_size=32, validation_split=0.1, verbose=0)

    # Predict on the test set for the adversarial task
    test_y_adv_pred = (adversary_model.predict(test_x_adv) > 0.5).astype("int32")

    # Print the confusion matrix and classification report for the adversarial task
    cm_adv = confusion_matrix(test_y_adv, test_y_adv_pred)
    cr_adv = classification_report(test_y_adv, test_y_adv_pred, target_names=['No Race Privilege', 'Race Privilege'])

    print("Confusion Matrix for Adversarial Task:")
    print(cm_adv)
    print("\nClassification Report for Adversarial Task:")
    print(cr_adv)

    return df

#eqilized odds

# Post-Processing with Equalized Odds (Fairlearn)
# In this example, we apply a post-processing fairness technique using Fairlearn’s ThresholdOptimizer. The goal is to adjust the prediction thresholds after training a standard logistic regression model, without changing the model itself. This method enforces the equalized odds constraint — meaning the true positive and false positive rates should be similar across different groups (e.g., by race).

# The process is as follows:

# We first train a standard logistic regression model.

# Then, we use ThresholdOptimizer to adjust decision thresholds based on group membership (race_privilege) to satisfy the fairness constraint.

# We evaluate model performance before and after applying post-processing, and also calculate fairness metrics like demographic parity difference and equalized odds difference.

# Note: Post-processing is often more flexible because it doesn’t require altering the original model, making it especially useful in situations where retraining is not feasible.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def apply_equalized_odds(feature_scaling):
    df= feature_scaling
    # Define features and target
    y = df['CHRONICLUNGDISEASE_indicator']
    X = df.drop(['CHRONICLUNGDISEASE_indicator'], axis=1)
    sensitive_feature = df['race_privilege']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(X, y, sensitive_feature, test_size=0.2, random_state=42)

    # Train the logistic regression model
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = logreg.predict(X_test)

    # Print the confusion matrix and classification report before post-processing
    print("Confusion Matrix before post-processing:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report before post-processing:")
    print(classification_report(y_test, y_pred))

    # Apply the Equality of Odds post-processing technique
    threshold_optimizer = ThresholdOptimizer(
        estimator=logreg,
        constraints="equalized_odds",
        prefit=True
    )
    threshold_optimizer.fit(X_train, y_train, sensitive_features=sensitive_train)
    y_pred_post = threshold_optimizer.predict(X_test, sensitive_features=sensitive_test)

    # Print the confusion matrix and classification report after post-processing
    print("Confusion Matrix after post-processing:")
    print(confusion_matrix(y_test, y_pred_post))
    print("\nClassification Report after post-processing:")
    print(classification_report(y_test, y_pred_post))

    # Assess fairness metrics
    dem_parity_diff = demographic_parity_difference(y_test, y_pred_post, sensitive_features=sensitive_test)
    eq_odds_diff = equalized_odds_difference(y_test, y_pred_post, sensitive_features=sensitive_test)

    print("Demographic Parity Difference after post-processing:", dem_parity_diff)
    print("Equalized Odds Difference after post-processing:", eq_odds_diff)

    # Add predictions to the original dataframe for the entire dataset
    df['predicted_CHRONICLUNGDISEASE_indicator'] = logreg.predict(X)
    df['post_processed_CHRONICLUNGDISEASE_indicator'] = threshold_optimizer.predict(X, sensitive_features=df['race_privilege'])

    return df



