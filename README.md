install required libraries for AI attendence system

using pip install <library_name>

pip install numpy
pip install pandas
etc..

Ensure that all your files (scripts) are saved in a single folder for easy access and organization.

Steps:
Create Datasets Using 1_datasetCreation.py File:

Run this script.
Enter your name and roll number when prompted.
It will capture photos for creating your dataset.
Preprocess the Dataset Using 2_preprocessingEmbeddings.py:

Run this script to preprocess the dataset, which will prepare the images for model training by creating embeddings.
Train Your AI Model Using 3_trainingFaceML.py:

Use this script to train your model with the preprocessed data.
Recognize People with 5_recognizingPersonwithCSVDatabase.py:

Finally, run this script to check and implement the system by recognizing individuals and updating a CSV-based attendance database.
