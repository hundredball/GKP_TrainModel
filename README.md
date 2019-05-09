# GKP_TrainModel
This is used for training the model of classifying GKP tasks

## Prepare EEG data and event file
1. Modify subject_session in read_hdf5.py and run it to generate EEG data with 5 channels in a plain text file
2. Create a new folder named 'date' (e.g. 0420)
3. Put the text file and log file from e-prime in the folder

## Train the model
1. Modify subject_session and date in Lab2_Run.py
2. Run Lab2_Run.py to generate a model path file

## Test the model with other dates
Call the runTest() function in Lab2_Run.py with subject_session and testDate of test data and modelDate of the model you want
