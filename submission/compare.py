import sys
import os.path
import pandas as pd
from sklearn.metrics import accuracy_score

# argument is file pass which you want to check.
args = sys.argv

answer_csv = pd.read_csv('answer.csv')
answer_csv = answer_csv.drop('query_name', axis=1)
answer_csv = answer_csv.drop(1)

if len(sys.argv) > 1 and os.path.isfile(args[1]):
    submit_csv = pd.read_csv(args[1])
    submit_csv = submit_csv.drop('query_name', axis=1)
    submit_csv = submit_csv.drop(1)
else:
    print ('\n\nFile not found.\nReading sample submission')
    submit_csv = pd.read_csv('sample_submission.csv')
    submit_csv = submit_csv.drop('query_name', axis=1)
    submit_csv = submit_csv.drop(1)
val_score = accuracy_score(answer_csv, submit_csv)
print ('Accuracy :' + '{:.2f}'.format(val_score*100) + '%\n')
