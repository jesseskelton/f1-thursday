from datetime import datetime
import pandas as pd
import numpy as np
file = 'test.psv'

true_positives_str = 'True Positive'
false_positives_str = 'False Positive'
false_negatives_str = 'False Negative'
true_negatives_str = 'True Negative'
outcome_column_str = 'Outcome'

# columns=['dates', 'y', 'yhat']
df = pd.read_csv(file, skiprows=1, sep='|')
print('Length of file', len(df.index))

# filter Thursdays
df['dates'] = pd.to_datetime(df.dates, format='%Y-%m-%d %H:%M:%S')
df = df[df['dates'].dt.day_name() == 'Thursday']
print('Number of Thursdays', len(df.index))

conditions = [
    (df['y'].eq(df['yhat']) & df['yhat'].eq(True)),
    (df['y'].ne(df['yhat']) & df['yhat'].eq(True)),
    (df['y'].ne(df['yhat']) & df['yhat'].eq(False)),
    (df['y'].eq(df['yhat']) & df['yhat'].eq(False)),
]

choices = [true_positives_str, false_positives_str, false_negatives_str, true_negatives_str]

df[outcome_column_str] = np.select(conditions, choices, default='N/A').tolist()

print(df)

true_positives = df[outcome_column_str].eq(true_positives_str).sum()
false_positives = df[outcome_column_str].eq(false_positives_str).sum()
false_negatives = df[outcome_column_str].eq(false_negatives_str).sum()
true_negatives = df[outcome_column_str].eq(true_negatives_str).sum()

# precision = truep / truep + falsep
precision = true_positives / (true_positives + false_positives)
print('Preicison', precision)

# recall = truep / truep + falsen
recall = true_positives / (true_positives + false_negatives)
print('Recall', precision)

# f1 = 2 x ((precision * recall) / (precision + recall))
f1 = 2 * ((precision*recall) / (precision + recall))

print('F1 to 5dp', round(f1, 5))
