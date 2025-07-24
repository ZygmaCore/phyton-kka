import pandas as pd

file_path = '/content/drive/MyDrive/web/TGM 2020-2023_eng.csv'
df = pd.read_csv(file_path, sep=';')
display(df.head())

display(df.info())
display(df.isnull().sum())


# 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

cols_to_convert = [
    'Reading Frequency per week',
    'Number of Readings per Quarter',
    'Daily Reading Duration (in minutes)',
    'Internet Access Frequency per Week',
    'Daily Internet Duration (in minutes)',
    'Tingkat Kegemaran Membaca (Reading Interest)'
]

for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)

cols_to_fill = [
    'Internet Access Frequency per Week',
    'Daily Internet Duration (in minutes)'
]

for col in cols_to_fill:
    df[col] = df[col].fillna(df[col].mean())

display(df.isnull().sum())

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_category = encoder.fit_transform(df[['Category']])
encoded_category_df = pd.DataFrame(encoded_category, columns=encoder.get_feature_names_out(['Category']))

X = df[cols_to_convert]
y = encoded_category_df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

display(X_train.head())
display(y_train.head())


# 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)


# 
import numpy as np
from sklearn.metrics import accuracy_score

# Convert one-hot encoded y to class labels
y_train_labels = np.argmax(y_train.values, axis=1)
y_test_labels = np.argmax(y_test.values, axis=1)

model.fit(X_train, y_train_labels)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")


# 
import numpy as np
from sklearn.metrics import accuracy_score

y_train_labels = np.argmax(y_train.values, axis=1)
y_test_labels = np.argmax(y_test.values, axis=1)

model.fit(X_train, y_train_labels)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")