import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('torchnotebooks/modified_churn.csv')
df.columns = df.columns.str.lower()

encoder = LabelEncoder()


def col_enc(*args, data_frame: pd.DataFrame):
    for arg in args:
        data_frame[arg] = encoder.fit_transform(data_frame[arg])
    return data_frame


df = col_enc('geography', 'gender', data_frame=df)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

