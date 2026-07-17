from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
data = {
    "Age": [18, 25, 40, 50, 60],
    "Salary": [20000, 50000, 100000, 150000, 200000]
}
df=pd.DataFrame(data)
print(df)
mm=MinMaxScaler()
st=StandardScaler()
df_clean=pd.DataFrame(mm.fit_transform(df),columns=df.columns)
print(df_clean)
df_clean2=pd.DataFrame(st.fit_transform(df),columns=df.columns)
print(df_clean2)


