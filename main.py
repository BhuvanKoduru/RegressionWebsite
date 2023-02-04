import numpy as np
import pandas as pd
import streamlit as sl

sl.title("Profit prediction web app")
sl.write ("""

Use this  model to predict your profit!
""")
state=sl.sidebar.selectbox("Select State",("New York","California","Florida"))
sl.write(f"Your state is {state}")
rnd=sl.sidebar.number_input("Enter R&D expenditure:",step=10000)
sl.write(f"Your R&D expenditure is {rnd}")
admn=sl.sidebar.number_input("Enter Admin expenditure:",step=10000)
sl.write(f"Your admin expenditure is {admn}")
mkt=sl.sidebar.number_input("Enter marketing expenditure:",step=10000)
sl.write(f"Your marketing expenditure is {mkt}")

data=pd.read_csv("50_Startups.csv")
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
if state=="New York":
    ans=(regressor.predict([[0.0, 0.0, 1.0, rnd,admn,mkt]]))
elif state=="California":
    ans=(regressor.predict([[1.0, 0.0, 0.0, rnd,admn,mkt]]))
else:
    ans=(regressor.predict([[0.0, 1.0, 0.0, rnd,admn,mkt]]))

sl.write("Your profit is:")
sl.write(f"**{ans[0]}**\n\n")

table=pd.DataFrame(
    {
        "State":[state],
        "R&D Expense":[rnd],"Admin Expense":[admn],"Marketing Expense":[mkt],"Profit":[ans[0]]
    }
)
sl.dataframe(table)


