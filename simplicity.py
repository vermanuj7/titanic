import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# TODO: create converters for columns

#%%
df = pd.read_csv('./data/train.csv')



# %%
X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Sex', 'Cabin',
                     'Ticket', 'Embarked'])

X.fillna(method='ffill', inplace=True)
# X = pd.get_dummies(X, columns=['species'], prefix_sep='=')
y = df['Survived']

model = LinearRegression()
model.fit(X, y)

colors = ['Positive' if c > 0 else 'Negative' for c in model.coef_]

fig = px.bar(
    x=X.columns, y=model.coef_, color=colors,
    color_discrete_sequence=['red', 'blue'],
    labels=dict(x='Feature', y='Linear coefficient'),
    title='Weight of each feature for predicting Titanic Survival',
    template='plotly_dark'
)
fig.show()
