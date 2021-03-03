#!/usr/bin/env python
# coding: utf-8

# In[13]:


import sys;
import numpy as np
import pandas as pd
import random
import os

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_core_components as dcc
import dash_html_components as html

import matplotlib.pyplot as plt
import matplotlib as mpl

pio.templates.default = "plotly_dark"

# %% set root folder

os.chdir('/Users/anujverma/PycharmProjects/titanic/')
print("proot set to: {}".format(os.curdir))

# %% set up dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# In[2]:


from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.width', 1920)
pd.set_option('display.max.columns', None)

# In[3]:


## hello world this is a new touch
df = pd.read_csv('./data/train.csv')

obj = df.groupby('Sex').agg({'Survived': ['count', 'sum']})

obj.columns = ["-".join(col) for col in obj.columns]

print(obj.columns)
obj.rename(columns={"Survived-count": "total",
                    "Survived-sum": "survived"}, inplace=True)

# In[ ]:


obj.head()
obj = obj.reset_index()
obj.set_index('Sex', inplace=True)

obj.head()
obj['survival rate'] = (obj.survived / obj.total) * 100

fig = px.line(obj, y='survival rate',
              title='Gender Survival Rate',
              width=800, height=500,

              labels={"Sex": "Gender", "survival rate": "% Survived"},
              )

fig = fig.update_layout(  # customize font and legend orientation & position
    font_family="Rockwell",
    legend=dict(
        title=None, orientation="h", y=1, yanchor="bottom", x=0.5,
        xanchor="center"
    )
)

fig = fig.add_annotation(  # add a text callout with arrow
    text="Proportion of \nsurvived women \nis higher", x="female", y=75,
    arrowhead=1,
    showarrow=True
)

# In[ ]:


my_fig = make_subplots(rows=3, cols=3)
my_fig.append_trace(fig.data[0], row=1, col=1, )

my_fig.append_trace(px.histogram(df, x="Fare", nbins=5, color_discrete_sequence=['indianred']).data[0], row=1, col=2)
my_fig.append_trace(px.histogram(df, x="Fare", nbins=5,
                                 color_discrete_sequence=['purple']).data[0],
                    row=1,
                    col=3)
my_fig.append_trace(px.histogram(df, x="Fare", nbins=5,
                                 color_discrete_sequence=['magenta']).data[0],
                    row=2,
                    col=1)
my_fig.append_trace(px.histogram(df, x="Fare", nbins=5).data[0], row=2, col=2)
my_fig.append_trace(px.histogram(df, x="Fare", nbins=5).data[0], row=2, col=3)
my_fig.append_trace(px.histogram(df, x="Fare", nbins=5).data[0], row=3, col=1)
my_fig.append_trace(px.histogram(df, x="Fare", nbins=5).data[0], row=3, col=2)
my_fig.append_trace(px.histogram(df, x="Fare", nbins=5).data[0], row=3, col=3)

my_fig = my_fig.update_layout(
    # customize font and legend orientation & position
    font_family="Rockwell",
    legend=dict(
        title=None, orientation="h", y=1, yanchor="bottom", x=0.5,
        xanchor="center"
    )
)
my_fig = my_fig.add_annotation(  # add a text callout with arrow
    text="Proportion of \nsurvived women \nis higher", x="female", y=75,
    arrowhead=5,
    showarrow=True
)

# %% upload figure to dash
my_fig.show()
