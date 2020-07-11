
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.ensemble import BaggingClassifier

data=pd.read_csv("C:\\Users\\PC\\Desktop\\DBPROJECT\\kc_house_after_preprocessing.csv")

