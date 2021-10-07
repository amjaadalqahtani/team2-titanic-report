import pandas as pd 
import seaborn as sns 
import statsmodels.api as sm 
import math

titanic = pd.read_csv('data/train.csv')
titanic

#Descriptive statistics: 
titanic.info()
titanic.describe()

# the number od survivals 
survivals = titanic[titanic['Survived'] == 1]
len(survivals)

survival_percent = (len(survivals)/891) * 100

#number of survival based on sex 
group_s = survivals.groupby('Sex')["Survived"].count()
group_s= pd.DataFrame(group_s)
group_s

#number of survival based on Class 
group_class = survivals.groupby('Pclass')["Survived"].count()
group_class= pd.DataFrame(group_class)
group_class

#last name 
lastNames= titanic['Name'].str.split(',', expand=True)
lastNames = lastNames.rename({'0': 'Last Name', '1': 'Full Name'}, axis='columns')
lastNames.columns=['Last','Full'] 
last_count = lastNames.groupby('Last').agg(['count'])
last_count

#plots 
sns.catplot("Sex",data=titanic,kind="count")
"""Here, we see the gender distribution of passengers."""
sns.catplot(x ="Sex", hue ="Survived",kind ="count", data = titanic).set(title='Surivival by Sex, n=342')
"""Based on the plot, we can conclude that more women (233) survived overall compared to men (109)."""
sns.catplot(x ="Pclass", hue ="Survived",kind ="count", data = titanic).set(title='Surivival by Class, n=342')
"""Based on the plot, we can conclude that Class 1 had the highest survival rate (136), Class 2 had the 
lowest survival rate (87), and Class 3 had the most deaths."""