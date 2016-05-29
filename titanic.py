
# coding: utf-8

# Titanic

# Question to Answer:
# * What factors are most correlated with people who survived?
# * How many entire families survived?
# * What was the average price paid per class for a ticket?
# 
# The first question covers a wide variety of analyses, so I will focus on that one.

# In[132]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[133]:

# First read the titanic data and store it in a dataframe 

filename = '/Users/elizabethallen/Documents/titanic/titanic_data.csv'
titanic_df = pd.read_csv(filename)


# In[134]:

# Let's take a look at the first few rows of data

titanic_df.head()


# In[135]:

# I'm just going to sanity check to see if there are duplicate passenger entries
number_passengers = len(titanic_df['PassengerId'])
number_unique_passengers = len(np.unique(titanic_df['PassengerId']))
print number_passengers
print number_unique_passengers


# In[136]:

# Now I'm just going to investigate the data and gather some summary metrics
titanic_df.describe()


# In[137]:

# I want to find the number of survivors
titanic_df['Survived'].sum()


# In[138]:

# Out of curiousity, I want to find the passengers who paid the min and max fares
titanic_df.iloc[titanic_df['Fare'].argmax()]


# In[139]:

titanic_df.iloc[titanic_df['Fare'].argmin()]


# In[142]:

# I'm going to look at the distribution of Fares

import matplotlib.mlab as mlab

get_ipython().magic(u'pylab inline')
# example data
mu = titanic_df.mean()['Fare']  # mean of distribution
sigma = titanic_df.std()['Fare']  # standard deviation of distribution
x = mu + sigma * np.random.randn(10000)

num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Fare')
plt.ylabel('Probability')
plt.title(r'Histogram of Fares: $\mu=0.38$, $\sigma=0.48$')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()


# In[143]:

# I want to dig into this further and understand how these variables correlate with each other

def standardize(column):
    # Takes a dataframe column as input and returns the standardized column 
    return (column - column.mean()) / column.std(ddof=0)


# In[144]:

age_with_nulls = standardize(titanic_df['Age'])


# In[145]:

# It looks like there might be some missing values for age for some passengers.
# Let's explore this and resolve it by replacing NaN with appropriate values

titanic_df.isnull().sum()


# In[146]:

# It looks like we have a lot of Null values in the age, Cabin and Embarked columns
# Let's focus on fixing the age column by replacing null values with 0
fill_values = {'Age': 0}
titanic_no_nulls = titanic_df.fillna(fill_values)


# In[147]:

titanic_no_nulls.isnull().sum()


# In[148]:

# Now let's try standardizing again
standardized_age = standardize(titanic_no_nulls['Age'])


# In[149]:

# Now I want to find a correlation between different variables, specifically focusing on survival
def correlation(x,y):
    # Takes two arrays or series as input and returns the correlation between the two
    return (standardize(x) * standardize(y)).mean()


# In[150]:

age = titanic_no_nulls['Age']
cabin_class = titanic_no_nulls['Pclass']
survived = titanic_no_nulls['Survived']
num_siblings_spouses = titanic_no_nulls['SibSp']
num_parents_children = titanic_no_nulls['Parch']
fare = titanic_no_nulls['Fare']


# In[151]:

print correlation(age,survived)
print correlation(num_siblings_spouses,survived)
print correlation(num_parents_children,survived)
print correlation(fare,survived)
print correlation(cabin_class,survived)


# From this, I see the strongest correlations between fare and survival, and cabin class and survival.
# 
# For fare and survival, the correlation tells us that as the fare increases, so does the survival rate.
# 
# For cabin class and survival, the correlation tells us that as the cabin class increases, which means that the cabin class is actually lower, the rate of survival decreseases.
# 
# To me, this points to a correlation between passenger class and odds of survival. I want to visualizae the relationship between survival and other factors, such as class, sex, age, fare and port of embarkation.

# In[152]:

# I'm going to remind myself what the dataframe looks like
titanic_no_nulls.head()


# In[153]:

# Now I'm going to explore survival rates by class
avg_by_class = titanic_no_nulls.groupby('Pclass').mean()


# In[154]:

avg_by_class.head()


# In[155]:

get_ipython().magic(u'pylab inline')
plt.plot(avg_by_class['Survived'])

plt.xlabel('Class')
plt.ylabel('Avg Survival')
plt.title('Avg survival rate as class decreases')
plt.grid(True)
plt.show()


# You can see very clearly the increase in survival rate as class gets higher.

# In[156]:

# Now I'm going to explore survival rates by sex
avg_by_sex = titanic_no_nulls.groupby('Sex').mean()


# In[157]:

avg_by_sex.head()


# Looks like far more women survived than men on average.

# In[158]:

avg_by_sex['Survived'].head


# In[159]:

get_ipython().magic(u'pylab inline')
survived = ('female', 'male')
y_pos = np.arange(len(survived))
performance = avg_by_sex['Survived'].values
#error = np.random.rand(len(people))

plt.barh(y_pos, performance, align='center', alpha=0.8)
plt.yticks(y_pos, survived)
plt.xlabel('Avg rate of survival')
plt.title('Avg survival rates between men and women')

plt.show()


# You can clearly see that women fared better than men during this disaster.

# Conclusions

# Through the exploration of this dataset, I see the strongest correlations between fare and survival, and cabin class and survival.
# 
# We can see that as the fare increases, so does the survival rate for passengers.
# We can also see that as the cabin class increases, which means that the cabin class is actually lower, the rate of survival decreseases.
# Additionally, we saw that women generally survived at a higher rate than men.
# 
# This isn't to say that there weren't additional interesting relationships to be explored, but the data did come
# with some limitations, such as:
# * missing cabin data
# * missing Age data -- this was handled by replacing NaN with 0, which helped when calculating summary metrics but doesn't allow us to paint a true picture of the relationship between age and other variables
# * it would've been great to have information about whether any of the passengers were crew members to see how the crew fared vs the passengers
# * having another dataset that mapped siblings and parents together would've helped to understand the survival rates of  families
# 
# The biggest takeaway for me was the correlation between passenger class and odds of survival. 
