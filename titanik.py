# %% [markdown]
# Data Preparation

# %%
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# read data
train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')
all = pd.concat([train,test])

# %%
# menampilkan data
test
test.head()

# %% [markdown]
# Data Preprocessing

# %%
# duplikasi data mentah ke data untuk diolah
train_olah = train.copy()
train_olah.head()

# %%
# melihat presentase data missing
train_olah.isna().sum() / train_olah.shape[0] * 100

# %%
#  delete cabin
del train_olah['Cabin']

# %%
#  hapus data age missing
train_olah = train_olah[~(train_olah['Age'].isna())]

# %%
#  ambil modus data embarked
train_olah['Embarked'].mode()

# %%
#  isi data missing embarked dengan modus
train_olah['Embarked'] = train_olah['Embarked'].fillna(train_olah['Embarked'].mode()[0])
train_olah['Embarked'].unique()

# %%
#  cek ulang data missing
train_olah.isna().sum() / train_olah.shape[0] * 100

# %%
#  cek data aneh di age
train_olah.loc[(train_olah['Age']%1 != 0)]

# %%
#  fix data usia desimal > 1
train_olah['Age'].loc[(train_olah['Age']%1 != 0) & (train_olah['Age'] > 1)] = np.floor(train_olah['Age'].loc[(train_olah['Age']%1 != 0) & (train_olah['Age'] > 1)])

# %%
#  cek ulang ngab
train_olah[train_olah['PassengerId'] == 768]

# %%
#  fix data usia desimal < 1
train_olah['Age'].loc[train_olah['Age'] < 1] = train_olah['Age'].loc[train_olah['Age'] < 1] * 100

# %%
#  cek ulang lagi aowkwkw
train_olah[train_olah['PassengerId'] == 832]

# %%
#  cek tipe data kolom
train_olah.info()

# %%
#  pisahkan data berdasarkan tipe data
train_num = train_olah[list(train_olah.select_dtypes(include = ['int','float']))]
train_obj = train_olah[['Sex', 'Ticket', 'Embarked']]

# %% [markdown]
# Data Visualization

# %%
#  visualisasi data numerik
for x in train_num:
    plt.hist(train_num[x])
    plt.title(x)
    plt.show()

# %%
#  visualisasi data non numerik
for x in train_obj:
  sns.barplot(train_obj[x].value_counts().index,train_obj[x].value_counts()).set_title(x)
  plt.show()


