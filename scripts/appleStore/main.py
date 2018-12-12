import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

store = pd.read_csv("../../data/appleStore/appleStore.csv")


def lack_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    lack_table = pd.concat([null_val, percent], axis=1)
    lack_table_ren_columns = lack_table.rename(
        columns={0: 'num of lack', 1: '%'}
    )
    return lack_table_ren_columns


store.describe()
store_shape = store.shape
print(lack_table(store))
print(store_shape)

corr = np.array([store['lang.num'], store['rating_count_tot']])
ans = np.corrcoef(corr)
print(ans)


store['size_MB'] = store['size_bytes'] / (1024 * 1024.0)
store['isNotFree'] = store['price'].apply(lambda xx: 1 if xx > 0 else 0)


plt.scatter(store['size_MB'], store['price'], color="red")
plt.xlabel("size_MB")
plt.ylabel("price")
plt.show()

store['isNotFree'].value_counts().plot.bar()
plt.xlabel('Free = 0, NotFree = 1)')
plt.ylabel('Count')
plt.show()


x = store.size_MB
y = store.rating_count_tot
plt.scatter(x, y, color="blue")
plt.xlabel("size_MB")
plt.ylabel("rating_count_tot")
plt.show()

x = store['ipadSc_urls.num']
y = store.rating_count_tot
plt.scatter(x, y, color="orange")
plt.xlabel("screen shot num")
plt.ylabel("rating_count_tot")
plt.show()

x = store['lang.num']
y = store.rating_count_tot
plt.scatter(x, y, color="orange")
plt.xlabel("lang num")
plt.ylabel("rating_count_tot")
plt.show()

x = store.isNotFree
y = store.size_MB
plt.scatter(x, y, color="purple")
plt.xlabel("Free = 0, NotFree = 1")
plt.ylabel("size_MB")
plt.show()

x = store['size_MB']
y = store['prime_genre']
plt.scatter(x, y, color="green")
plt.xlabel("size_MB")
plt.ylabel("prime_genre")
plt.show()


df_corr = store.drop('id', axis=1).corr()
df_corr['price'].sort_values(ascending=False)
plt.scatter(store['user_rating'], store['rating_count_ver'])
plt.xlabel("user_rating")
plt.ylabel("rating_count")
plt.show()


s = store.prime_genre.value_counts().index[:4]


def category(val):
    if val in s:
        return val
    else:
        return "Others"


store['broad_genre'] = store.prime_genre.apply(lambda x: category(x))
free = store[store.price == 0].broad_genre.value_counts().sort_index().to_frame()
paid = store[store.price > 0].broad_genre.value_counts().sort_index().to_frame()
total = store.broad_genre.value_counts().sort_index().to_frame()
free.columns = ['free']
paid.columns = ['paid']
total.columns = ['total']
dist = free.join(paid).join(total)
dist['paid_per'] = dist.paid*100/dist.total
dist['free_per'] = dist.free*100/dist.total

colors = ['blue', 'red', 'green', 'pink', 'skyblue']
plt.figure(figsize=(10, 10))
label_names = store.broad_genre.value_counts().sort_index().index
size = store.broad_genre.value_counts().sort_index().tolist()
circle = plt.Circle((0, 0), 0.5, color='black')
plt.pie(size, labels=label_names, colors=colors)
p = plt.gcf()
p.gca().add_artist(circle)
plt.show()
#
plt.figure(figsize=(10, 10))
f = pd.DataFrame(index=np.arange(0, 10, 2), data=dist.free.values, columns=['num'])
p = pd.DataFrame(index=np.arange(1, 11, 2), data=dist.paid.values, columns=['num'])
final = pd.concat([f, p], names=['labels']).sort_index()
final.num.tolist()

plt.figure(figsize=(20, 20))
group_names = store.broad_genre.value_counts().sort_index().index
group_size = store.broad_genre.value_counts().sort_index().tolist()
h = ['Free', 'Paid']
subgroup_names = 5*h
sub = ['#45cea2', '#fdd470']
subcolors = 5*sub
subgroup_size = final.num.tolist()

