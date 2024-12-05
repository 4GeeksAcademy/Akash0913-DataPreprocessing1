from utils import db_connect
engine = db_connect()

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
print(total_data.head())

os.makedirs("../data/raw", exist_ok=True)
total_data.to_csv("../data/raw/total_data.csv", index=False)

print(total_data.shape)
print(total_data.info())
print(f"The number of duplicated Name records is: {total_data['name'].duplicated().sum()}")
print(f"The number of duplicated Host ID records is: {total_data['host_id'].duplicated().sum()}")
print(f"The number of duplicated ID records is: {total_data['id'].duplicated().sum()}")

nan_counts = total_data.isnull().sum().sort_values(ascending=False)
print("NaN counts in each column:\n", nan_counts)

total_data.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis=1, inplace=True)
print(total_data.head())

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

sns.boxplot(ax=axes[0, 0], data=total_data, y="neighbourhood_group")
sns.boxplot(ax=axes[0, 1], data=total_data, y="price")
sns.boxplot(ax=axes[0, 2], data=total_data, y="minimum_nights")
sns.boxplot(ax=axes[1, 0], data=total_data, y="number_of_reviews")
sns.boxplot(ax=axes[1, 1], data=total_data, y="calculated_host_listings_count")
sns.boxplot(ax=axes[1, 2], data=total_data, y="availability_365")
sns.boxplot(ax=axes[2, 0], data=total_data, y="room_type")

plt.tight_layout()
plt.savefig("boxplots.png")
print("Saved: boxplots.png")

price_stats = total_data["price"].describe()
print("Price Stats:\n", price_stats)

price_iqr = price_stats["75%"] - price_stats["25%"]
upper_limit = price_stats["75%"] + 1.5 * price_iqr
lower_limit = price_stats["25%"] - 1.5 * price_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(price_iqr, 2)}")

total_data = total_data[total_data["price"] > 0]
count_0 = total_data[total_data["price"] == 0].shape[0]
count_1 = total_data[total_data["price"] == 1].shape[0]

nights_stats = total_data["minimum_nights"].describe()
print("Minimum Nights Stats:\n", nights_stats)

total_data = total_data[total_data["minimum_nights"] <= 15]
count_0 = total_data[total_data["minimum_nights"] == 0].shape[0]
count_1 = total_data[total_data["minimum_nights"] == 1].shape[0]
count_2 = total_data[total_data["minimum_nights"] == 2].shape[0]
count_3 = total_data[total_data["minimum_nights"] == 3].shape[0]
count_4 = total_data[total_data["minimum_nights"] == 4].shape[0]

print("Count of Minimum Nights:")
print("Count of 0: ", count_0)
print("Count of 1: ", count_1)
print("Count of 2: ", count_2)
print("Count of 3: ", count_3)
print("Count of 4: ", count_4)

os.makedirs("../data/processed", exist_ok=True)
total_data.to_csv("../data/processed/cleaned_total_data.csv", index=False)
print("Cleaned data saved to ../data/processed/cleaned_total_data.csv")

total_data["neighbourhood_group"] = pd.factorize(total_data["neighbourhood_group"])[0]
total_data["room_type"] = pd.factorize(total_data["room_type"])[0]

num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", "availability_365", "neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(total_data[num_variables])
df_scal = pd.DataFrame(scal_features, index=total_data.index, columns=num_variables)
df_scal["price"] = total_data["price"]

X = df_scal.drop("price", axis=1)
y = df_scal["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selection_model = SelectKBest(chi2, k=4)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns=X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns=X_test.columns.values[ix])

X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)

X_train_sel.to_csv("../data/processed/clean_train.csv", index=False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index=False)
print("Processed train and test datasets saved.")