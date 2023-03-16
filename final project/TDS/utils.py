import numpy as np
import pandas as pd
# data visualizations
import seaborn as sns
import matplotlib.pyplot as plt
# Machine learning library
from sklearn.model_selection import train_test_split

# Machine Learning Extensions - Apriori & Association Rules
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def display_data(data):
    print("\033[1mdataset information:\033[0m\n")
    print(data.info())
    print("\n")
    print("\033[1mVerify if there are any null values:\033[0m\n")
    print(data.isnull().sum().sort_values(ascending=False))


def top_ten_sold_items(data):
    ## Creating distribution of Item Sold
    item_desc = "itemDescription"
    freq = 'Frequency'
    Item_distr = data.groupby(by=item_desc).size().reset_index(name=freq).sort_values(by=freq, ascending=False).head(10)

    ## Declaring variables

    bars = Item_distr[item_desc]
    height = Item_distr[freq]
    x_pos = np.arange(len(bars))

    ## Defining Figure Size

    plt.figure(figsize=(15, 7))

    # Create bars
    sns.set_style('whitegrid')
    plt.bar(x_pos, height, color='skyblue')

    # Add title and axis names
    plt.title("Top 10 Sold Items")
    plt.xlabel("Item Name")
    plt.ylabel("Number of Quantity Sold")

    # Create names on the x-axis
    plt.xticks(x_pos, bars)

    # Show graph
    plt.show()


def encode_data(data):
    if data <= 0:
        return 0
    else:
        return 1


def convert_numbers_to_one_or_zero(data):
    return data.applymap(encode_data)


def build_records(data, column1, column2, items):
    records = data.groupby([column1, column2])[items[:]].sum()
    records = records.reset_index()[items]
    return records


def one_hot(data, column):
    one_hot = pd.get_dummies(data[column])
    data.drop(column, inplace=True, axis=1)
    data = data.join(one_hot)
    return data


def train_test(df):
    return train_test_split(df, test_size=0.2)


def frequent_itemsets_and_association_rules(df, support):
    # Apply the Apriori algorithm to the entire dataset
    frequent_itemsets = apriori(df, min_support=support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
    return frequent_itemsets, rules


def calculate_performance(rules_test, rules, test):
    precision = len(rules_test) / len(rules)
    recall = len(rules_test) / len(test)
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def plot_results(labels, values):
    plt.bar(labels, values)
    plt.ylabel('Value')
    plt.title('Comparison between Apriori on entire dataset and on training set, and performance on test set')
    plt.show()


