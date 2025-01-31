import random
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd


def generate_random_transactions(num_transactions=10, num_items=5):
    items = [
        "Milk",
        "Bread",
        "Saffron",
        "Butter",
        "Wafer",
        "Cheese",
        "Eggs",
        "Juice",
        "Sugar",
        "Tea",
    ]
    transactions = []

    for _ in range(num_transactions):
        transaction_length = random.randint(2, num_items)
        transaction = random.sample(items, transaction_length)
        transactions.append(transaction)

    return transactions


def create_one_hot_encoded_df(transactions, items):
    encoded_vals = []
    for transaction in transactions:
        item_flags = [1 if item in transaction else 0 for item in items]
        encoded_vals.append(item_flags)

    df = pd.DataFrame(encoded_vals, columns=items)
    return df.astype(bool)


transactions = [
    ["Wafer", "Butter"],
    ["Tea", "Wafer", "Saffron"],
    ["Wafer", "Sugar", "Bread"],
    ["Milk", "Eggs", "Saffron", "Cheese", "Butter"],
    ["Saffron", "Cheese"],
    ["Saffron", "Cheese", "Wafer", "Eggs"],
    ["Milk", "Tea", "Bread", "Cheese", "Sugar"],
    ["Wafer", "Saffron"],
    ["Eggs", "Milk"],
    ["Saffron", "Wafer", "Milk"],
]


all_items = sorted(
    list(set(item for transaction in transactions for item in transaction))
)

df = create_one_hot_encoded_df(transactions, all_items)

min_support = 0.3
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

print(f"\nFrequent Itemsets (Support >= {min_support}):")

frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
print(frequent_itemsets)
print("\n" + "-" * 50)

min_confidence = 0.6
rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=min_confidence
)

print(f"\nAssociation Rules (Confidence >= {min_confidence}):")
if len(rules) > 0:

    rules = rules.sort_values(by=["confidence", "lift"], ascending=[False, False])
    for idx, row in rules.iterrows():
        antecedents = ", ".join(list(row["antecedents"]))
        consequents = ", ".join(list(row["consequents"]))
        print(f"Rule: {antecedents} -> {consequents}")
        print(f"Support: {row['support']:.3f}")
        print(f"Confidence: {row['confidence']:.3f}")
        print(f"Lift: {row['lift']:.3f}\n")
else:
    print("No rules found meeting the confidence threshold.")
