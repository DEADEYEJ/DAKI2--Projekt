import pandas as pd
import re
from collections import Counter

# Load data
df = pd.read_parquet("train-00000-of-00001.parquet")

# Regex til emails
pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" 

#  1. Find emails med regex
predicted_emails = []

for col in df.columns:
    for value in df[col].astype(str):
        found = re.findall(pattern, value)
        predicted_emails.extend(found)

#  2. Hent de rigtige emails (ground truth)
true_emails = []

for row in df["privacy"]:
    for item in row:
        if item["label"] == "EMAIL":
            true_emails.append(item["value"])

# 3. Lav sets (til recall)
predicted_set = set(predicted_emails)
true_set = set(true_emails)

#  4. Beregn recall
tp = len(predicted_set & true_set)
fn = len(true_set - predicted_set)

recall = tp / (tp + fn) if (tp + fn) > 0 else 0

#  5. -------- DUBLETTER --------
pred_counter = Counter(predicted_emails)
true_counter = Counter(true_emails)

# Ekstra dubletter (det tal du leder efter)
pred_duplicates = len(predicted_emails) - len(predicted_set)
true_duplicates = len(true_emails) - len(true_set)

# Antal unikke emails der er dubletter
pred_duplicate_keys = len([k for k, v in pred_counter.items() if v > 1])
true_duplicate_keys = len([k for k, v in true_counter.items() if v > 1])

#  6. Print ALT
print("-------- RESULTATER --------")

print("\n--- RECALL ---")
print("Fundet emails (regex):", len(predicted_set))
print("Emails i datasættet:", len(true_set))
print("True Positives:", tp)
print("False Negatives:", fn)
print("Recall:", recall)
print("Recall (%):", recall * 100)

print("\n--- DUBLETTER ---")
print("Total predicted emails:", len(predicted_emails))
print("Unikke predicted emails:", len(predicted_set))
print("Dubletter (predicted):", pred_duplicates)

print("\nTotal true emails:", len(true_emails))
print("Unikke true emails:", len(true_set))
print("Dubletter (true):", true_duplicates)
