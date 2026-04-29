import pandas as pd
import re
from collections import Counter

df = pd.read_parquet('train-00000-of-00001.parquet')
print(df.head())  # Displays the first 5 rows

# Extract all labels from the 'privacy' column
all_labels = re.findall(r"'label': ([A-Z_]+)", df['privacy'].str.cat(sep=' '))
label_counts = Counter(all_labels)

# Define label categories
categories = {
    'Personal Identity': [
        'FULL_NAME', 'FIRST_NAME', 'LAST_NAME', 'USERNAME', 'DATE_OF_BIRTH',
        'SSN', 'ID_CARD_NUMBER', 'PASSPORT_NUMBER', 'DRIVER_LICENSE_NUMBER',
        'EMPLOYEE_ID', 'CUSTOMER_ID', 'TAX_NUMBER'
    ],
    'Contact Information': ['EMAIL', 'PHONE_NUMBER'],
    'Address / Location': [
        'STREET_ADDRESS', 'STREET', 'CITY', 'ZIPCODE',
        'BUILDING_NUMBER', 'COORDINATES'
    ],
    'Financial': [
        'CREDIT_CARD_NUMBER', 'CREDIT_CARD_CVV', 'ACCOUNT_NUMBER',
        'BANK_ACCOUNT_NUMBER', 'IBAN', 'SWIFT_CODE',
        'ROUTING_NUMBER', 'PIN_NUMBER'
    ],
    'Temporal': ['DATE', 'TIME', 'DATETIME'],
    'Other': ['COMPANY', 'PASSWORD', 'API_KEY']
}

# Print overview
print("\n" + "=" * 50)
print("LABEL CATEGORY OVERVIEW")
print("=" * 50)

total = sum(label_counts.values())
print(f"\nTotal label occurrences: {total:,}")
print(f"Unique label types: {len(label_counts)}\n")

for category, labels in categories.items():
    print(f"--- {category} ---")
    for label in labels:
        if label in label_counts:
            print(f"  {label:<25} {label_counts[label]:>7,}")
    print()

# Show any labels not in the predefined categories
all_categorized = {l for labels in categories.values() for l in labels}
uncategorized = {l: c for l, c in label_counts.items() if l not in all_categorized}
if uncategorized:
    print("--- Uncategorized ---")
    for label, count in sorted(uncategorized.items(), key=lambda x: -x[1]):
        print(f"  {label:<25} {count:>7,}")
    print()
