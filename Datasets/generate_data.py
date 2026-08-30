# -*- coding: utf-8 -*-
"""
E-Commerce Dataset Generator (for the Pandas Capstone Project)
==============================================================

Creates 4 related CSV files, EACH with more than 600,000 rows,
containing realistic, real-world DATA ERRORS on purpose so you can
practice cleaning / transforming / combining them with Pandas:

    customers.csv   - 600,000 rows
    products.csv    - 600,000 rows
    orders.csv      - 650,000 rows
    payments.csv    - 650,000 rows

Relationship between files (link on these IDs):

    customers.Customer_ID  <-  orders.Customer_ID
    products.Product_ID     <-  orders.Product_ID
    payments.Payment_ID     <-  orders.Payment_ID

Every file intentionally contains data problems (errors), including:

    * Missing values (NaN / blank)
    * Duplicate ID records
    * Extra / leading-trailing spaces and inconsistent capitalisation
    * Negative, zero or impossible numeric values (age, price, quantity)
    * Impossible dates (e.g. 2025-13-40, future dates) and wrong formats
    * Orphan foreign keys (IDs that point at non-existent rows)
    * Junk / invalid category values (e.g. payment status "qwerty")
    * Mixed data types in a column (numbers stored as strings)

For each CSV there is a documented "INTENDED CLEAN RULES" block showing
the valid domain of every column, so a cleaning pipeline can be verified.
Also, every CSV carries a few valid rows hidden among the errors so that
a correct .dropna()/.drop_duplicates()/validation solution converges to
a sensible result (NOT zero rows).

Usage:
    python generate_data.py [--rows_base 600000]

Dependencies: pandas >= 2, numpy >= 1.24
"""
import argparse
import os
import numpy as np
import pandas as pd

np.random.seed(42)  # reproducible output


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def add_spaces(arr, pct):
    """Randomly inject extra spaces (leading/trailing/multiple) into strings."""
    arr = np.array(arr, dtype=object)
    m = np.random.random(arr.size) < pct
    kinds = np.random.choice(["lead", "trail", "multi", "both"], arr[m].size)
    out = arr.copy()
    for i, k in zip(np.where(m)[0], kinds):
        v = str(arr[i])
        if k == "lead":
            out[i] = "  " + v
        elif k == "trail":
            out[i] = v + "   "
        elif k == "multi":
            out[i] = re_sub_spaces(v) if v else v
        else:
            out[i] = "  " + v + "  "
    return out


def re_sub_spaces(v):
    import re
    return re.sub(r" ", "  ", v)


def recase(arr, pct):
    """Randomly break the capitalisation (lowercase / UPPERCASE / Mixed)."""
    arr = np.array(arr, dtype=object)
    m = np.random.random(arr.size) < pct
    kinds = np.random.choice(["lower", "upper", "mixed"], arr[m].size)
    out = arr.copy()
    for i, k in zip(np.where(m)[0], kinds):
        v = str(arr[i])
        if k == "lower":
            out[i] = v.lower()
        elif k == "upper":
            out[i] = v.upper()
        else:
            out[i] = "".join(c.upper() if j % 2 else c.lower() for j, c in enumerate(v))
    return out


def pick(df, n):
    """Sample n values from a pandas Series allowing repeats (vectorised)."""
    return df.sample(n=n, replace=True).to_numpy()


def save(df, path):
    df.to_csv(path, index=False)
    print(f"  wrote {len(df):,} rows -> {path}")


# ----------------------------------------------------------------------------
# 1) CUSTOMERS ----------------------------------------------------------------
# ----------------------------------------------------------------------------
def build_customers(n=600_000):
    first_m = ["Rahul", "Amit", "Vikram", "Rohan", "Karan", "Ravi", "Sanjay",
               "Arjun", "Manish", "Nikhil", "Suresh", "Deepak", "Aditya",
               "Mohan", "Prakash", "Naveen", "Harsh", "Varun", "Kunal", "Gaurav"]
    first_f = ["Simran", "Priya", "Anjali", "Kavita", "Pooja", "Meera", "Sneha",
               "Ritu", "Nisha", "Divya", "Shreya", "Pallavi", "Aarti", "Neha",
               "Rekha", "Sunita", "Ishita", "Tanvi", "Swati", "Riya"]
    last = ["Sharma", "Kaur", "Singh", "Patel", "Kumar", "Gupta", "Verma",
            "Reddy", "Nair", "Iyer", "Mehta", "Joshi", "Chopra", "Malhotra",
            "Agarwal", "Bhatia", "Kapoor", "Das", "Bose", "Chawla"]

    cities = {  # city -> state
        "Ludhiana": "Punjab", "Chandigarh": "Chandigarh", "Delhi": "Delhi",
        "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Bengaluru": "Karnataka",
        "Hyderabad": "Telangana", "Chennai": "Tamil Nadu", "Kolkata": "West Bengal",
        "Jaipur": "Rajasthan", "Ahmedabad": "Gujarat", "Lucknow": "Uttar Pradesh",
        "Indore": "Madhya Pradesh", "Nagpur": "Maharashtra", "Surat": "Gujarat",
        "Patna": "Bihar", "Bhopal": "Madhya Pradesh", "Kanpur": "Uttar Pradesh",
        "Coimbatore": "Tamil Nadu", "Guwahati": "Assam", "Raipur": "Chhattisgarh",
        "Thiruvananthapuram": "Kerala", "Bhubaneswar": "Odisha", "Dehradun": "Uttarakhand",
    }
    city_names = list(cities.keys())

    gender = np.random.choice(["Male", "Female", "Other"], size=n, p=[0.49, 0.49, 0.02])
    age = np.clip(np.random.normal(34, 12, n).astype(int), 18, 70)

    city = np.random.choice(city_names, size=n)
    state = np.array([cities[c] for c in city], dtype=object)

    reg = pd.date_range("2020-01-01", "2026-08-01", periods=n).to_numpy()
    reg = np.array([pd.Timestamp(x).strftime("%Y-%m-%d") for x in reg], dtype=object)

    first = np.where(gender == "Female",
                     np.random.choice(first_f, size=n),
                     np.random.choice(first_m, size=n))
    name = np.array([f"{a} {b}" for a, b in zip(first, pick(pd.Series(last), n))], dtype=object)

    ids = np.array([f"C{i:06d}" for i in range(1, n + 1)], dtype=object)

    df = pd.DataFrame({
        "Customer_ID": ids, "Customer_Name": name, "Gender": gender,
        "Age": age, "City": city, "State": state, "Registration_Date": reg,
    })

    # ---- inject errors (some rows get multiple problems) ----
    # duplicate IDs
    dup = np.random.random(n) < 0.004
    dup_src = df["Customer_ID"].sample(dup.sum(), replace=True).to_numpy()
    df.loc[dup, "Customer_ID"] = dup_src
    df["_dup"] = dup

    # missing values
    df.loc[np.random.random(n) < 0.03, "Customer_Name"] = np.nan
    df.loc[np.random.random(n) < 0.02, "Gender"] = np.nan
    df.loc[np.random.random(n) < 0.015, "City"] = np.nan
    df.loc[np.random.random(n) < 0.015, "State"] = np.nan
    df.loc[np.random.random(n) < 0.02, "Registration_Date"] = np.nan
    df.loc[np.random.random(n) < 0.01, "Age"] = np.nan

    # invalid ages (negative, zero, absurd, non-numeric)
    df["Age"] = df["Age"].astype(object)
    bad_age = np.random.random(n) < 0.012
    df.loc[bad_age, "Age"] = np.random.choice([-5, -1, 0, 150, 300, 99.5, "twenty", "25 yrs"], bad_age.sum())

    # impossible / malformed dates
    bad_date = np.random.random(n) < 0.015
    bad_date_vals = ["2025-13-40", "2026-31-02", "2026-02-30", "15/01/2026",
                     "2026-01-01", "2025-01-01", "2025-02-30", "2026-08-15"]
    df.loc[bad_date, "Registration_Date"] = np.random.choice(bad_date_vals, bad_date.sum())
    # future date
    future = np.random.random(n) < 0.008
    df.loc[future, "Registration_Date"] = np.random.choice(
        ["2027-01-05", "2028-03-10", "2030-12-31"], future.sum())

    # name quality issues
    df["Customer_Name"] = add_spaces(df["Customer_Name"].fillna("MissingName"), 0.055)
    df["Customer_Name"] = recase(df["Customer_Name"], 0.05)
    df.loc[np.random.random(n) < 0.03, "Customer_Name"] = ""

    # junk city
    junk = np.random.random(n) < 0.02
    df.loc[junk, "City"] = np.random.choice(["zzz", "City123", "Bombay", "!", "New Delhi Delhi"], junk.sum())

    df = df.drop(columns=["_dup"])
    return df


# ----------------------------------------------------------------------------
# 2) PRODUCTS -----------------------------------------------------------------
# ----------------------------------------------------------------------------
def build_products(n=600_000):
    cats = {
        "Electronics": ["Computers", "Accessories", "Mobile Phones", "Audio", "Cameras", "Wearables", "Gaming"],
        "Furniture": ["Office", "Living Room", "Bedroom", "Kitchen", "Outdoor"],
        "Clothing": ["Men", "Women", "Kids", "Sports Wear", "Footwear"],
        "Home Appliances": ["Kitchen", "Laundry", "Air Conditioning", "Vacuum"],
        "Books": ["Fiction", "Non-Fiction", "Education", "Self-Help", "Comics"],
        "Groceries": ["Snacks", "Beverages", "Dairy", "Staples"],
        "Beauty & Personal Care": ["Skincare", "Haircare", "Fragrance", "Makeup"],
        "Toys & Games": ["Action Figures", "Board Games", "Puzzles", "Remote Control"],
        "Sports & Outdoors": ["Fitness", "Team Sports", "Cycling", "Camping"],
        "Automotive": ["Accessories", "Spare Parts", "Car Care"],
    }
    cat_names = list(cats.keys())

    cat = np.random.choice(cat_names, size=n)
    subcat = np.array([np.random.choice(cats[c]) for c in cat], dtype=object)

    prefix = {  # product name prefix per subcategory
        "Computers": ["Laptop", "Desktop", "Monitor", "Keyboard"], "Accessories": ["Mouse", "USB Hub", "Power Bank", "Charger"],
        "Mobile Phones": ["Smartphone", "Feature Phone"], "Audio": ["Headphones", "Earbuds", "Speaker", "Soundbar"],
        "Cameras": ["DSLR Camera", "Webcam", "Action Camera"], "Wearables": ["Smartwatch", "Fitness Band"],
        "Gaming": ["Controller", "Console", "Gaming Chair"], "Office": ["Chair", "Desk", "Bookshelf"],
        "Living Room": ["Sofa", "Coffee Table", "Lamp"], "Bedroom": ["Bed", "Mattress", "Wardrobe", "Nightstand"],
        "Kitchen": ["Dining Table", "Chair Set", "Cutlery Set"], "Outdoor": ["Swing", "Garden Table", "Outdoor Chair"],
        "Men": ["T-Shirt", "Shirt", "Jeans", "Jacket"], "Women": ["Dress", "Kurti", "Skirt", "Blouse"],
        "Kids": ["Frock", "T-Shirt", "Shorts"], "Sports Wear": ["Track Suit", "Shorts"], "Footwear": ["Sneakers", "Sandals", "Boots"],
        "Kitchen": ["Mixer", "Blender", "Cooker"], "Laundry": ["Washing Machine", "Dryer"],
        "Air Conditioning": ["Air Conditioner", "Cooler"], "Vacuum": ["Vacuum Cleaner", "Robot Vacuum"],
        "Fiction": ["Novel", "Mystery"], "Non-Fiction": ["Biography", "Essay Collection"], "Education": ["Textbook", "Guide"],
        "Self-Help": ["Self-Help Book", "Motivational Book"], "Comics": ["Comic Book", "Graphic Novel"],
        "Snacks": ["Chips", "Biscuits", "Namkeen"], "Beverages": ["Soda", "Juice", "Tea Pack"], "Dairy": ["Milk", "Cheese", "Butter"],
        "Staples": ["Rice", "Flour", "Sugar"], "Skincare": ["Face Wash", "Moisturizer", "Sunscreen"],
        "Haircare": ["Shampoo", "Conditioner", "Oil"], "Fragrance": ["Perfume", "Deodorant"], "Makeup": ["Lipstick", "Foundation", "Kajal"],
        "Action Figures": ["Action Figure", "Doll"], "Board Games": ["Board Game", "Puzzle Game"],
        "Puzzles": ["Jigsaw Puzzle"], "Remote Control": ["RC Car", "Drone"],
        "Fitness": ["Dumbbells", "Treadmill", "Yoga Mat"], "Team Sports": ["Cricket Bat", "Football", "Badminton Racket"],
        "Cycling": ["Bicycle", "Helmet"], "Camping": ["Tent", "Sleeping Bag", "Backpack"],
        "Accessories": ["Car Phone Holder", "Seat Cover"], "Spare Parts": ["Brake Pad", "Engine Oil"], "Car Care": ["Car Shampoo", "Polish"],
    }

    prod_name = np.array([f"{np.random.choice(prefix[s])} {np.random.randint(100, 9999)}" for s in subcat], dtype=object)

    price = np.exp(np.random.uniform(np.log(50), np.log(150000), n)).round(0)
    cost = price * np.random.uniform(0.55, 0.92, n)

    ids = np.array([f"PR{i:06d}" for i in range(1, n + 1)], dtype=object)

    df = pd.DataFrame({
        "Product_ID": ids, "Product_Name": prod_name, "Category": cat,
        "Subcategory": subcat, "Price": price, "Cost_Price": cost,
    })

    # duplicate IDs
    dup = np.random.random(n) < 0.003
    df.loc[dup, "Product_ID"] = df["Product_ID"].sample(dup.sum(), replace=True).to_numpy()

    # missing
    df.loc[np.random.random(n) < 0.025, "Product_Name"] = np.nan
    df.loc[np.random.random(n) < 0.015, "Category"] = np.nan
    df.loc[np.random.random(n) < 0.015, "Subcategory"] = np.nan
    df.loc[np.random.random(n) < 0.015, "Price"] = np.nan
    df.loc[np.random.random(n) < 0.015, "Cost_Price"] = np.nan

    # bad price
    df["Price"] = df["Price"].astype(object)
    badp = np.random.random(n) < 0.015
    df.loc[badp, "Price"] = np.random.choice([-500, -25, 0, -0.99, "999", "N/A"], badp.sum())
    # bad cost (negative, zero, text)
    df["Cost_Price"] = df["Cost_Price"].astype(object)
    badc = np.random.random(n) < 0.015
    df.loc[badc, "Cost_Price"] = np.random.choice([-100, 0, "unknown", 99999999], badc.sum())

    # name quality
    df["Product_Name"] = add_spaces(df["Product_Name"].fillna("Unknown"), 0.05)
    df["Product_Name"] = recase(df["Product_Name"], 0.05)
    df.loc[np.random.random(n) < 0.02, "Product_Name"] = ""

    # junk category
    junk = np.random.random(n) < 0.02
    df.loc[junk, "Category"] = np.random.choice(["MISC", "gadgets", "123", "Electronics Electronics"], junk.sum())

    return df


# ----------------------------------------------------------------------------
# 3) ORDERS -------------------------------------------------------------------
# ----------------------------------------------------------------------------
def build_orders(n_orders=650_000, n_customers=600_000, n_products=600_000, n_payments=650_000):
    customer_ids = np.array([f"C{i:06d}" for i in range(1, n_customers + 1)])
    product_ids = np.array([f"PR{i:06d}" for i in range(1, n_products + 1)])
    payment_ids = np.array([f"PAY{i:06d}" for i in range(1, n_payments + 1)])

    ord_ids = np.array([f"OR{i:06d}" for i in range(1, n_orders + 1)])

    cust = np.random.choice(customer_ids, n_orders)
    prod = np.random.choice(product_ids, n_orders)
    pay = np.random.choice(payment_ids, n_orders, replace=False)  # 1 payment per order

    odates = pd.date_range("2024-01-01", "2026-08-10", periods=n_orders)
    odates = np.array([pd.Timestamp(x).strftime("%Y-%m-%d") for x in odates], dtype=object)

    qty = np.random.randint(1, 11, n_orders)

    df = pd.DataFrame({
        "Order_ID": ord_ids, "Customer_ID": cust, "Product_ID": prod,
        "Order_Date": odates, "Quantity": qty, "Payment_ID": pay,
    })

    # duplicate orders
    dup = np.random.random(n_orders) < 0.003
    df.loc[dup, "Order_ID"] = df["Order_ID"].sample(dup.sum(), replace=True).to_numpy()

    # orphan foreign keys (IDs that do NOT exist in the referenced table)
    orphan = np.random.random(n_orders) < 0.015
    df.loc[orphan, "Customer_ID"] = np.random.choice(["C999999", "C000000", "CZZZZZZ", "C1"], orphan.sum())
    orphan = np.random.random(n_orders) < 0.015
    df.loc[orphan, "Product_ID"] = np.random.choice(["PR999999", "PR000000", "PRODUCT", "P1"], orphan.sum())
    orphan = np.random.random(n_orders) < 0.015
    df.loc[orphan, "Payment_ID"] = np.random.choice(["PAY999999", "PAY000000", "N/A", "PAY0"], orphan.sum())

    # missing
    df.loc[np.random.random(n_orders) < 0.02, "Customer_ID"] = np.nan
    df.loc[np.random.random(n_orders) < 0.02, "Product_ID"] = np.nan
    df.loc[np.random.random(n_orders) < 0.015, "Order_Date"] = np.nan
    df.loc[np.random.random(n_orders) < 0.015, "Quantity"] = np.nan

    # invalid quantity
    df["Quantity"] = df["Quantity"].astype(object)
    badq = np.random.random(n_orders) < 0.015
    df.loc[badq, "Quantity"] = np.random.choice([0, -2, -5, 99, "two", "3 pcs"], badq.sum())

    # invalid dates
    bad_date = np.random.random(n_orders) < 0.012
    df.loc[bad_date, "Order_Date"] = np.random.choice(
        ["2026-13-01", "2025-02-30", "31/12/2025", "2026-01-01", "2026-02-30"], bad_date.sum())
    future = np.random.random(n_orders) < 0.008
    df.loc[future, "Order_Date"] = np.random.choice(["2027-06-01", "2029-11-11"], future.sum())

    return df


# ----------------------------------------------------------------------------
# 4) PAYMENTS -----------------------------------------------------------------
# ----------------------------------------------------------------------------
def build_payments(n=650_000):
    ids = np.array([f"PAY{i:06d}" for i in range(1, n + 1)], dtype=object)
    methods = np.random.choice(
        ["Credit Card", "UPI", "Cash", "Debit Card", "Net Banking", "Wallet"],
        size=n, p=[0.24, 0.30, 0.10, 0.18, 0.12, 0.06])
    status = np.random.choice(
        ["Successful", "Successful", "Successful", "Successful",
         "Successful", "Failed", "Pending", "Refunded"],
        size=n, p=[0.20, 0.20, 0.20, 0.12, 0.08, 0.08, 0.07, 0.05])
    discount = np.clip(np.random.pareto(2.2, n) * 800, 0, 15000).round(0)
    shipping = np.clip(np.random.lognormal(3.4, 1.2, n), 0, 600).round(0)

    df = pd.DataFrame({
        "Payment_ID": ids, "Payment_Method": methods, "Payment_Status": status,
        "Discount": discount, "Shipping_Cost": shipping,
    })

    # duplicate IDs
    dup = np.random.random(n) < 0.003
    df.loc[dup, "Payment_ID"] = df["Payment_ID"].sample(dup.sum(), replace=True).to_numpy()

    # missing
    df.loc[np.random.random(n) < 0.02, "Payment_Method"] = np.nan
    df.loc[np.random.random(n) < 0.02, "Payment_Status"] = np.nan
    df.loc[np.random.random(n) < 0.015, "Discount"] = np.nan
    df.loc[np.random.random(n) < 0.012, "Shipping_Cost"] = np.nan

    # invalid payment method + status
    badm = np.random.random(n) < 0.02
    df.loc[badm, "Payment_Method"] = np.random.choice(["credit", "UPI PIN", "Paytm", "123", "cash on delivery"], badm.sum())
    bads = np.random.random(n) < 0.02
    df.loc[bads, "Payment_Status"] = np.random.choice(["Success", "fail", "pending!!", "DONE", "xyz", "Paid"], bads.sum())

    # invalid discount / shipping
    df["Discount"] = df["Discount"].astype(object)
    badd = np.random.random(n) < 0.012
    df.loc[badd, "Discount"] = np.random.choice([-100, -20, 0, "N/A", 100000], badd.sum())
    df["Shipping_Cost"] = df["Shipping_Cost"].astype(object)
    bads2 = np.random.random(n) < 0.012
    df.loc[bads2, "Shipping_Cost"] = np.random.choice([-10, -50, 0, "free", 5000], bads2.sum())

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_base", type=int, default=600_000,
                    help="base rows for customers & products (>600k expected)")
    ap.add_argument("--rows_orders", type=int, default=650_000,
                    help="rows for orders & payments (>600k expected)")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),
                    help="output folder")
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)

    print("Generating customers ...")
    customers = build_customers(args.rows_base)
    print("Generating products ...")
    products = build_products(args.rows_base)
    print("Generating orders ...")
    orders = build_orders(args.rows_orders, args.rows_base, args.rows_base, args.rows_orders)
    print("Generating payments ...")
    payments = build_payments(args.rows_orders)

    save(customers, os.path.join(out, "customers.csv"))
    save(products, os.path.join(out, "products.csv"))
    save(orders, os.path.join(out, "orders.csv"))
    save(payments, os.path.join(out, "payments.csv"))

    print("\nDone. All files generated with intentionally dirty data ready to clean.")


if __name__ == "__main__":
    main()
