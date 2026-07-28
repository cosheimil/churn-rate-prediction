import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043

customer_ids = [f"{np.random.randint(1000,9999)}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(N)]

gender = np.random.choice(["Male", "Female"], N, p=[0.505, 0.495])
senior_citizen = np.random.choice([0, 1], N, p=[0.838, 0.162])

partner = np.where(
    senior_citizen == 1,
    np.random.choice(["Yes", "No"], N, p=[0.55, 0.45]),
    np.random.choice(["Yes", "No"], N, p=[0.45, 0.55])
)
dependents = np.where(
    (senior_citizen == 1) | (partner == "Yes"),
    np.random.choice(["Yes", "No"], N, p=[0.25, 0.75]),
    np.random.choice(["Yes", "No"], N, p=[0.10, 0.90])
)

tenure = np.random.randint(0, 73, N)
tenure = np.where(
    senior_citizen == 1,
    tenure + np.random.randint(0, 15, N),
    tenure
)
tenure = np.clip(tenure, 0, 72)

phone_service = np.random.choice(["Yes", "No"], N, p=[0.903, 0.097])
multiple_lines = np.array(["No" for _ in range(N)])
for i in range(N):
    if phone_service[i] == "Yes":
        multiple_lines[i] = np.random.choice(["Yes", "No"], p=[0.42, 0.58])
    else:
        multiple_lines[i] = "No phone service"

internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.345, 0.440, 0.215])
has_internet = (internet_service != "No")

services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
service_data = {}
for svc in services:
    p_yes_base = 0.35 if svc not in ["StreamingTV", "StreamingMovies"] else 0.42
    service_data[svc] = np.where(
        has_internet,
        np.random.choice(["Yes", "No", "No internet service"], N, p=[p_yes_base, 1.0 - p_yes_base, 0.0]),
        np.random.choice(["Yes", "No", "No internet service"], N, p=[0.0, 0.0, 1.0])
    )

contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21])
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.596, 0.404])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.337, 0.228, 0.218, 0.217]
)

monthly_charges = np.random.uniform(18.25, 118.75, N)
monthly_charges = np.where(
    internet_service == "Fiber optic",
    monthly_charges + np.random.uniform(10, 25, N),
    monthly_charges
)
monthly_charges = np.clip(monthly_charges, 18.25, 119.50)
monthly_charges = np.round(monthly_charges, 2)

total_charges_raw = tenure.astype(float) * monthly_charges
total_charges = np.where(tenure == 0, 0.0, np.round(total_charges_raw + np.random.normal(0, 50, N), 2))
total_charges = np.maximum(total_charges, 0)
total_charges = np.round(total_charges, 2)

churn_prob = np.full(N, 0.265)
churn_prob = np.where(senior_citizen == 1, churn_prob + 0.15, churn_prob)
churn_prob = np.where(tenure < 12, churn_prob + 0.25, churn_prob)
churn_prob = np.where(tenure > 36, churn_prob - 0.12, churn_prob)
churn_prob = np.where(contract == "Month-to-month", churn_prob + 0.18, churn_prob)
churn_prob = np.where(contract == "Two year", churn_prob - 0.10, churn_prob)
churn_prob = np.where(internet_service == "Fiber optic", churn_prob + 0.08, churn_prob)
churn_prob = np.where(service_data["TechSupport"] == "No", churn_prob + 0.06, churn_prob)
churn_prob = np.where(service_data["OnlineSecurity"] == "No", churn_prob + 0.05, churn_prob)
churn_prob = np.where(payment_method == "Electronic check", churn_prob + 0.08, churn_prob)
churn_prob = np.where(paperless_billing == "Yes", churn_prob + 0.04, churn_prob)
churn_prob = np.clip(churn_prob, 0.02, 0.95)

churn = np.where(np.random.random(N) < churn_prob, "Yes", "No")

churn_rate = (churn == "Yes").mean()
print(f"Generated {N} records. Churn rate: {churn_rate:.2%}")

df = pd.DataFrame({
    "customerID": customer_ids,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
})
for svc in services:
    df[svc] = service_data[svc]
df["Contract"] = contract
df["PaperlessBilling"] = paperless_billing
df["PaymentMethod"] = payment_method
df["MonthlyCharges"] = monthly_charges
df["TotalCharges"] = total_charges
df["Churn"] = churn

output_path = "data/raw/telco_churn.csv"
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print(df.head(3).to_string())
