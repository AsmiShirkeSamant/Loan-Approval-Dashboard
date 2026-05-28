import numpy as np
import pandas as pd

np.random.seed(42)

n = 45000

#demographics
ages = np.random.gamma(shape=4, scale=7, size=n) + 18
ages = np.clip(ages, 18, 75).astype(int)

genders = np.random.choice(['Male', 'Female', 'Non-binary'], size=n, p=[0.495, 0.495, 0.01])

education_levels = ['High School', 'Certificate/Diploma', 'Bachelor', 'Postgraduate', 'Doctorate']
education_probs   = [0.22, 0.25, 0.33, 0.17, 0.03]
education = np.random.choice(education_levels, size=n, p=education_probs)

states = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT']
state_probs = [0.31, 0.26, 0.20, 0.11, 0.07, 0.02, 0.017, 0.013]
state = np.random.choice(states, size=n, p=state_probs)

#income 
income_base = np.random.lognormal(mean=11.0, sigma=0.55, size=n)
income = np.clip(income_base, 20000, 450000).astype(int)

emp_exp = np.random.gamma(shape=2.5, scale=3.5, size=n).astype(int)
emp_exp = np.clip(emp_exp, 0, ages - 18)

emp_status = np.random.choice(
    ['Full-time', 'Part-time', 'Self-employed', 'Casual', 'Unemployed'],
    size=n, p=[0.55, 0.18, 0.12, 0.10, 0.05]
)

home_ownership = np.random.choice(
    ['RENT', 'MORTGAGE', 'OWN', 'FAMILY', 'OTHER'],
    size=n, p=[0.31, 0.39, 0.18, 0.10, 0.02]
)

#HELP/HECS DEBT
has_help = np.zeros(n, dtype=int)
for i in range(n):
    prob = 0.0
    if education[i] in ['Bachelor', 'Postgraduate', 'Doctorate']:
        prob = 0.60
    elif education[i] == 'Certificate/Diploma':
        prob = 0.15
    if ages[i] > 45:
        prob *= 0.4
    has_help[i] = 1 if np.random.random() < prob else 0

help_debt = np.where(has_help == 1,
    np.random.lognormal(mean=9.8, sigma=0.6, size=n).clip(5000, 120000).astype(int),
    0)

#Loan Details
loan_amnt = np.random.lognormal(mean=9.7, sigma=0.65, size=n)
loan_amnt = np.clip(loan_amnt, 2000, 75000).round(-2).astype(int)  # round to nearest 100

loan_intent = np.random.choice(
    ['VEHICLE', 'DEBT_CONSOLIDATION', 'HOME_IMPROVEMENT', 'MEDICAL', 'HOLIDAY', 'EDUCATION', 'PERSONAL'],
    size=n, p=[0.37, 0.25, 0.14, 0.08, 0.06, 0.05, 0.05]
)

# Interest rates
loan_int_rate = np.random.normal(loc=13.87, scale=3.5, size=n)
loan_int_rate = np.clip(loan_int_rate, 5.5, 25.0).round(2)

# Loan term in months
loan_term_months = np.random.choice([12, 24, 36, 48, 60, 72, 84], size=n,
    p=[0.05, 0.15, 0.30, 0.20, 0.20, 0.07, 0.03])

# Date of application
application_date = pd.to_datetime(np.random.choice(pd.date_range('2022-01-01', '2025-12-31'), size=n))

# Credit score
credit_score = np.random.normal(loc=750, scale=160, size=n).astype(int)
credit_score = np.clip(credit_score, 0, 1200)

cred_hist_length = np.clip(ages - 18 - np.random.randint(0, 5, size=n), 0, None)

previous_defaults = np.random.choice(['Yes', 'No'], size=n, p=[0.18, 0.82])

#derived features
loan_percent_income = (loan_amnt / income).round(4)

# LOAN STATUS (used ai help)
# Factors: credit score, DTI, defaults, employment, income
def calc_approval_prob(i):
    score = credit_score[i]
    dti = loan_percent_income[i]
    default = previous_defaults[i]
    emp = emp_status[i]
    inc = income[i]
    rate = loan_int_rate[i]

    prob = 0.5

    # Credit score (Equifax bands)
    if score >= 853:   prob += 0.30
    elif score >= 735: prob += 0.18
    elif score >= 661: prob += 0.05
    elif score >= 460: prob -= 0.15
    else:              prob -= 0.35

    # DTI
    if dti < 0.15:    prob += 0.12
    elif dti < 0.35:  prob += 0.03
    elif dti < 0.50:  prob -= 0.10
    else:             prob -= 0.25

    # Defaults
    if default == 'Yes': prob -= 0.30

    # Employment
    if emp == 'Full-time':    prob += 0.10
    elif emp == 'Part-time':  prob += 0.02
    elif emp == 'Self-employed': prob -= 0.05
    elif emp == 'Casual':     prob -= 0.08
    elif emp == 'Unemployed': prob -= 0.30

    # Income
    if inc > 100000:  prob += 0.08
    elif inc > 60000: prob += 0.03
    elif inc < 30000: prob -= 0.15

    # HELP debt (minor negative signal for large debts)
    if help_debt[i] > 50000: prob -= 0.05

    return np.clip(prob, 0.02, 0.97)

probs = np.array([calc_approval_prob(i) for i in range(n)])
loan_status = (np.random.random(n) < probs).astype(int)

print('Approval rate:', loan_status.mean().round(3))

df = pd.DataFrame({
    'person_age': ages,
    'person_gender': genders,
    'person_education': education,
    'person_state': state,
    'person_income': income,
    'person_emp_status': emp_status,
    'person_emp_exp': emp_exp,
    'person_home_ownership': home_ownership,
    'has_help_debt': has_help,
    'help_debt_amount': help_debt,
    'loan_amnt': loan_amnt,
    'loan_intent': loan_intent,
    'loan_term_months': loan_term_months,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cred_hist_length,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': previous_defaults,
    'loan_status': loan_status,
    'application_date': application_date
})

from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
load_dotenv()
PATH = os.getenv('PATH')
df.to_csv(r'PATH\au_loan_data.csv', index=False)

#path check just in case
#print(os.getcwd())