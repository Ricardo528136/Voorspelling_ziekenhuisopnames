# Stap 1: Importeren van bibliotheken
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Stap 2: Dataset inladen
file_path = "Source/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_path)

# Stap 3: Eerste verkenning
print("Eerste 5 rijen:")
display(df.head())

print("Dimensies van de dataset:")
print(df.shape)

print("Beschrijving van numerieke variabelen:")
display(df.describe())

print("Aantal unieke waarden per kolom:")
print(df.nunique())

# Stap 4: Controle op null-waarden
print("Ontbrekende waarden per kolom:")
print(df.isnull().sum())

# Stap 5: Basis visualisaties
# Distributie van de target
plt.figure(figsize=(6,4))
sns.countplot(x='Diabetes_binary', data=df)
plt.title("Distributie van diabetes (proxy voor ziekenhuisopname)")
plt.xticks([0,1], ['Geen diabetes', 'Diabetes'])
plt.ylabel("Aantal")
plt.show()

# Relatie tussen BMI en diabetes
plt.figure(figsize=(8,5))
sns.boxplot(x='Diabetes_binary', y='BMI', data=df)
plt.title("BMI per groep (wel/geen diabetes)")
plt.xticks([0,1], ['Geen diabetes', 'Diabetes'])
plt.show()

# Correlatiematrix
plt.figure(figsize=(12,10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlatiematrix van features")
plt.show()