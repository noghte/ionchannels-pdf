import pandas as pd

# Load the Excel file
excel_file = 'human_IC_annotation_details.xlsx'

# Read the Excel file, skipping the first three rows (headers)
df = pd.read_excel(excel_file, skiprows=2, sheet_name=3)

# Create a new DataFrame with the desired columns and mapping
df_csv = pd.DataFrame({
    'Uniprot': df.iloc[:, 0].str.strip(),
    'IonChannelName': df.iloc[:, 1].str.strip(),
    'AlternativeNames': '',  # Empty for now
    'IonChannelSymbol': df.iloc[:, 2].str.strip(),
    'Ion': df.iloc[:, 10].str.strip(),
    'Family': df.iloc[:, 8].str.strip(),
    'GateMechanism': df.iloc[:, 11].str.strip(),
    'PubMed': df.iloc[:, 12].str.strip()
})

# Save the DataFrame to a CSV file with the appropriate settings
df_csv.to_csv('human_IC_annotation.csv', index=False)
print("CSV file saved successfully.")

# Create the sample CSV file with specific Uniprot values
sample_uniprot_values = [
    'Q14722', 'Q96RP8', 'B7ZAQ6', 'Q9Y696', 'Q9BQ31', 
    'Q70Z44', 'Q9H3M0', 'Q9Y5S1', 'Q02641', 'Q13303'
]
df_sample = df_csv[df_csv['Uniprot'].isin(sample_uniprot_values)]
df_sample.to_csv('./human_IC_annotation_sample.csv', index=False)
print("Sample CSV file saved successfully.")