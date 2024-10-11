import pandas as pd
import re
from typing import List, Dict, Any
import json

def parse_ion(ion_str: str) -> List[str]:
    ions = [ion.strip() for ion in ion_str.split(',')]
    return [ion for ion in ions if ion.lower() not in ['non-selective', 'non selective', 'nonselective']]

def is_non_selective(ion_str: str) -> bool:
    return any(term in ion_str.lower() for term in ['non-selective', 'non selective', 'nonselective'])

def generate_ion_queries(ion_str: str, ion_channel_str: str) -> Dict[str, Dict[str, str]]:
    ions = parse_ion(ion_str)
    queries = {}
    
    if is_non_selective(ion_str):
        queries["query1"] = {"text": f"Is the `{ion_channel_str}` non-selective ion channel in terms of ion selectivity?"}
        for i, ion in enumerate(ions, start=2):
            queries[f"query1_{i}"] = {"text": f"Is there any evidence that `{ion}` is the ion selectivity of the `{ion_channel_str}` ion channel?"}
    else:
        for i, ion in enumerate(ions, start=1):
            key = "query1" if i == 1 else f"query1_{i}"
            queries[key] = {"text": f"Is there any evidence that `{ion}` is the ion selectivity of the `{ion_channel_str}` ion channel?"}
    
    return queries

def process_ion_channel_data(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    result = {}
    
    for index, row in df.iterrows():
        uniprot_id = row["Uniprot"].strip()
        ion = row["Ion"].strip()
        ionchannel_name = row["IonChannelName"].strip()
        ionchannel_symbol = row["IonChannelSymbol"].split("(")[0].strip()
        gate_mechanism = row["GateMechanism"].strip()
        
        alternative_names = [name.strip() for name in row["AlternativeNames"].split(',')] if pd.notna(row["AlternativeNames"]) and row["AlternativeNames"].strip() != '' else []
        
        print(f"Processing Uniprot Id {uniprot_id} ...")
        
        ion_channel_str = f"{ionchannel_symbol}"
        if alternative_names:
            ion_channel_str += f" Alternative names: {', '.join(alternative_names)}"
        
        ion_queries = generate_ion_queries(ion, ion_channel_str)
        gate_query = {"query2": {"text": f"Does this article provide evidence for `{gate_mechanism}` as the gating mechanism for the `{ion_channel_str}` ion channel?"}}
        
        result[uniprot_id] = {**ion_queries, **gate_query}
    
    return result

if __name__ == "__main__":
    df = pd.read_csv("human_IC_annotation.csv", dtype=object, encoding='utf-8')
    output = process_ion_channel_data(df)
    with open("ion_channel_queries.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)