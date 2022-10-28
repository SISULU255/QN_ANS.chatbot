from transformers import pipeline
import pandas as pd
import streamlit as st
# prepare table + question
data = {"Institute": ["Kiitec"], "CONTACT INFO of kiitec": ["Kiitec contact info\nP.O.Box 3172 Arusha, Tanzania.\nPhone: +255 27 250 4384\nMobile: +255 757 845 118\nEmail: info@kiitec.ac.tz"], "Location": ["is located in Moshono, Arusha next to Masai Camp."], "Description of kiitec": [" is a technical institution registered by NACTE (REG/EOS/027) based in Moshono, Arusha next to Masai Camp. "], "Registration": ["Kiitec institution is registered by NACTE (REG/EOS/027)"], "Courses or programmes": ["Electronics & Telecommunication Engineering ,Industrial Automation\nComputer Engineering and Networking\nRenewable Energies, Environmental Impact\nFuture training programs or courses in development: Biomedical\nAvionics."], "Fee amount for diploma course": ["for first semester in diploma amount is 695,000 and  625,000 \n in second semister  can be paid in two installment "], "Fees payment info": ["The fees should be paid through the BANK of ABSA\n and Account No is: 002-4001687\nAccount Name: KIITEC Ltd "], "Uniform information ": ["The dressing code of kiitec institution are dark blue trouser, light blue shirts for men and dark blue skirts, light blue shirts for women "]}
table = pd.DataFrame.from_dict(data)
query = st.text_input('Question', 'what is  kiitec?')

# pipeline model
# Note: you must to install torch-scatter first.
tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")

# result

print(tqa(table=table, query=query)['cells'][0])
#53
