import config
import google.generativeai as genai
import streamlit

genai.configure(api_key=config.GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Give me python code to sort a list")
print(response.text)