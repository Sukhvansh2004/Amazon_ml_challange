# Import the Python SDK
import google.generativeai as genai
genai.configure(api_key='AIzaSyBzgDyLj-8OavjfknFf6u3lLrMa_YuvjPA')

model = genai.GenerativeModel('gemini-1.5-pro')

from PIL import Image
img  = Image.open(r'new_image.jpg')
response = model.generate_content(img)
print(response.text)
result = genai.embed_content(
    model="models/text-embedding-004",
    content=response,
    task_type="retrieval_query")

# 1 input > 1 vector output
print(str(result['embedding'])[:50], '... TRIMMED]')