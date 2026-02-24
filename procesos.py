from PyPDF2 import PdfReader
import os

def extract_text(directory_path):
    pdf_texts = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PdfReader(file)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text()
                pdf_texts[filename] = text
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return pdf_texts

# Usage example:
directory = '.\\Contratos'
texts = extract_text(directory)
for filename, content in texts.items():
    print(f"--- {filename} ---")
    print(content)