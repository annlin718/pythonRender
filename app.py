import os
import fitz
import requests
from flask import Flask, request, jsonify
from openai import OpenAI
from io import BytesIO
from docx import Document

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAX_CHARS = 6000

def chunk_text(text, max_chars=MAX_CHARS):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def analyze_chunk(chunk):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一個專業的文件分析 AI。"},
            {"role": "user", "content": f"請分析以下文件內容並摘要重點：\n{chunk}"}
        ]
    )
    return response.choices[0].message.content.strip()

def extract_pdf_text(stream):
    doc = fitz.open(stream=stream, filetype="pdf")
    return "".join(page.get_text() for page in doc)

def extract_docx_text(stream):
    document = Document(stream)
    return "\n".join([para.text for para in document.paragraphs])

@app.route("/analyze", methods=["POST"])
def analyze_file():
    try:
        file_stream = None
        filename = None

        if 'file' in request.files:
            file = request.files['file']
            file_stream = BytesIO(file.read())
            filename = file.filename
        elif 'file_url' in request.form:
            file_url = request.form['file_url']
            response = requests.get(file_url)
            response.raise_for_status()
            file_stream = BytesIO(response.content)
            filename = file_url.split("/")[-1]
        else:
            return jsonify({"error": "請上傳檔案或提供 file_url"}), 400

        ext = filename.lower().split('.')[-1]
        if ext == 'pdf':
            text = extract_pdf_text(file_stream)
        elif ext == 'docx':
            text = extract_docx_text(file_stream)
        else:
            return jsonify({"error": "目前僅支援 PDF (.pdf) 與 Word (.docx) 檔案"}), 400

        chunks = chunk_text(text)
        summaries = [analyze_chunk(c) for c in chunks]
        return jsonify({"summaries": summaries})

    except Exception as e:
        return jsonify({"error": f"分析過程發生錯誤: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def home():
    return "🎉 GPT-4o 文件分析 API 正常啟動！支援 PDF (.pdf) 與 Word (.docx)，請使用 POST /analyze 上傳檔案或 file_url。"

if __name__ == "__main__":
    app.run()
