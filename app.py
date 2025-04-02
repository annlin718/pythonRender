import os
import fitz
from flask import Flask, request, jsonify
from openai import OpenAI, OpenAIError

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
            {"role": "user", "content": f"請分析以下 PDF 內容並摘要重點：\n{chunk}"}
        ]
    )
    return response.choices[0].message.content.strip()

@app.route("/analyze", methods=["POST"])
def analyze_pdf():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "請上傳 PDF 檔案"}), 400

    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        chunks = chunk_text(text)

        summaries = [analyze_chunk(c) for c in chunks]
        return jsonify({"summaries": summaries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
