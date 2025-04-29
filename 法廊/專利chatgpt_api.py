from flask import Flask, request, render_template_string
import openai
import chardet

app = Flask(__name__)
openai.api_key = "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"

def read_file_to_text(file_storage):
    raw = file_storage.read()
    encoding = chardet.detect(raw)['encoding'] or 'utf-8'
    return raw.decode(encoding, errors='ignore')

UPLOAD_HTML = '''
<!doctype html>
<title>文義侵權判斷系統</title>
<h1>上傳請求項原始文件 與 系爭產品要件</h1>
<form action="/upload" method=post enctype=multipart/form-data>
  <p>請求項原始文件: <input type=file name=req_file></p>
  <p>系爭產品要件文件: <input type=file name=prod_file></p>
  <input type=submit value=上傳並比對>
</form>
'''

system_prompt_for_splitting = """
你是一位專利審查人員，請協助將下面的請求項技術內容拆分成一個一個要件，並以列表方式列出。
每個要件應簡明扼要，不需保留專利文件中的編號或參考符號，只需文字描述。
"""

system_prompt_for_comparison = """
你是一位專利侵權分析專家，請判定下列的「請求項要件」是否被「系爭產品要件」涵蓋。
只回傳 True 或 False，其他內容不要輸出。
- 若系爭產品要件完整包含請求項要件技術內容，則回傳 True。
- 若有缺漏或明顯不同，則回傳 False。
"""

system_prompt_for_equivalence = """
你是一位專利審查人員，根據均等侵權的三步測試（Way, Function, Result）判斷技術方案能否置換。
請依照以下標準判定每步：
- 若差異屬「結構連接方式相同，僅尺寸微調」則 Way 判 1，否則 0。
- 若功能或使用者感知方式不同則 Function 判 0。
- 若效果或使用者取得方式不同則 Result 判 0。
請以真值表列出結果：

| 比對項目 | Way | Function | Result |
|:--------|:---:|:--------:|:------:|
| 請求項 vs 系爭產品 | 1/0 | 1/0 | 1/0 |

最後補一句結論：三步均為1則均等侵權成立，否則不成立。
"""

@app.route('/')
def index():
    return UPLOAD_HTML

@app.route('/upload', methods=['POST'])
def upload_file():
    req_file = request.files['req_file']
    prod_file = request.files['prod_file']

    if req_file.filename and prod_file.filename:
        req_content = read_file_to_text(req_file)
        prod_content = read_file_to_text(prod_file)

        # 拆解請求項要件
        split_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt_for_splitting},
                {"role": "user", "content": req_content}
            ],
            temperature=0.2
        )
        req_elements = [line.strip() for line in split_response.choices[0].message.content.strip().split('\n') if line.strip()]
        prod_elements = [line.strip() for line in prod_content.strip().split('\n') if line.strip()]

        comparison_results = []
        equivalence_results = []
        total_reqs = len(req_elements)
        total_matched = 0  # 完全落入的項目數

        for req in req_elements:
            true_count = 0
            for prod in prod_elements:
                prompt = f"請求項要件：{req}\n系爭產品要件：{prod}"
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt_for_comparison},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                gpt_result = response.choices[0].message.content.strip().lower()
                if gpt_result == "true":
                    true_count += 1

            match_ratio = true_count / len(prod_elements)
            matched = true_count > 0
            if matched:
                total_matched += 1
            comparison_results.append((req, f"{match_ratio:.2%}", "True" if matched else "False"))


            # 均等侵權比對
            eq_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt_for_equivalence},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            eq_result = eq_response.choices[0].message.content.strip()
            equivalence_results.append((req, eq_result))

        match_ratio = f"{(total_matched / total_reqs) * 100:.1f}%"

        # 均等侵權個別比例統計
        individual_eq_stats = []
        for req, eq in equivalence_results:
            way = func = res = None
            for line in eq.splitlines():
                if '|' in line and '請求項' in line:
                    parts = line.split('|')
                    try:
                        way = int(parts[2].strip())
                        func = int(parts[3].strip())
                        res = int(parts[4].strip())
                        break
                    except:
                        continue
            if way is not None and func is not None and res is not None:
                total = way + func + res
                proportion = f"{total}/3"
            else:
                proportion = "無法解析"
            individual_eq_stats.append((req, proportion, eq))

        # HTML 輸出
        html_result = "<h1>【拆分後請求項要件】</h1><ul>"
        for elem in req_elements:
            html_result += f"<li>{elem}</li>"
        html_result += "</ul>"

        html_result += "<h1>【文義侵權比對結果】</h1><table border=1><tr><th>請求項要件</th><th>GPT判定</th><th>落入比例</th></tr>"
        for req, ratio, verdict in comparison_results:
            html_result += f"<tr><td>{req}</td><td>{verdict}</td><td>{ratio}</td></tr>"
        html_result += "</table>"
        html_result += f"<h2>綜合判定：{'落入文義侵權' if total_matched == total_reqs else '未全部落入文義侵權'}</h2>"

        html_result += "<h1>【均等侵權判斷結果】</h1>"
        html_result += "<h2>各要件均等侵權通過比例：</h2><table border=1><tr><th>請求項要件</th><th>通過比例 (3段)</th><th>詳細比對表</th></tr>"
        for req, proportion, eq_text in individual_eq_stats:
            bg = "#d4edda" if proportion == "3/3" else "#f8d7da" if "0/3" in proportion else "#fff3cd"
            html_result += f"<tr style='background-color: {bg};'><td>{req}</td><td>{proportion}</td><td><pre>{eq_text}</pre></td></tr>"
        html_result += "</table>"

        return html_result

    return "請上傳請求項原始文件及系爭產品要件檔案"

if __name__ == '__main__':
    app.run(debug=True)
