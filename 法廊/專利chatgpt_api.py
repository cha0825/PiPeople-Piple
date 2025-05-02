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
你是一位專利審查師。目標：依「Way-Function-Result (W-F-R)」框架，抽取並歸納下列技術內容的① 技術手段 (Way)、② 直接功能 (Function)、③ 最終技術結果 (Result)。
【示例】  
- 單環溝＋單環圈環 ↔ 內外雙齒卡榫 → 不可置換 (0)
-僅作外觀形狀縮放、孔深微調、表面處理置換 → 判 1。  
◎置換判定補充規則  
A. 若兩技術藉由「不同感測／顯示裝置」讓使用者得知狀態，則 Function & Result 視為不同。  
B. Result 僅在『達成效果 + 使用者感知或控制方式』完全一致時判 1。 
在 Function 明示「感知／顯示方式」屬核心功能
若兩設計透過不同感測、顯示或指示元件讓使用者獲知同一資訊，視為功能不同。
在 Result 加入「使用者感知介面必須一致」
Result 僅在「達成效果 + 使用者取得該效果的方式」皆相同時判 1；否則判 0。
C. Way 判 1 僅限於不改變結構連接路徑、僅作形狀/尺寸微調之情況。

直接告訴我這兩項在專利比對層面上，相同專業領域者就Way | Function | Result是否具備能夠置換的可能，若有可能回覆1不可能或是有難度回0
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
