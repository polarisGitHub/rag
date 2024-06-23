import re
import fitz
import json
import codecs
from langchain_text_splitters import RecursiveCharacterTextSplitter

# extract_pdf
start_page = 42
end_page = 1235
pdf_path = "data/美国儿科学会育儿百科(第7版).pdf"
parse_path = "data/extract.json"

# 正常文本块
normal_title_metas = [
    {"level": 1, "size": 30, "font": "SimHei"},
    {"level": 1, "size": 30, "font": "SimSun"},
    {"level": 2, "size": 27.75, "font": "SimSun"},
    {"level": 3, "size": 21, "font": "SimSun"},
]
normal_text_metas = [
    {"font": "SimSun", "size": 15, "left_padding": 72},
]

# 特殊文本块
special_title_metas = [
    {"level": 4, "size": 17.25, "font": "SimSun"},
]
special_text_metas = [
    {"font": "KaiTi", "size": 15, "left_padding": 85.5},
]
special_feature = special_title_metas + special_text_metas


# 解析pdf span
def parse(spans):
    span_meta = []
    for span in spans:
        span_meta.append(
            {
                "left_padding": span["bbox"][0],
                "size": span["size"],
                "font": span["font"],
                "text": span["text"],
            }
        )
    text = ""
    # 文本处理
    for span in span_meta:
        text += span["text"]

    size, font, left_padding = span_meta[0]["size"], span_meta[0]["font"], span_meta[0]["left_padding"]
    special, is_title, titel_level, is_newline = False, False, -1, False

    # 判断是否是特殊块
    for feature in special_feature:
        if size == feature["size"] and font == feature["font"]:
            special = True

    for title_info in special_title_metas if special else normal_title_metas:
        if size == title_info["size"] and font == title_info["font"]:
            is_title, titel_level = True, title_info["level"]

    for text_info in special_text_metas if special else normal_text_metas:
        if size == text_info["size"] and font == text_info["font"] and left_padding > text_info["left_padding"]:
            is_newline = True
    return text, is_title, titel_level, is_newline, special


doc = fitz.open(pdf_path)


def set_title(title: dict, level: int, content: str) -> dict:
    new_title = {}
    if level == 1:
        new_title["1"] = content
        new_title["2"] = ""
        new_title["3"] = ""
    elif level == 2:
        new_title["1"] = title["1"]
        new_title["2"] = content
        new_title["3"] = ""
    elif level == 3:
        new_title["1"] = title["1"]
        new_title["2"] = title["2"]
        new_title["3"] = content
    return new_title


title = {
    "1": "",
    "2": "",
    "3": "",
}
previous_special, special_title = False, ""
data, span_text, content = [], "", ""
for page in doc:
    page_num = page.number
    if page_num < start_page or page_num > end_page:
        continue

    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            span_text, is_title, title_level, is_newline, special = parse(line["spans"])
            if is_title:
                # 如果是title，记录之前的content和title，并复位content
                data.append({"content": content, "title": title, "special": ""})
                content = ""
                # 如果是特殊文本title，记录下来，但不设置title
                if special:
                    special_title = span_text
                else:
                    # 非特殊文本title，设置title，并复位special_title
                    title = set_title(title, title_level, span_text)
                    special_title = ""
            else:
                # 如果之前是特殊文本块，现在不是，也做一次切割
                if is_newline and len(content) != 0:
                    content += "\n"
                if previous_special and not special:
                    data.append({"content": content, "title": title, "special": special_title})
                    content, special_title = "", ""
                content += span_text
            previous_special = special
data.append({"content": content, "title": title, "special": special_title})

json_str = json.dumps(data, ensure_ascii=False)
with codecs.open(parse_path, mode="w", encoding="utf-8") as f:
    f.write(json_str)

# split
json_file = "data/extract.json"
split_contents_path = "data/splited.json"
json_content = "[]"
with codecs.open(json_file, mode="r", encoding="utf-8") as reader:
    json_content = reader.read()
json_array = json.loads(json_content)

# 文本替换
replace_meta = {
    "　": " ",
    "■ ": "",
    "\n。": "",
    "^。": "",
    "\n{2,}": "\n",
}


def process_block_text(content: str) -> str:
    content = content.strip()
    for key, value in replace_meta.items():
        content = re.sub(key, value, content)
    return content


splited_list, text_splitter = [], RecursiveCharacterTextSplitter(
    chunk_size=128,
    chunk_overlap=32,
    separators=[
        "。",
        " ",
    ],
    length_function=len,
    is_separator_regex=False,
)
for json_object in json_array:
    content, title, special = json_object["content"], json_object["title"], json_object["special"]
    if len(content.strip()) == 0:
        continue
    texts = text_splitter.create_documents([content])
    for text in texts:
        splited_list.append(
            {
                "content": process_block_text(text.page_content),
                "title": [
                    process_block_text(json_object["title"]["1"]),
                    process_block_text(json_object["title"]["2"]),
                    process_block_text(json_object["title"]["3"]),
                ],
                "special_title": json_object["special"],
            }
        )

with codecs.open(split_contents_path, mode="w", encoding="utf-8") as f:
    f.write(json.dumps(splited_list, ensure_ascii=False))
