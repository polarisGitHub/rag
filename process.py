import os
import json
import codecs
from common_utils import sha256


def process_json(file_name: str, file_prefix: str, processed_file_prefix: str):
    temp_file_prefix = "data/temp/"
    with codecs.open(file_prefix + file_name, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    name, category, isbn, contents = data["name"], data["category"], data["isbn"], data["contents"]

    length = len(contents)
    temp_list = []
    for i in range(0, length):
        text, paragraph = contents[i]["text"], contents[i]["paragraph"]
        row_info = f"{isbn}:{i}:{sha256(text)}"
        temp_list.append(
            {
                "row_id": sha256(row_info),
                "text": text,
                "paragraph": paragraph,
            }
        )

    temp_list[0]["previous_id"] = ""
    temp_list[0]["next_id"] = temp_list[1]["row_id"]
    temp_list[length - 1]["previous_id"] = temp_list[length - 2]["row_id"]
    temp_list[length - 1]["next_id"] = ""
    for i in range(1, length - 1):
        temp_list[i]["previous_id"] = temp_list[i - 1]["row_id"]
        temp_list[i]["next_id"] = temp_list[i + 1]["row_id"]

    with codecs.open(temp_file_prefix + file_name, mode="w", encoding="utf-8") as f:
        json.dump(temp_list, f, ensure_ascii=False)

    fragment_list = []
    sentence_list = []
    for item in temp_list:
        text = item["text"]
        fragment_list.append(
            {
                ##
                "isbn": isbn,
                "name": name,
                "category": category,
                "paragraph": item["paragraph"],
                ##
                "fragment": text,
                "fragment_id": item["row_id"],
                "fragment_next_id": item["next_id"],
                "fragment_previous_id": item["previous_id"],
            }
        )
        sentences = text.split("。")
        for sentence in sentences:
            if not sentence:
                continue
            sentence_info = f'{item["row_id"]}:{sha256(sentence)}'
            sentence_list.append(
                {
                    ##
                    "isbn": isbn,
                    "name": name,
                    "category": category,
                    "paragraph": item["paragraph"],
                    ##
                    "fragment_id": item["row_id"],
                    ##
                    "sentence_id": sha256(sentence_info),
                    "sentence": sentence,
                }
            )

    length = len(sentence_list)
    sentence_list[0]["sentence_previous_id"] = ""
    sentence_list[0]["sentence_next_id"] = sentence_list[1]["sentence_id"]
    sentence_list[length - 1]["sentence_previous_id"] = sentence_list[length - 2]["sentence_id"]
    sentence_list[length - 1]["sentence_next_id"] = ""
    for i in range(1, length - 1):
        sentence_list[i]["sentence_previous_id"] = sentence_list[i - 1]["sentence_id"]
        sentence_list[i]["sentence_next_id"] = sentence_list[i + 1]["sentence_id"]

    processed_list = []
    for item in fragment_list:
        processed_list.append(
            {
                ##
                "isbn": isbn,
                "name": name,
                "category": category,
                "paragraph": item["paragraph"],
                ##
                "content_type": "fragment",
                ##
                "id": item["fragment_id"],
                "previous_id": item["fragment_previous_id"],
                "next_id": item["fragment_next_id"],
                "parent_id": "",
                ##
                "text": item["fragment"],
            }
        )

    for item in sentence_list:
        processed_list.append(
            {
                ##
                "isbn": isbn,
                "name": name,
                "category": category,
                "paragraph": item["paragraph"],
                ##
                "content_type": "sentence",
                ##
                "id": item["sentence_id"],
                "previous_id": item["sentence_previous_id"],
                "next_id": item["sentence_next_id"],
                "parent_id": item["fragment_id"],
                "text": item["sentence"],
            }
        )

    with codecs.open(processed_file_prefix + file_name, mode="w", encoding="utf-8") as f:
        json.dump(processed_list, f, ensure_ascii=False)


if __name__ == "__main__":
    file_prefix = "data/"
    for f in os.listdir(file_prefix):
        if f.endswith(".json"):
            process_json(f, file_prefix=file_prefix, processed_file_prefix="data/processed/")
