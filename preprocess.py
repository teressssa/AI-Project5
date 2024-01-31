import csv
import json


# Define a function to convert tags to integers
def tag_to_int(tag):
    if tag == "negative":
        return 0
    elif tag == "neutral":
        return 1
    elif tag == "positive":
        return 2


# data = {"test": [], "train": [], "val": []}
train = []
val = []
test = []
cur = 0
with open("train.txt", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    for row in reader:
        with open(f"data/{row[0]}.txt", "r", encoding="unicode_escape") as txt_file:
            reader_txt = csv.reader(txt_file)
            txt = next(reader_txt)[0]
            if cur < 400:
                val.append(
                    {
                        "id": row[0],
                        "text": txt,
                        "label": tag_to_int(row[1]),
                        "path": f"data/{row[0]}.jpg",
                    }
                )
            else:
                train.append(
                    {
                        "id": row[0],
                        "text": txt,
                        "label": tag_to_int(row[1]),
                        "path": f"data/{row[0]}.jpg",
                    }
                )
            cur += 1

with open("test_without_label.txt", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    for row in reader:
        with open(f"data/{row[0]}.txt", "r", encoding="unicode_escape") as txt_file:
            reader_txt = csv.reader(txt_file)
            txt = next(reader_txt)[0]
            test.append(
                {
                    "id": row[0],
                    "text": txt,
                    "label": tag_to_int(row[1]),
                    "path": f"data/{row[0]}.jpg",
                }
            )


with open("train.json", "w") as f:
    json.dump(train, f, indent=4)
with open("val.json", "w") as f:
    json.dump(val, f, indent=4)
with open("test.json", "w") as f:
    json.dump(test, f, indent=4)
