import os
import argparse
import os.path as path

BASE_DIR = "raw_data/trdg"

parser = argparse.ArgumentParser(description="Train a language OCR")
parser.add_argument(
    "--lang",
    help="language to train in ",
    required=True,
    choices=["eng", "yor", "igbo"],
)

args = parser.parse_args()


# with open(path.join(BASE_DIR, f"{args.lang}_target.txt"), mode="w") as f:
#     for file in os.listdir(path.join(BASE_DIR, f"{args.lang}_image")):
#         new_file = file.replace(" ", "_")
#         print(new_file)
#         os.rename(
#             path.join(BASE_DIR, f"{args.lang}_image", file),
#             path.join(BASE_DIR, f"{args.lang}_image", new_file),
#         )
#         f.write(new_file + " " + file.replace(".jpg", "").split("_")[0] + "\n")

# with open(path.join(BASE_DIR, f"{args.lang}_target.txt"), mode="r") as f:
#     text = f.read()
#     text = text.replace("\n", " ")
#     print("".join(sorted(set(text))))

# wrong_char2 = " !\"#$%&'()*+,-.0123456789:;=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz©ÀÁÅÈÉÌÍÒÓÙÚàáèéëìíòóôõöùúüćėŃńōšǸǹʻˈː̣̀́ṢṣẸẹịỌọ​‎–—‘’“”…←−▪►"
# wrong_char = ""

# with open(path.join(BASE_DIR, f"{args.lang}_target.txt"), mode="r") as f:
#     text = f.read().split("\n")
#     text_new = text.copy()
#     for ccar in wrong_char:
#         for label in text_new:
#             if ccar in set(label):
#                 try:
#                     text.remove(label)
#                     os.remove(
#                         BASE_DIR + "/" + f"{args.lang}_image" + "/" + label.split()[0]
#                     )
#                 except:
#                     pass

#                 print(
#                     os.path.exists(
#                         BASE_DIR + "/" + f"{args.lang}_image" + "/" + label.split()[0]
#                     )
#                 )
#                 print(label)
#     print(len(text_new))
#     print(len(text))

for file in os.listdir("raw_data/trdg/yor_image"):
    if "\u200b" in file:
        print(file)
        os.rename(
            os.path.join("raw_data/trdg/yor_image", file),
            os.path.join("raw_data/trdg/yor_image", file.replace("\u200b", "")),
        )
