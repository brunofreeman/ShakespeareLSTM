import os
import re
import datetime

#Data Settings
USING_WORDS = False
MIN_UNIT_COUNT = 1
ANALYSIS_TYPE = "word" if USING_WORDS else "character"

#File Settings
ROOT = "."
DATA_DIR = os.path.join(ROOT, "shakespeare_data")
PRINT_TO_FILE = True
TIMESTAMP_FILE = False

def get_time_for_file():
    return datetime.datetime.now().strftime("_%m.%d.%y-%H.%M.%S")

OUTPUT_FILE = ANALYSIS_TYPE + "_data_analysis"
if TIMESTAMP_FILE:
    OUTPUT_FILE += get_time_for_file()
OUTPUT_FILE += ".txt"
OUTPUT_FILE = os.path.join(ROOT, "data_analysis", OUTPUT_FILE)

file_count = 0
text = ""
for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        file_count += 1
        text += open(os.path.join(DATA_DIR, file)).read()

if USING_WORDS:
    text = text.lower()

regex = r"(?:[A-Za-z']*(?:(?<!-)-(?!-))*[A-Za-z']+)+" + r"|\.|\?|!|,|;|:|-|\(|\)|\[|\]|\{|\}|\'|\"|\|\/|<|>| |\t|\n" if USING_WORDS else "."
units = re.findall(regex, text)
unit_counts = dict()

for unit in units: #create a dict mapping unit to count
    unit_counts[unit] = unit_counts.get(unit, 0) + 1

unit_counts = sorted(list(unit_counts.items()), key=lambda i: (-i[1], i[0])) #convert dict to list of tuples sort by count then unit

total_units = 0
total_top_units = 0
num_top_units = 0
for i in range(0, len(unit_counts)):
    total_units += unit_counts[i][1]
    if unit_counts[i][1] >= MIN_UNIT_COUNT:
        num_top_units += 1
        total_top_units += unit_counts[i][1]

pre_unk_len = len(unit_counts)

less_than_min = 0
for i in range(len(unit_counts) - 1, -1, -1):
    if unit_counts[i][1] < MIN_UNIT_COUNT:
        less_than_min += unit_counts[i][1]
        del unit_counts[i]

unit_counts.append(("<UNK>", less_than_min))

num_top_units_with_unk = num_top_units
if less_than_min >= MIN_UNIT_COUNT:
    num_top_units_with_unk += 1

unique_percent = num_top_units / pre_unk_len * 100
total_percent = total_top_units / total_units * 100

output = "%d files analyzed\n\n" % file_count
output += ("%d unique " + ANALYSIS_TYPE + "s\n%d total " + ANALYSIS_TYPE + "s\n\n") % (pre_unk_len, total_units)
output += ("Showing " + ANALYSIS_TYPE + "s with count >= %d (top %d)\n") % (MIN_UNIT_COUNT, num_top_units)
output += "%.1f%% of unique, %.1f%% of total\n\n" % (unique_percent, total_percent)
if num_top_units_with_unk > num_top_units:
    output += "Sum of counts of non-top " + ANALYSIS_TYPE + "s included under <UNK>\n"
    output += "<UNK> not included in stats, but is ranked\n\n"
output += "%6s%16s%10s\n" % ("Rank:", "Word:", "Count:")
output += "--------------------------------"

unit_counts.sort(key=lambda i: (-i[1], i[0])) #resort for <UNK>

for i in range(0, num_top_units_with_unk):
    w = unit_counts[i][0]
    if w == "\n":
        w = "<NLN>"
    elif w == "\t":
        w = "<TAB>"
    elif w == " ":
        w = "<SPC>"
    output += "\n%5d)%16s%10d" % (i + 1, w, unit_counts[i][1])

if PRINT_TO_FILE:
    with open(OUTPUT_FILE, "w") as output_file:
        output_file.write(output)
else:
    print(output)