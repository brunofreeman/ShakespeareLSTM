import os
import re

#Data Settings
MIN_WORD_COUNT = 50

#File Settings
ROOT = "."
DATA_DIR = os.path.join(ROOT, "shakespeare_data")
OUTPUT_FILE = os.path.join(ROOT, "data_analysis.txt")
PRINT_TO_FILE = True

file_count = 0
text = ""
for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        file_count += 1
        text += open(os.path.join(DATA_DIR, file)).read().lower()

word_regex = "(?:[A-Za-z\']*(?:(?<!-)-(?!-))*[A-Za-z\']+)+"
punct_regex = r"|\.|\?|!|,|;|:|-|\(|\)|\[|\]|\{|\}|\'|\"|\|\/|<|>| |\t|\n"
regex = word_regex + punct_regex
words = re.findall(regex, text)
word_counts = dict()

for word in words: #create a dict mapping word to count
    word_counts[word] = word_counts.get(word, 0) + 1

word_counts = sorted(list(word_counts.items()), key=lambda i: (-i[1], i[0])) #convert dict to list of tuples sort by count then word

total_words = 0
total_top_words = 0
num_top_words = 0
for i in range(0, len(word_counts)):
    total_words += word_counts[i][1]
    if word_counts[i][1] >= MIN_WORD_COUNT:
        num_top_words += 1
        total_top_words += word_counts[i][1]

pre_unk_len = len(word_counts)

less_than_min = 0
for i in range(len(word_counts) - 1, -1, -1):
    if word_counts[i][1] < MIN_WORD_COUNT:
        less_than_min += word_counts[i][1]
        del word_counts[i]

word_counts.append(("<UNK>", less_than_min))

num_top_words_with_unk = num_top_words
if less_than_min >= MIN_WORD_COUNT:
    num_top_words_with_unk += 1

unique_percent = num_top_words / pre_unk_len * 100
total_percent = total_top_words / total_words * 100

output = "%d files analyzed\n\n" % file_count
output += "%d unique words\n%d total words\n\n" % (pre_unk_len, total_words)
output += "Showing words with count >= %d (top %d)\n" % (MIN_WORD_COUNT, num_top_words)
output += "%.1f%% of unique, %.1f%% of total\n\n" % (unique_percent, total_percent)
if num_top_words_with_unk > num_top_words:
    output += "Sum of counts of non-top words included under <UNK>\n"
    output += "<UNK> not included in stats, but is ranked\n\n"
output += "%6s%16s%10s\n" % ("Rank:", "Word:", "Count:")
output += "--------------------------------"

word_counts.sort(key=lambda i: (-i[1], i[0])) #resort for <UNK>

for i in range(0, num_top_words_with_unk):
    w = word_counts[i][0]
    if w == "\n":
        w = "<NLN>"
    elif w == "\t":
        w = "<TAB>"
    elif w == " ":
        w = "<SPC>"
    output += "\n%5d)%16s%10d" % (i + 1, w, word_counts[i][1])

if PRINT_TO_FILE:
    with open(OUTPUT_FILE, "w") as output_file:
        output_file.write(output)
else:
    print(output)