import os
import re
import operator

PRINT_TO_FILE = True
MIN_COUNT = 50

file_count = 0
text = ""
for file in os.listdir("./shakespeare_data"):
    if file.endswith(".txt"):
        file_count += 1
        text += open(os.path.join("./shakespeare_data", file)).read().lower()

word_regex = "(?:[A-Za-z\']*(?:(?<!-)-(?!-))*[A-Za-z\']+)+"
punct_regex = "|\.|\?|!|,|;|:|-|\(|\)|\[|\]|\{|\}|\'|\"|\\|\/|<|>| |\t|\n"
regex = word_regex + punct_regex
words = re.findall(regex, text)
word_counts = dict()

for word in words: #create a dict mapping word to count
    word_counts[word] = word_counts.get(word, 0) + 1

word_counts = sorted(list(word_counts.items()), key = operator.itemgetter(1)) #convert dict to list of tuples sort by count then word

total_words = 0
total_top_words = 0
num_top_words = 0
for i in range(len(word_counts) - 1, -1, -1):
    total_words += word_counts[i][1]
    if word_counts[i][1] >= MIN_COUNT:
        num_top_words += 1
        total_top_words += word_counts[i][1]

unique_percent = float(num_top_words) / len(word_counts) * 100
total_percent = float(total_top_words) / total_words * 100

output = "%d files analyzed\n\n" % file_count
output += "%d unique words\n%d total words\n\n" % (len(word_counts), total_words)
output += "Showing words with count >= %d (top %d)\n" % (MIN_COUNT, num_top_words)
output += "%.1f%% of unique, %.1f%% of total\n\n" % (unique_percent, total_percent)
output += "%6s%16s%10s\n" % ("Rank:", "Word:", "Count:")
output += "--------------------------------"

for i in range(len(word_counts) - 1, len(word_counts) - 1 - num_top_words, -1):
    w = word_counts[i][0];
    if w == "\n":
        w = "<NEWLINE>"
    elif w == "\t":
        w = "<TAB>"
    elif w == " ":
        w = "<SPACE>"
    output += "\n%5d)%16s%10d" % (len(word_counts) - i, w, word_counts[i][1])

if PRINT_TO_FILE:
    with open("./data_analysis.txt", "w") as output_file:
        output_file.write(output)
else:
    print(output)