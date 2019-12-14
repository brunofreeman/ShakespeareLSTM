import re
import operator

text = open("./shakespeare_data/Othello.txt").read().lower()
regex = '(?:[A-Za-z\']*(?:(?<!-)-(?!-))*[A-Za-z\']+)+|\.|\?|!|,|;|:|-|–|—|―|\(|\)|\[|\]|\{|\}|\'|\"|\\|\/|<|>| |\t|\n'
words = re.findall(regex, text)
word_counts = dict()

for word in words:
    word_counts[word] = word_counts.get(word, 0) + 1 #test if zero-indexing

word_counts = sorted(list(word_counts.items()), key = operator.itemgetter(1))

num_top_words = 500
print("%5s%20s%8s" % ("Rank:", "Word:", "Count:"))
for i in range(len(word_counts) - 1, len(word_counts) - 1 - num_top_words, -1):
    print("%5d)%20s%8d" % (len(word_counts) - i, word_counts[i][0], word_counts[i][1]))

#print(word_counts, "\n")
#print(len(word_counts))