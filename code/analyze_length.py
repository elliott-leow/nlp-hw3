#!/usr/bin/env python3
"""
Analyze classification performance vs file length
"""
import re
from collections import defaultdict

#read classification results (skip info lines)
results = []
with open('classification_results.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('gen.model') or line.startswith('spam.model'):
            results.append(line)

#parse each result
data = []
for line in results:
    parts = line.split('\t')
    predicted = parts[0]  #gen.model or spam.model
    filepath = parts[1]   #../data/gen_spam/dev/gen/gen.101.156.txt
    
    #extract actual category and length from filename
    filename = filepath.split('/')[-1]  #gen.101.156.txt
    if filename.startswith('gen.'):
        actual = 'gen.model'
        #extract length: gen.101.156.txt -> 101
        length = int(filename.split('.')[1])
    elif filename.startswith('spam.'):
        actual = 'spam.model'  
        #extract length: spam.104.052.txt -> 104
        length = int(filename.split('.')[1])
    
    correct = (predicted == actual)
    data.append((length, correct, actual, predicted))

#group by length ranges
length_ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, float('inf'))]
range_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

for length, correct, actual, predicted in data:
    for min_len, max_len in length_ranges:
        if min_len <= length < max_len:
            range_key = f"{min_len}-{max_len if max_len != float('inf') else '∞'}"
            range_stats[range_key]['total'] += 1
            if correct:
                range_stats[range_key]['correct'] += 1
            break

#print results
print("Performance by File Length:")
print("Length Range\tTotal\tCorrect\tAccuracy")
for min_len, max_len in length_ranges:
    range_key = f"{min_len}-{max_len if max_len != float('inf') else '∞'}"
    stats = range_stats[range_key]
    if stats['total'] > 0:
        accuracy = stats['correct'] / stats['total'] * 100
        print(f"{range_key}\t{stats['total']}\t{stats['correct']}\t{accuracy:.1f}%")

#also show individual file lengths with mistakes
print("\nFiles with incorrect classifications:")
print("Length\tActual\tPredicted\tFilename")
mistakes = [(length, actual, predicted, filepath.split('/')[-1]) for length, correct, actual, predicted, filepath in [(length, correct, actual, predicted, line.split('\t')[1]) for length, correct, actual, predicted in data for line in results if (line.split('\t')[0] != actual)] if not correct]

#fix the mistake tracking
mistakes = []
for line in results:
    parts = line.split('\t')
    predicted = parts[0]
    filepath = parts[1]
    filename = filepath.split('/')[-1]
    
    if filename.startswith('gen.'):
        actual = 'gen.model'
        length = int(filename.split('.')[1])
    elif filename.startswith('spam.'):
        actual = 'spam.model'
        length = int(filename.split('.')[1])
    
    if predicted != actual:
        mistakes.append((length, actual, predicted, filename))

#sort mistakes by length
mistakes.sort()
for length, actual, predicted, filename in mistakes[:20]:  #show first 20
    print(f"{length}\t{actual}\t{predicted}\t{filename}")

if len(mistakes) > 20:
    print(f"... and {len(mistakes) - 20} more mistakes")
