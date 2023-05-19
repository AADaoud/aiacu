import os
import re

def extract_key_topics(file_path):
    with open(file_path, 'r', encoding="utf-8", errors='ignore') as file:
        content = file.read()

    # Extract content under "# Key Topics" heading
    pattern = r'# Key Topics\s*([\s\S]*?)(?=#|$)'
    match = re.search(pattern, content)
    if match:
        key_topics = match.group(1)
        return key_topics.strip()
    else:
        return ''

def extract_words(directory):
    word_set = set()
    for file_name in os.listdir(directory):
        if file_name.endswith('.md'):
            file_path = os.path.join(directory, file_name)
            key_topics = extract_key_topics(file_path)
            words = re.findall(r'\w+', key_topics)
            word_set.update(words)

    return word_set

# Provide the directory containing the .md files
directory_path = "C:\\Users\\Kefah\\OneDrive\\Desktop\\md files2"
word_set = extract_words(directory_path)

# Write the word set to a set.txt file with 'utf-8' encoding
with open('set.txt', 'w', encoding='utf-8') as file:
    for word in word_set:
        file.write(word + '\n')

print('Word set extracted and saved to set.txt.')
