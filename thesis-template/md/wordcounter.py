import glob
import os


def count_words_in_file(file_path):
    """Zählt die Wörter in einer einzelnen Markdown-Datei."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        words = text.split()  # Zerlegt den Text in Wörter
        return len(words)


def count_words_in_directory(directory, exclude_files):
    """Zählt Wörter in allen Markdown-Dateien eines Verzeichnisses, außer den ausgeschlossenen."""
    if exclude_files is None:
        exclude_files = []

    total_words = 0
    file_word_counts = {}

    # Alle .md-Dateien im Verzeichnis suchen
    markdown_files = glob.glob(os.path.join(directory, "*.md"))

    for file_path in markdown_files:
        file_name = os.path.basename(file_path)

        # Datei überspringen, wenn sie in der Ausschlussliste ist
        if file_name in exclude_files:
            print(f" Datei übersprungen: {file_name}")
            continue

        word_count = count_words_in_file(file_path)
        file_word_counts[file_name] = word_count
        total_words += word_count

    return file_word_counts, total_words


# Verzeichnis mit Markdown-Dateien angeben
directory_path = "/Users/juliaseibold/Desktop/wise25/seminar_thesis/thesis-template/md"

# Dateien, die ausgeschlossen werden sollen
exclude_list = []

# Wortzählung durchführen
file_counts, total = count_words_in_directory(directory_path, exclude_list)

# Ergebnisse ausgeben
print("\n Wortanzahl pro Datei:")
for file, count in file_counts.items():
    print(f"{file}: {count} Wörter")

print(
    f"\nGesamtanzahl der Wörter in allen Markdown-Dateien (ohne ausgeschlossene): {total}"
)
