"""Embed the existing reviewed bibliography in the standalone LNCS source."""
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / 'paper/titantpp_pakdd_2027_draft/main.tex'
BIB = ROOT / 'reports/titantpp_manuscript_integration_20261001_v1/references.bib'


def entries(text):
    result = []
    for block in re.split(r'(?m)^@', text):
        if not block.strip():
            continue
        head = re.match(r'(\w+)\{([^,]+),', block)
        if not head:
            continue
        fields = {}
        pos = head.end()
        while pos < len(block):
            m = re.search(r'(\w+)\s*=\s*\{', block[pos:])
            if not m:
                break
            start = pos + m.end()
            depth, end = 1, start
            while depth:
                if block[end] == '{':
                    depth += 1
                elif block[end] == '}':
                    depth -= 1
                end += 1
            fields[m.group(1)] = block[start:end-1].strip('{}')
            pos = end
        result.append((head.group(2), fields))
    return result


def tex(text):
    text = text.replace('Ł', r'\L{}').replace('ł', r'\l{}')
    accents = {'\u0301': "'", '\u0300': '`', '\u0302': '^',
               '\u0308': '"', '\u0303': '~', '\u0327': 'c', '\u030c': 'v'}
    text = unicodedata.normalize('NFD', text)
    for mark, cmd in accents.items():
        text = re.sub(r'([A-Za-z])' + mark, lambda m: '\\' + cmd + '{' + m[1] + '}', text)
    for a, b in [('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')]:
        text = text.replace(a, b)
    return text


def name(author):
    if ',' in author:
        last, first = author.split(',', 1)
    else:
        parts = author.split()
        last, first = parts[-1], ' '.join(parts[:-1])
    initial = ' '.join(''.join(t[0] + '.' for t in token.split('-') if t)
                       for token in first.split())
    return tex(last.strip()) + ', ' + initial


def main():
    content = TEX.read_text()
    for before, after in [('shchur2020intensity', 'shchur2020intensityfree'),
                          ('teunter2011intermittent', 'teunter2011obsolescence'),
                          ('he2016resnet', 'he2016residual')]:
        content = content.replace(before + '}', after + '}').replace(before + ',', after + ',')
    used = set(','.join(re.findall(r'\\cite\{([^}]+)\}', content)).split(','))
    refs = entries(BIB.read_text())
    assert len(refs) == 32
    assert used == {key for key, _ in refs}, (used, {key for key, _ in refs})
    # Alphabetical reference ordering; citation keys retain the reviewed identity.
    refs.sort(key=lambda item: (unicodedata.normalize('NFKD', item[1]['author'].split(' and ')[0].split(',')[0]).casefold(), item[1]['year'], item[1]['title']))
    lines = [r'\begin{thebibliography}{32}']
    for key, f in refs:
        authors = f['author'].split(' and ')
        authors_text = ', '.join(name(a) for a in authors[:6])
        if len(authors) > 6:
            authors_text += ', et al.'
        venue = f.get('journal', f.get('booktitle', f.get('howpublished', '')))
        # Clickable titles preserve official links without long URL blocks.
        title = r'\href{' + f['url'] + '}{' + tex(f['title']) + '}' if 'url' in f else tex(f['title'])
        record = authors_text + ': ' + title + '. ' + tex(venue)
        if f.get('volume'):
            record += r' \textbf{' + f['volume'] + '}'
        if f.get('number'):
            record += '(' + f['number'] + ')'
        if f.get('pages'):
            record += ', ' + f['pages']
        record += ' (' + f['year'] + ').'
        if f.get('doi'):
            record += r' \doi{' + f['doi'] + '}'
        lines.extend([r'\bibitem{' + key + '}', record, ''])
    lines.append(r'\end{thebibliography}')
    if '% BIBLIOGRAPHY_INSERT' in content:
        content = content.replace('% BIBLIOGRAPHY_INSERT', '\n'.join(lines))
    else:
        content = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
                         lambda m: '\n'.join(lines), content, flags=re.S)
    TEX.write_text(content)
    print('Embedded all 32 reviewed references; citation keys are complete.')


if __name__ == '__main__':
    main()
