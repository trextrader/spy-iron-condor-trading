from docx import Document

doc = Document(r'C:\SPYOptionTrader_test\docs\trading_rules\THE 14 CANONICAL PRIMITIVES.docx')

with open(r'C:\SPYOptionTrader_test\docs\canonical_primitives_extracted.txt', 'w', encoding='utf-8') as f:
    for p in doc.paragraphs:
        if p.text.strip():
            f.write(p.text + '\n')

print("Extraction complete!")
