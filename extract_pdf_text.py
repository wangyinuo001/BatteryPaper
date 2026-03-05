from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("Smartphone_Battery_Continuous_Time_Model_MCM2026.pdf")
reader = PdfReader(str(pdf_path))
text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
out_path = pdf_path.with_suffix(".txt")
out_path.write_text(text, encoding="utf-8")
print(f"pages={len(reader.pages)}")
print(f"chars={len(text)}")
print(f"out={out_path}")
print(text[:3000])
