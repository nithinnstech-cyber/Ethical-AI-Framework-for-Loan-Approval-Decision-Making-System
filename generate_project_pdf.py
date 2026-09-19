from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, SimpleDocTemplate

md_path = r"C:\Users\NITHIN N S\Desktop\Ethical AI Framework for Loan approval Decision Making System\project_overview.md"
out_path = r"C:\Users\NITHIN N S\Desktop\Ethical AI Framework for Loan approval Decision Making System\project_overview.pdf"

with open(md_path, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

styles = getSampleStyleSheet()
story = []

style_h1 = ParagraphStyle('h1', parent=styles['Heading1'], fontSize=18, leading=22, spaceAfter=12, textColor=colors.HexColor('#1f2937'))
style_h2 = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=14, leading=18, spaceAfter=8, textColor=colors.HexColor('#111827'))
style_body = ParagraphStyle('body', parent=styles['BodyText'], fontSize=10.5, leading=14, spaceAfter=6)
style_bullet = ParagraphStyle('bullet', parent=styles['BodyText'], fontSize=10.5, leading=14, leftIndent=18, bulletIndent=10, spaceAfter=4)

for line in lines:
    if not line.strip():
        story.append(Spacer(1, 6))
        continue
    if line.startswith('# '):
        story.append(Paragraph(line[2:], style_h1))
    elif line.startswith('## '):
        story.append(Paragraph(line[3:], style_h2))
    elif line.startswith('### '):
        story.append(Paragraph(line[4:], style_h2))
    elif line.startswith('- '):
        story.append(Paragraph(line[2:], style_bullet))
    else:
        story.append(Paragraph(line, style_body))

pdf = SimpleDocTemplate(out_path, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=40, bottomMargin=40)
pdf.build(story)
print(f"Created PDF: {out_path}")
