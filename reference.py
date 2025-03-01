import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def create_download_link(text, filename="document"):
    """Generate a downloadable PDF link from text content using ReportLab with Unicode support."""
    # Register a Unicode-compatible font
    pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
    
    # Create PDF
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont('DejaVuSans', 12)
    
    # Simple text wrapping
    width, height = letter
    y = height - 50  # Start near the top
    margin = 50
    line_height = 14
    
    # Split text into lines
    words = text.split()
    line = ""
    
    for word in words:
        if p.stringWidth(line + " " + word, 'DejaVuSans', 12) < width - 2*margin:
            line += " " + word if line else word
        else:
            p.drawString(margin, y, line)
            y -= line_height
            line = word
            
            # Check if we need a new page
            if y < margin:
                p.showPage()
                p.setFont('DejaVuSans', 12)
                y = height - 50
    
    # Draw the last line
    if line:
        p.drawString(margin, y, line)
    
    p.save()
    
    # Create base64 string and HTML link
    buffer.seek(0)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf" target="_blank">📄 Download PDF</a>'
    return href