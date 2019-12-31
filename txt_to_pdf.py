import sys
import os
from fpdf import FPDF

ROOT = "."
OUTPUT_DIR_1 = "output"
OUTPUT_DIR_2 = "pdf_output"

def txt_to_pdf(file_path):
    try:
        text = open(file_path).read().replace("\t", "\t\t").split("\n") #a single tab doesn't work well with this font
        file_name = file_path.split(os.path.sep)[-1][:-4] #relies on current file naming convention
    except:
        print("Invalid file path provided to txt_to_pdf(). Exiting...")
        sys.exit()

    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_font("courier", "B", 13.0)
    for line in text:
        pdf.cell(ln=1, h=5.0, align="L", w=0, txt=line, border=0)
    pdf.output(os.path.join(".", os.path.join(ROOT, OUTPUT_DIR_1, OUTPUT_DIR_2, file_name + ".pdf")), "F")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Provide path to text file relative to current path as program argument. Exiting...")
        sys.exit()

    file_to_convert = sys.argv[1]

    txt_to_pdf(os.path.join(ROOT, file_to_convert))