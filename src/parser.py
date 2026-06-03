import fitz
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import os
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class DocumentParser:
    def __init__(self, poppler_path: str = None):
        self.poppler_path = poppler_path

    def is_page_scanned(self, page) -> bool:
        return len(page.get_text().strip()) < 50

    def detect_section_header(self, text: str):
        patterns = [
            r"^(Chapter\s+\d+[:\.]?.*)$",
            r"^(Section\s+\d+[\.\d]*[:\.]?.*)$",
            r"^(Rule\s+\d+[\.\d]*[:\.]?.*)$",
            r"^(Article\s+\d+[\.\d]*[:\.]?.*)$",
            r"^(\d+[\.\d]*\s+[A-Z][^\n]{5,50})$",
        ]
        for line in text.split("\n"):
            line = line.strip()
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    return line
        return None

    def clean_text(self, text: str) -> str:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r"^\d+$", line):
                continue
            if len(line) < 4:
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    def table_to_text(self, table: list, title: str = "") -> str:
        if not table or len(table) < 2:
            return ""
        headers = [str(h).strip() if h else "" for h in table[0]]
        rows = table[1:]
        sentences = []
        if title:
            sentences.append(f"Table: {title}.")
        for row in rows:
            if not any(row):
                continue
            parts = []
            for header, cell in zip(headers, row):
                cell_val = str(cell).strip() if cell else ""
                if cell_val and header:
                    parts.append(f"{header}: {cell_val}")
            if parts:
                sentences.append(". ".join(parts) + ".")
        return " ".join(sentences)

    def extract_lists(self, text: str) -> list:
        lines = text.split("\n")
        list_items = []
        for line in lines:
            line = line.strip()
            if re.match(r"^[\•\-\*\–]\s+.+", line):
                list_items.append(line.lstrip("•-*– ").strip())
            elif re.match(r"^\d+[\.\)]\s+.+", line):
                list_items.append(re.sub(r"^\d+[\.\)]\s+", "", line).strip())
        return list_items

    def ocr_page(self, pdf_path: str, page_num: int) -> str:
        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1,
            poppler_path=self.poppler_path,
            dpi=300
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0], lang="hin+eng").strip()

    def parse(self, pdf_path: str) -> list:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_name = os.path.basename(pdf_path)
        fitz_doc = fitz.open(pdf_path)
        total_pages = len(fitz_doc)
        all_chunks = []
        current_section = "Introduction"

        print(f"\nParsing: {doc_name} ({total_pages} pages)")

        for page_num in range(total_pages):
            fitz_page = fitz_doc[page_num]

            if self.is_page_scanned(fitz_page):
                print(f"  Page {page_num+1}: scanned → OCR")
                ocr_text = self.ocr_page(pdf_path, page_num)
                if ocr_text:
                    all_chunks.append({
                        "text":     ocr_text,
                        "type":     "ocr_text",
                        "source":   doc_name,
                        "page":     page_num + 1,
                        "section":  current_section,
                        "chunk_id": f"{doc_name}_p{page_num+1}_ocr"
                    })
                continue

            print(f"  Page {page_num+1}: digital")
            raw_text = self.clean_text(fitz_page.get_text())

            header = self.detect_section_header(raw_text)
            if header:
                current_section = header
                all_chunks.append({
                    "text":     header,
                    "type":     "section_header",
                    "source":   doc_name,
                    "page":     page_num + 1,
                    "section":  current_section,
                    "chunk_id": f"{doc_name}_p{page_num+1}_header"
                })

            list_items = self.extract_lists(raw_text)
            if list_items:
                all_chunks.append({
                    "text":     "Items: " + ". ".join(list_items) + ".",
                    "type":     "list",
                    "source":   doc_name,
                    "page":     page_num + 1,
                    "section":  current_section,
                    "chunk_id": f"{doc_name}_p{page_num+1}_list"
                })

            with pdfplumber.open(pdf_path) as plumber_doc:
                plumber_page = plumber_doc.pages[page_num]
                tables = plumber_page.extract_tables()
                for t_idx, table in enumerate(tables):
                    table_text = self.table_to_text(
                        table,
                        title=f"Table {t_idx+1} from {doc_name} page {page_num+1}"
                    )
                    if table_text:
                        all_chunks.append({
                            "text":     table_text,
                            "type":     "table",
                            "source":   doc_name,
                            "page":     page_num + 1,
                            "section":  current_section,
                            "chunk_id": f"{doc_name}_p{page_num+1}_t{t_idx}"
                        })

            if raw_text:
                all_chunks.append({
                    "text":     raw_text,
                    "type":     "paragraph",
                    "source":   doc_name,
                    "page":     page_num + 1,
                    "section":  current_section,
                    "chunk_id": f"{doc_name}_p{page_num+1}_text"
                })

        fitz_doc.close()
        print(f"  Done — {len(all_chunks)} chunks extracted\n")
        return all_chunks
