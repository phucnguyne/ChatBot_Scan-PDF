import re

def extract_directives(message: str):
    """
    Tìm source:filename.pdf và page:3 trong câu hỏi.
    Trả về (cleaned_query, filter_dict).
    """
    filt = {}
    q    = message

    m = re.search(r'\bsource:(\S+)', q, re.IGNORECASE)
    if m:
        filt["source"] = m.group(1)
        q = q[:m.start()] + q[m.end():]

    m = re.search(r'\bpage:(\d+)', q, re.IGNORECASE)
    if m:
        filt["page"] = int(m.group(1))
        q = q[:m.start()] + q[m.end():]

    return q.strip(), filt