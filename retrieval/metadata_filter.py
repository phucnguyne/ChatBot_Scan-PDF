import re

def extract_directives(message: str):
    """
    Find source:filename.pdf and page:3 directives in the question.
    Return (cleaned_query, filter_dict).
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