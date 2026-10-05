PROMPT_TEMPLATE = """<|im_start|>system
Answer only using the `context` provided below.
If the context does not contain the answer, say: "The document does not contain this information."
Do not guess. When possible, cite sources using [Source: <filename> - page <page>].
Answer in the same language as the user's question.

{context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant"""