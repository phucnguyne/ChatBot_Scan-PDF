import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.gradio_ui import demo

if __name__ == "__main__":
    import gradio as gr
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Monochrome(),
    )