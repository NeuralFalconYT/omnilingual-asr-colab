import gradio as gr
import torch
import os
import warnings
import sys
import os
fix_import=f"{os.getcwd()}/server"
sys.path.append(fix_import)
from inference.audio_chunker import AudioChunker
from inference.audio_sentence_alignment import AudioAlignment
from inference.mms_model_pipeline import MMSModel
from media_transcription_processor import MediaTranscriptionProcessor
from subtitle import make_subtitle
from lang_dict import lang_code  # ✅ your language dictionary

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# ---- Setup Model Globals ----
_model_loaded = False
_model_loading = False

# ---- Initialize model ----
def load_model(model_name="omniASR_LLM_1B"):
    """Load MMS model on startup - only once."""
    global _model_loaded, _model_loading
    if _model_loaded or _model_loading:
        return

    _model_loading = True
    print(f"🔄 Loading {model_name} model...")

    AudioChunker()
    AudioAlignment()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MMSModel(model_card=model_name, device=device)

    _model_loaded = True
    _model_loading = False
    print("✅ Model loaded successfully.")


# ---- Transcription function ----
def media_transcription(file_path, lang_code="eng_Latn"):
    """Perform transcription + subtitle generation."""
    with open(file_path, "rb") as f:
        media_bytes = f.read()

    processor = MediaTranscriptionProcessor(
        media_bytes=media_bytes,
        filename=file_path,
        language_with_script=lang_code
    )

    processor.convert_media()
    processor.transcribe_full_pipeline()
    results = processor.get_results()

    transcription = results['transcription']
    word_level_timestamps = [
        {"word": s['text'], "start": s['start'], "end": s['end']}
        for s in results.get('aligned_segments', [])
    ]

    sentence_srt, word_level_srt, shorts_srt = make_subtitle(word_level_timestamps, file_path)
    return transcription, sentence_srt, word_level_srt, shorts_srt


# ---- Gradio Interface ----
def transcribe_interface(audio, selected_lang):
    """Main Gradio wrapper."""
    if audio is None:
        return "Please upload or record audio.", None, None, None

    # Save uploaded/recorded audio
    file_path = audio
    find_lang_code = lang_code[selected_lang]

    print(f"🎙 Transcribing {file_path} in {selected_lang} ({find_lang_code})...")

    try:
        transcription, sentence_srt, word_level_srt, shorts_srt = media_transcription(file_path, find_lang_code)
        return transcription, sentence_srt, word_level_srt, shorts_srt
    except Exception as e:
        return f"❌ Error: {e}", None, None, None



def ui():
    lang_list = list(lang_code.keys())
    custom_css = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""
    with gr.Blocks(theme=gr.themes.Soft(),css=custom_css) as demo:
        gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 10px;">omniasr-transcriptions</h1>
            <a href="https://github.com/NeuralFalconYT/omnilingual-asr-colab" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #4285F4; color: white; border-radius: 6px; text-decoration: none; font-size: 1em;">😇 Run on Google Colab</a>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=[ "microphone","upload"], type="filepath", label="🎙 Upload or Record Audio")
                language_dropdown = gr.Dropdown(
                                    choices=lang_list,
                                    value=lang_list[0],
                                    label="🌐 Select Language"
                                )
                transcribe_btn = gr.Button("🚀 Transcribe")
            with gr.Column():
              transcription_output = gr.Textbox(label="Transcription", lines=8,show_copy_button=True)
              with gr.Accordion("🎬 Subtitle (Not Accurate)", open=False):
                    sentence_srt_out = gr.File(label="Sentence-level Subtitle File")
                    word_srt_out = gr.File(label="Word-level Subtitle File")
                    shorts_srt_out = gr.File(label="Shorts Subtitle File")

        transcribe_btn.click(
            fn=transcribe_interface,
            inputs=[audio_input, language_dropdown],
            outputs=[transcription_output, sentence_srt_out, word_srt_out, shorts_srt_out]
        )

    return demo




import click

@click.command()
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode (shows detailed logs)."
)
@click.option(
    "--share",
    is_flag=True,
    default=False,
    help="Create a public Gradio share link (for Colab or remote usage)."
)
@click.option(
    "--model",
    default="omniASR_LLM_1B",
    type=click.Choice([
        "omniASR_CTC_300M",
        "omniASR_CTC_1B",
        "omniASR_CTC_3B",
        "omniASR_CTC_7B",
        "omniASR_LLM_300M",
        "omniASR_LLM_1B",
        "omniASR_LLM_3B",
        "omniASR_LLM_7B",
        "omniASR_LLM_7B_ZS",
    ]),
    help="Choose the OmniASR model to load."
)
def main(debug, share, model):
# def main(debug=True, share=True,model="omniASR_LLM_1B"):

    """Universal CLI entry point for omniASR transcription UI."""
    print(f"\n🚀 Starting omniASR UI with model: {model}")
    # ✅ Load model
    load_model(model)
    # ✅ Launch UI
    demo = ui()
    demo.queue().launch(share=share, debug=debug)

if __name__ == "__main__":
    main()



