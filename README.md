
# 🎙️ Meta Omnilingual ASR (Gradio App)

This project adapts the official **[facebook/omniasr-transcriptions](https://huggingface.co/spaces/facebook/omniasr-transcriptions)** by **Meta AI**,
which was originally built as a **Flask application**, into a **Gradio App**.

It allows you to run the **same Omnilingual ASR models** directly on **Google Colab** using the free **T4 GPU**  or on your **local system** if you want to test all models (`CTC` and `LLM` variants).

The main purpose of this project is to make the official Hugging Face demo **easier to run and experiment with** on Colab  no complex setup required.

> 🧠 **Credits:** All model logic and architecture belong to **Meta’s original [facebook/omniasr-transcriptions](https://huggingface.co/spaces/facebook/omniasr-transcriptions)**.
> This repository only converts their Flask-based app into a **Gradio version** for easier access and usage.

