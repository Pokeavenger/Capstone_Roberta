# 🤖 AI vs Human Text Classifier (RoBERTa)

This project fine-tunes **RoBERTa** to classify text as either:
- 🧑 Human-Written  
- 🤖 AI-Generated  

It was developed as a **Capstone Project** to explore the power of transformer-based models in detecting AI-generated content.

---

## 📌 Project Overview
With the rapid rise of LLMs like GPT and other AI text generators, distinguishing between human-written and AI-generated text is becoming crucial in education, research, and online authenticity.  
This project leverages **RoBERTa**, a transformer-based model, to build a binary text classifier.

---

## 🛠️ Features
- Fine-tuned **RoBERTa-base** model  
- Binary classification: `Human (0)` vs `AI (1)`  
- Deployed with **Gradio** for easy interaction  
- Model hosted on **Hugging Face Model Hub**  

---

## 📂 Dataset
The dataset used in training contains two columns:
- **Text** → the input text sample  
- **Generated** → label (`0 = Human`, `1 = AI`)  

---

## 🚀 Training
The model was fine-tuned on Google Colab using the Hugging Face `transformers` library.

**Steps:**
1. Load dataset (`Text`, `Generated`)  
2. Preprocess using Hugging Face `AutoTokenizer`  
3. Fine-tune RoBERTa with `Trainer` API  
4. Evaluate using Accuracy, Precision, Recall, F1-score  

---

## 📊 Results
Validation accuracy achieved: **~99%**  
