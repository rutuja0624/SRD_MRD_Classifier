# 🚀 SRD vs MRD Classifier + Query Rewriter

A smart NLP pipeline that classifies user queries as:

* **SRD (Standalone Query)**
* **MRD (Multi-turn / Follow-up Query)**

and intelligently rewrites follow-up queries into **fully self-contained queries** using LLMs.

---

## 📌 Problem Statement

In conversational Text-to-SQL systems, users often ask:

> "Show sales for 2023"
> "What about 2024?"

The second query is **incomplete on its own**.

👉 This project solves that by:

1. Detecting whether a query depends on context
2. Rewriting it into a complete query using conversation history

---

## 🧠 Key Features

* ✅ SRD vs MRD classification using ML model
* ✅ Confidence-based decision making
* ✅ Context-aware query rewriting using LLM (Groq API)
* ✅ Conversation history tracking
* ✅ Feature engineering (TF-IDF + query length)
* ✅ Modular pipeline design

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* NumPy / SciPy
* Pickle
* Groq API (LLM - LLaMA 3)
* dotenv

---

## 📂 File Explanation

### 🔹 classifier.py

Main pipeline:

* Loads trained classifier
* Predicts SRD or MRD
* Uses LLM to resolve follow-up queries
* Maintains conversation history

### 🔹 predictor.py

Simple version:

* Only predicts SRD or MRD
* No LLM or history

### 🔹 srd_mrd_classifier.pkl

Contains:

* TF-IDF Vectorizer
* Trained ML model

### 🔹 .env.example

Stores API key template

### 🔹 .gitignore

Ignores:

* `.env`
* `__pycache__`
* `.pyc` files

---


## 🔄 How It Works

### Step 1: Classification

* Converts query into vector form
* Adds query length as feature
* Predicts SRD or MRD

### Step 2: Confidence Check

* If confidence < threshold → treat as SRD

### Step 3: MRD Resolution

* Uses conversation history
* Sends prompt to LLM
* Rewrites query

### Step 4: Output

Returns a **complete, standalone query**

---

## 💡 Example

**Input:**

```
User: Show sales in 2023
User: What about 2024?
```

**Output:**

```
Show sales in 2024
```

---
👩‍💻 Author

Rutuja Kamale
Computer Engineering Student
