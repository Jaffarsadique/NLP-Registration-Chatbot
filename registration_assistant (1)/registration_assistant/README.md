# AI Registration Assistant (Task AI-SS-001)

A chatbot that guides students through internship registration for the Free Online AI & Data Science Internship (Data Alcott Systems).

## Features
- Intent recognition: NLTK-lemmatized text, TF-IDF + Logistic Regression (scikit-learn), trained from `intents.json`
- Entity extraction: name, email, field of study, experience level (regex)
- Validation: name and email format, duplicate-email check
- Dialog manager: state machine `ASK_NAME > ASK_EMAIL > ASK_FIELD > ASK_EXPERIENCE > CONFIRM > DONE`, with fallbacks, re-prompts, and FAQ interruptions mid-registration
- Storage: `registrations.json` (no database)
- Bonus: FAQ handling, simple sentiment analysis, chat logging (`chat_log.jsonl`), Flask web UI, admin dashboard (`/admin?key=...`)

## Setup
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python chatbot.py          # console chat
python app.py              # web UI at http://127.0.0.1:5000
python -m pytest -q        # tests
```
Admin stats: `http://127.0.0.1:5000/admin?key=admin123` (set `ADMIN_KEY` env var to change).

## Files
| File | Purpose |
|---|---|
| `chatbot.py` | NLP pipeline, entities, dialog manager, storage |
| `intents.json` | Intent patterns and responses (edit to add FAQs) |
| `app.py`, `templates/index.html` | Flask API and chat UI |
| `test_chatbot.py` | Unit and flow tests |

## Sample conversation
```
You: hi, I want to register
Assistant: Great, let's get you registered. Please tell me your full name.
You: My name is Jaffar Sadique
Assistant: Nice to meet you, Jaffar Sadique! What is your email address?
You: is it free?
Assistant: This is a free online internship, so there is no registration fee.
         What is your email address?
```

## Extending
Add intents in `intents.json` (retrains at startup); swap the classifier for a BERT model or add translation for multi-language support.
