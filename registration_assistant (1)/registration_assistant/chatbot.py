"""AI Registration Assistant (Task AI-SS-001).

Pipeline: preprocess -> TF-IDF + Logistic Regression intent classifier
          -> regex entity extraction + validation -> state-machine dialog manager
          -> JSON storage.  Run `python chatbot.py` for a console chat.
"""
import json
import pathlib
import random
import re
import uuid
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

BASE = pathlib.Path(__file__).parent

# ---------------------------------------------------------------- NLP basics
try:  # NLTK lemmatizer (falls back to plain tokens if WordNet is unavailable)
    import nltk
    from nltk.stem import WordNetLemmatizer
    _lem = WordNetLemmatizer()
    try:
        _lem.lemmatize("tests")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        _lem.lemmatize("tests")
except Exception:
    _lem = None

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def preprocess(text: str) -> str:
    """Lowercase, mask emails, tokenize, lemmatize. Stop words are kept
    because words like 'how', 'is', 'do' carry intent in short questions."""
    text = EMAIL_RE.sub(" emailaddr ", text.lower())
    tokens = re.findall(r"[a-z]+", text)
    if _lem:
        tokens = [_lem.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------- entity extraction/validation
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,48}[A-Za-z.]$")
NAME_PHRASE = re.compile(r"(?:my name is|i am|i'm|im|call me|this is|name:)\s+([A-Za-z .'-]+)", re.I)
FIELDS = {
    r"computer science|\bcse?\b": "Computer Science",
    r"information technology|\bit\b": "Information Technology",
    r"data science": "Data Science",
    r"artificial intelligence|\bai\b": "Artificial Intelligence",
    r"electronics|\bece\b|\beee\b": "Electronics & Electrical",
    r"electrical": "Electrical Engineering",
    r"mechanical": "Mechanical Engineering",
    r"civil": "Civil Engineering",
    r"mathematics|maths?": "Mathematics",
    r"statistics": "Statistics",
    r"commerce|business|\bbba\b|\bmba\b": "Commerce & Business",
    r"physics": "Physics",
}


def extract_name(text):
    """Return (value, explicit). explicit=True when the user said 'my name is ...'."""
    m = NAME_PHRASE.search(text)
    if m:
        raw = re.split(r"\b(?:and|my|email|from|studying)\b", m.group(1), flags=re.I)[0]
        raw = " ".join(raw.split())
        if valid_name(raw):
            return raw.title(), True
        return None, False
    raw = " ".join(text.split())
    return (raw.title(), False) if valid_name(raw) else (None, False)


def valid_name(n: str) -> bool:
    return bool(NAME_RE.match(n)) and len(n.split()) <= 5


def extract_email(text):
    m = EMAIL_RE.search(text)
    return (m.group().lower(), True) if m else (None, False)


def extract_field(text):
    low = text.lower()
    for pat, canon in FIELDS.items():
        if re.search(pat, low):
            return canon, True
    cleaned = re.sub(r"^(i (am )?(study|studying)|my (field|degree|major) is|i'm studying|studying)\s+", "", low).strip()
    if re.fullmatch(r"[a-z &,\-]{2,60}", cleaned) and len(cleaned.split()) <= 6:
        return cleaned.title(), False
    return None, False


def extract_experience(text):
    low = text.lower()
    if re.search(r"\b(advanced|expert|professional|senior)\b", low):
        return "Advanced", True
    if re.search(r"\b(intermediate|moderate|some experience|[1-9] years?|good)\b", low):
        return "Intermediate", True
    if re.search(r"\b(beginner|basic|novice|new|none|no experience|just started|learning|fresher)\b", low):
        return "Beginner", True
    return None, False


# -------------------------------------------------------------- sentiment (bonus)
NEG = {"confused", "frustrated", "angry", "annoying", "stuck", "worst", "hate", "useless", "terrible", "upset", "waste", "bad"}
POS = {"great", "awesome", "good", "excellent", "love", "nice", "perfect", "happy", "excited"}


def sentiment(text: str) -> int:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & POS) - len(words & NEG)


# ------------------------------------------------------------------ dialog data
PROMPTS = {
    "ASK_NAME": "Please tell me your full name.",
    "ASK_EMAIL": "What is your email address?",
    "ASK_FIELD": "What is your field of study (for example Computer Science)?",
    "ASK_EXPERIENCE": "How would you rate your programming experience: Beginner, Intermediate or Advanced?",
}
ERRORS = {
    "ASK_NAME": "That doesn't look like a valid name. Use letters only, e.g. 'My name is Priya Sharma'.",
    "ASK_EMAIL": "That email isn't valid. Enter it like name@example.com.",
    "ASK_FIELD": "I couldn't read that. Enter your field of study, e.g. Computer Science.",
    "ASK_EXPERIENCE": "Please choose Beginner, Intermediate or Advanced.",
}
FLOW = ["ASK_NAME", "ASK_EMAIL", "ASK_FIELD", "ASK_EXPERIENCE"]
SLOT = {
    "ASK_NAME": ("name", extract_name),
    "ASK_EMAIL": ("email", extract_email),
    "ASK_FIELD": ("field", extract_field),
    "ASK_EXPERIENCE": ("experience", extract_experience),
}
ACK = {
    "name": "Nice to meet you, {v}!",
    "email": "Got it, I've recorded {v}.",
    "field": "Great, {v} it is.",
    "experience": "Noted: {v} level.",
}
INTERRUPTS = {"greeting", "help", "thanks", "goodbye", "status",
              "faq_duration", "faq_cost", "faq_skills", "faq_certificate", "faq_mode"}
THRESHOLD = 0.30


class Session:
    def __init__(self):
        self.state = "START"
        self.data = {}
        self.registration_id = None


class Chatbot:
    def __init__(self, reg_path=BASE / "registrations.json", log_path=BASE / "chat_log.jsonl"):
        self.intents = json.loads((BASE / "intents.json").read_text(encoding="utf-8"))["intents"]
        self.reg_path = pathlib.Path(reg_path)
        self.log_path = pathlib.Path(log_path) if log_path else None
        self._train()

    # ---- intent classification
    def _train(self):
        X, y = [], []
        for name, d in self.intents.items():
            for p in d["patterns"]:
                X.append(preprocess(p))
                y.append(name)
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                                   LogisticRegression(C=20, max_iter=1000))
        self.model.fit(X, y)

    def classify(self, text):
        probs = self.model.predict_proba([preprocess(text)])[0]
        i = probs.argmax()
        return self.model.classes_[i], float(probs[i])

    def _say(self, intent):
        return random.choice(self.intents[intent]["responses"])

    # ---- storage
    def _load(self):
        if self.reg_path.exists():
            return json.loads(self.reg_path.read_text(encoding="utf-8"))
        return []

    def _email_taken(self, email):
        return any(r["email"] == email for r in self._load())

    def _save(self, s):
        regs = self._load()
        s.registration_id = "DAS-" + uuid.uuid4().hex[:8].upper()
        regs.append({"id": s.registration_id, **s.data, "status": "Registered",
                     "timestamp": datetime.now().isoformat(timespec="seconds")})
        self.reg_path.write_text(json.dumps(regs, indent=2), encoding="utf-8")

    # ---- main entry point
    def handle(self, s: Session, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "Please type a message."
        intent, conf = self.classify(text)
        if conf < THRESHOLD:
            intent = "unknown"
        reply = self._route(s, text, intent)
        mood = sentiment(text)
        if mood < 0:
            reply = "Sorry this is frustrating. Let's sort it out together. " + reply
        self._log(s, text, intent, conf, mood)
        return reply

    def _route(self, s, text, intent):
        if intent == "cancel":
            s.state, s.data, s.registration_id = "START", {}, None
            return "Cleared. Say 'register' whenever you're ready."
        if s.state in FLOW:
            return self._flow(s, text, intent)
        if s.state == "CONFIRM":
            return self._confirm(s, text, intent)
        if intent == "register":
            if s.state == "DONE":
                return f"You're already registered (ID {s.registration_id}). Say 'cancel' to register someone else."
            s.state = "ASK_NAME"
            return self._say("register") + " " + PROMPTS["ASK_NAME"]
        return self._general(s, text, intent)

    def _general(self, s, text, intent):
        if intent == "status":
            email, _ = extract_email(text)
            if email:
                for r in self._load():
                    if r["email"] == email:
                        return f"Registration {r['id']} for {r['name']}: {r['status']}."
                return "I couldn't find a registration with that email."
            if s.registration_id:
                return f"Your registration {s.registration_id} is complete (status: Registered)."
            return "I don't see a registration yet. Say 'register' to start, or send your email to look one up."
        if intent in self.intents and intent not in {"affirm", "deny", "register"}:
            return self._say(intent)
        if intent in {"affirm", "deny"}:
            return "Would you like to register for the internship? Say 'register' to begin."
        return "I'm not sure I understood. You can say 'register', ask about duration, cost or skills, or type 'help'."

    def _flow(self, s, text, intent):
        key, extractor = SLOT[s.state]
        value, explicit = extractor(text)
        if not (explicit and value):
            if intent in INTERRUPTS:
                return self._general(s, text, intent) + "\n\n" + PROMPTS[s.state]
            if not value:
                return ERRORS[s.state]
        if key == "email" and self._email_taken(value):
            return "That email is already registered. Use a different email, or ask me for your status."
        s.data[key] = value
        idx = FLOW.index(s.state)
        ack = ACK[key].format(v=value)
        if idx + 1 < len(FLOW):
            s.state = FLOW[idx + 1]
            return f"{ack} {PROMPTS[s.state]}"
        s.state = "CONFIRM"
        return f"{ack}\n\n{self._summary(s)}\n\nIs this correct? (yes / no)"

    def _confirm(self, s, text, intent):
        low = text.lower()
        if re.search(r"\b(yes|yeah|yep|confirm|correct|ok|okay|sure)\b", low) or intent == "affirm":
            self._save(s)
            s.state = "DONE"
            return (f"Registration confirmed!\nID: {s.registration_id}\nName: {s.data['name']}\n"
                    f"A confirmation will be sent to {s.data['email']}. Good luck with the internship!")
        if re.search(r"\b(no|nope|change|wrong|edit)\b", low) or intent == "deny":
            s.state, s.data = "ASK_NAME", {}
            return "No problem, let's redo it. " + PROMPTS["ASK_NAME"]
        if intent in INTERRUPTS:
            return self._general(s, text, intent) + "\n\nPlease reply yes to confirm or no to start again."
        return "Please reply 'yes' to confirm or 'no' to start again."

    @staticmethod
    def _summary(s):
        d = s.data
        return (f"Please review your details:\n- Name: {d['name']}\n- Email: {d['email']}\n"
                f"- Field of study: {d['field']}\n- Experience: {d['experience']}")

    def _log(self, s, text, intent, conf, mood):
        if not self.log_path:
            return
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "state": s.state,
                 "input": text, "intent": intent, "confidence": round(conf, 3), "sentiment": mood}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    bot, sess = Chatbot(), Session()
    print("Assistant: Hello! Say 'register' to begin (type 'quit' to exit).")
    while True:
        msg = input("You: ")
        if msg.strip().lower() in {"quit", "exit"}:
            print("Assistant: Goodbye!")
            break
        print("Assistant:", bot.handle(sess, msg))
