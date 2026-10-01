import tempfile, pathlib, json
from chatbot import Chatbot, Session, extract_email, extract_name, extract_experience


def make():
    d = pathlib.Path(tempfile.mkdtemp())
    return Chatbot(reg_path=d / "r.json", log_path=None), Session(), d / "r.json"


def say(bot, s, *msgs):
    return [bot.handle(s, m) for m in msgs][-1]


def test_intents():
    bot, _, _ = make()
    for text, want in [("hello", "greeting"), ("I want to join the internship", "register"),
                       ("is it free?", "faq_cost"), ("what skills do I need", "faq_skills"),
                       ("how long is it", "faq_duration"), ("thanks a lot", "thanks")]:
        assert bot.classify(text)[0] == want, text


def test_no_substring_false_positive():
    bot, _, _ = make()
    assert bot.classify("this machine is white")[0] != "greeting"


def test_extractors():
    assert extract_email("mail me at Jaffar@Gmail.com pls")[0] == "jaffar@gmail.com"
    assert extract_name("my name is jaffar sadique and my email")[0] == "Jaffar Sadique"
    assert extract_name("12345")[0] is None
    assert extract_experience("I'm a total beginner")[0] == "Beginner"


def test_full_registration():
    bot, s, path = make()
    say(bot, s, "register")
    say(bot, s, "My name is Jaffar Sadique")
    say(bot, s, "jaffar@example.com")
    say(bot, s, "computer science")
    out = say(bot, s, "beginner")
    assert "Is this correct" in out and s.state == "CONFIRM"
    out = say(bot, s, "yes")
    assert "Registration confirmed" in out and s.state == "DONE"
    saved = json.loads(path.read_text())
    assert saved[0]["email"] == "jaffar@example.com" and saved[0]["field"] == "Computer Science"


def test_validation_and_faq_interrupt():
    bot, s, _ = make()
    say(bot, s, "register")
    assert "valid name" in say(bot, s, "!!!")
    say(bot, s, "Priya Sharma")
    assert "isn't valid" in say(bot, s, "priya@@mail")
    out = say(bot, s, "is it free?")
    assert "free" in out.lower() and "email address" in out and s.state == "ASK_EMAIL"


def test_duplicate_email_and_restart():
    bot, s, _ = make()
    for m in ["register", "Amit Kumar", "amit@example.com", "data science", "advanced", "yes"]:
        say(bot, s, m)
    s2 = Session()
    say(bot, s2, "register"); say(bot, s2, "Amit Two")
    assert "already registered" in say(bot, s2, "amit@example.com")
    assert s2.state == "ASK_EMAIL"
    say(bot, s2, "cancel")
    assert s2.state == "START"
