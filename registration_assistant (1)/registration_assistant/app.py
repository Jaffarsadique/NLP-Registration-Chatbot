"""Flask web interface + admin dashboard + analytics for the registration bot."""
import os
import uuid
from collections import Counter

from flask import Flask, jsonify, render_template, request, abort

from chatbot import Chatbot, Session

app = Flask(__name__)
bot = Chatbot()
sessions = {}
ADMIN_KEY = os.environ.get("ADMIN_KEY", "admin123")  # change this in production


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    sid = data.get("session_id") or uuid.uuid4().hex
    sess = sessions.setdefault(sid, Session())
    reply = bot.handle(sess, data.get("message", ""))
    return jsonify(reply=reply, session_id=sid, state=sess.state)


@app.get("/admin")
def admin():
    if request.args.get("key") != ADMIN_KEY:
        abort(403)
    regs = bot._load()
    stats = {
        "total": len(regs),
        "by_field": Counter(r["field"] for r in regs),
        "by_experience": Counter(r["experience"] for r in regs),
    }
    return jsonify(stats=stats, registrations=regs)


if __name__ == "__main__":
    app.run(debug=True)
