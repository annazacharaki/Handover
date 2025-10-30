# app.py
import os
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from datetime import timedelta

from flask import (
    Flask, request, session, redirect, url_for, render_template,
    jsonify, abort
)

APP_DIR = Path(__file__).parent.resolve()
DB_PATH = APP_DIR / "handover_web.db"
USERS_JSON = APP_DIR / "users.json"
SCHEMA_SQL = APP_DIR / "schema.sql"
COPILOT_PATH = APP_DIR / "handover_copilot.py"  # expected to exist

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", os.urandom(32))
app.permanent_session_lifetime = timedelta(hours=12)

# -------------------------------
# Utilities
# -------------------------------

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def ensure_db():
    # Always run schema.sql (safe due to IF NOT EXISTS)
    if not SCHEMA_SQL.exists():
        raise FileNotFoundError(f"schema.sql not found at: {SCHEMA_SQL}")
    with db() as con, open(SCHEMA_SQL, "r", encoding="utf-8") as f:
        con.executescript(f.read())

def load_users():
    # Safe load; create default if missing/empty/invalid
    default_data = {
        "users": [
            {"username": "user1", "password": "pass1"},
            {"username": "user2", "password": "pass2"},
        ]
    }
    try:
        if not USERS_JSON.exists() or USERS_JSON.stat().st_size == 0:
            USERS_JSON.write_text(json.dumps(default_data, ensure_ascii=False, indent=2), encoding="utf-8")
            return {u["username"]: u["password"] for u in default_data["users"]}

        with open(USERS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        users_list = data.get("users", [])
        if not isinstance(users_list, list) or not users_list:
            raise ValueError("`users` must be a non-empty list")

        return {u["username"]: u["password"] for u in users_list if "username" in u and "password" in u}

    except Exception as e:
        app.logger.warning("users.json invalid (%s). Rewriting a default users.json", e)
        USERS_JSON.write_text(json.dumps(default_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {u["username"]: u["password"] for u in default_data["users"]}

USERS = load_users()
ensure_db()

# -------------------------------
# Auth helpers
# -------------------------------

def current_user():
    return session.get("user")

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user():
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper

# -------------------------------
# Backend: call handover_copilot.py
# -------------------------------

def call_copilot(assembled_prompt: str, timeout_sec: int = 180) -> str:
    """Call the existing handover_copilot.py CLI with subcommand: ask <prompt>."""
    if not COPILOT_PATH.exists():
        raise FileNotFoundError(f"handover_copilot.py not found at: {COPILOT_PATH}")

    # Force UTF-8 mode for the child Python process and its stdio
    cmd = [sys.executable, "-X", "utf8", str(COPILOT_PATH), "ask", assembled_prompt]

    # Inherit env + enforce UTF-8 for stdio (windows-safe)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",   # decode stdout/stderr as UTF-8
            errors="replace",   # never crash on weird bytes
            timeout=timeout_sec,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("handover_copilot timed out") from e

    stdout = (res.stdout or "").strip()
    stderr = (res.stderr or "").strip()

    if res.returncode != 0:
        raise RuntimeError(
            f"handover_copilot failed (code {res.returncode}): {stderr or stdout}"
        )

    return stdout


# -------------------------------
# Routes
# -------------------------------

@app.get("/")
def root():
    if current_user():
        return redirect(url_for("chat"))
    return redirect(url_for("login"))

@app.get("/login")
def login():
    if current_user():
        return redirect(url_for("chat"))
    return render_template("login.html")

@app.post("/login")
def do_login():
    data = request.form or request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    real = USERS.get(username)
    if real and real == password:
        session.clear()
        session.permanent = True
        session["user"] = username
        session.setdefault("history", [])  # list of {role, content}
        return redirect(url_for("chat"))
    return render_template("login.html", error="Invalid credentials"), 401

@app.get("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.get("/chat")
@login_required
def chat():
    return render_template("chat.html", username=current_user())

@app.get("/api/me")
@login_required
def api_me():
    return jsonify({"username": current_user()})

@app.get("/api/history")
@login_required
def api_history():
    return jsonify(session.get("history", []))

@app.post("/api/ask")
@login_required
def api_ask():
    payload = request.get_json(force=True, silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        abort(400, description="Missing 'question'")

    # Build session memory for context
    history = session.setdefault("history", [])
    context_lines = [
        "You are the Handover Copilot. Continue the conversation.",
        "Use the prior Q&A for context. If information is missing, infer gently or ask for specifics.",
        "Conversation so far:"
    ]
    for msg in history[-20:]:
        role = msg.get("role")
        content = msg.get("content", "").replace("\n", " ")
        if role == "user":
            context_lines.append(f"User: {content}")
        else:
            context_lines.append(f"Assistant: {content}")

    context_lines.append(f"User: {question}")
    context_lines.append("Assistant:")
    assembled_prompt = "\n".join(context_lines)

    try:
        answer = call_copilot(assembled_prompt)
    except Exception as e:
        answer = f"[error] {type(e).__name__}: {e}"

    # Update memory
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    session["history"] = history

    # Log into SQLite (auto-heal if table missing)
    try:
        with db() as con:
            con.execute(
                "INSERT INTO logs (user_id, ip, question, answer) VALUES (?, ?, ?, ?)",
                (
                    current_user(),
                    request.headers.get("X-Forwarded-For", request.remote_addr),
                    question,
                    answer,
                ),
            )
    except sqlite3.OperationalError as e:
        if "no such table: logs" in str(e):
            ensure_db()
            with db() as con:
                con.execute(
                    "INSERT INTO logs (user_id, ip, question, answer) VALUES (?, ?, ?, ?)",
                    (
                        current_user(),
                        request.headers.get("X-Forwarded-For", request.remote_addr),
                        question,
                        answer,
                    ),
                )
        else:
            app.logger.exception("Failed to insert log: %s", e)

    return jsonify({"answer": answer})

@app.post("/api/clear")
@login_required
def api_clear():
    session["history"] = []
    return jsonify({"ok": True})

# Optional: avoid 404 spam for favicon
@app.get("/favicon.ico")
def favicon():
    return ("", 204)

# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    # Support PORT env var for Docker/Heroku-like envs
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
