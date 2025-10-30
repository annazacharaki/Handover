-- Run automatically by app on first boot if DB is missing
CREATE TABLE IF NOT EXISTS logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  ip TEXT,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs (created_at);
CREATE INDEX IF NOT EXISTS idx_logs_user ON logs (user_id);
