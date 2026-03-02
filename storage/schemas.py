"""SQLite table definitions for conviction state persistence."""

CONVICTION_STATES_TABLE = """
CREATE TABLE IF NOT EXISTS conviction_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    conviction REAL NOT NULL,
    conviction_prev REAL NOT NULL DEFAULT 0.0,
    expected_return REAL DEFAULT 0.0,
    idiosyncratic_vol REAL DEFAULT 0.0,
    alpha_t REAL DEFAULT 0.0,
    fe REAL,
    fvs REAL,
    rrs REAL,
    its REAL,
    total_loss REAL,
    gradient_value REAL,
    learning_rate REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(instrument_id, as_of_date)
)
"""

EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id TEXT NOT NULL,
    event_date TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity REAL NOT NULL,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

IC_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS ic_sessions (
    session_id TEXT PRIMARY KEY,
    instrument_id TEXT NOT NULL,
    session_date TEXT NOT NULL,
    status TEXT NOT NULL,
    red_team_analyst TEXT,
    notes TEXT DEFAULT '',
    its_value REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

PROBABILITY_SUBMISSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS probability_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    analyst_id TEXT NOT NULL,
    p_pre REAL,
    p_post REAL,
    submitted_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES ic_sessions(session_id)
)
"""

TRACK_RECORDS_TABLE = """
CREATE TABLE IF NOT EXISTS track_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analyst_id TEXT NOT NULL,
    evaluation_date TEXT NOT NULL,
    brier_score REAL,
    mean_update_alignment REAL,
    n_forecasts INTEGER DEFAULT 0,
    bias_direction TEXT DEFAULT 'aligned',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(analyst_id, evaluation_date)
)
"""

TARGET_WEIGHTS_TABLE = """
CREATE TABLE IF NOT EXISTS target_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    conviction REAL NOT NULL,
    raw_weight REAL NOT NULL,
    constrained_weight REAL NOT NULL,
    sizing_method TEXT DEFAULT 'basic',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(instrument_id, as_of_date)
)
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_cs_instrument_date"
    " ON conviction_states(instrument_id, as_of_date)",
    "CREATE INDEX IF NOT EXISTS idx_events_instrument ON events(instrument_id, event_date)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_instrument ON ic_sessions(instrument_id)",
    "CREATE INDEX IF NOT EXISTS idx_submissions_session ON probability_submissions(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_track_analyst ON track_records(analyst_id)",
    "CREATE INDEX IF NOT EXISTS idx_weights_date ON target_weights(as_of_date)",
]

ALL_TABLES = [
    CONVICTION_STATES_TABLE,
    EVENTS_TABLE,
    IC_SESSIONS_TABLE,
    PROBABILITY_SUBMISSIONS_TABLE,
    TRACK_RECORDS_TABLE,
    TARGET_WEIGHTS_TABLE,
]
