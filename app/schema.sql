CREATE TABLE IF NOT EXISTS lab_schema_version (
    version integer PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS assistant_users (
    user_id bigint PRIMARY KEY,
    mode text NOT NULL DEFAULT 'study' CHECK (mode IN ('study', 'translate', 'review')),
    temperature numeric(2,1) NOT NULL DEFAULT 0.3
        CHECK (temperature IN (0.0, 0.3, 0.7, 1.0))
);
CREATE TABLE IF NOT EXISTS assistant_turns (
    id bigserial PRIMARY KEY,
    user_id bigint NOT NULL REFERENCES assistant_users(user_id) ON DELETE CASCADE,
    user_text text NOT NULL,
    assistant_text text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS assistant_turns_user_order ON assistant_turns (user_id, id DESC);
INSERT INTO lab_schema_version (version) VALUES (1) ON CONFLICT DO NOTHING;
