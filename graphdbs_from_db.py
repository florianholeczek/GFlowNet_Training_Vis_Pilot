import sqlite3

INPUT_DB = "traindata1/traindata1.db"
INPUT_TABLE = "points"


def create_tables(conn):
    """Create both nodes and edges tables in the same database"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            node_type TEXT,
            image TEXT,
            reward REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT,
            source TEXT,
            target TEXT,
            trajectory_id INTEGER,
            iteration INTEGER,
            logprobs_forward REAL,
            logprobs_backward REAL
        )
    """)
    conn.commit()


def insert_root_node(conn):
    conn.execute("""
        INSERT OR IGNORE INTO nodes (id, node_type, image, reward)
        VALUES ('#', 'start', NULL, NULL)
    """)
    conn.commit()


def main():
    conn = sqlite3.connect(INPUT_DB)

    create_tables(conn)
    insert_root_node(conn)

    read_cur = conn.cursor()  # Cursor for reading
    write_cur = conn.cursor()  # Cursor for writing

    seen_nodes = set(["#"])

    read_cur.execute(f"""
        SELECT
            final_id,
            step,
            final_object,
            text,
            iteration,
            total_reward,
            logprobs_forward,
            logprobs_backward,
            image
        FROM {INPUT_TABLE}
        ORDER BY final_id, step
    """)

    current_trajectory = None
    prev_row = None

    for row in read_cur:
        (
            trajectory_id,
            step,
            final_object,
            text,
            iteration,
            total_reward,
            logpf,
            logpb,
            image
        ) = row

        # --- new trajectory starts ---
        if trajectory_id != current_trajectory:
            current_trajectory = trajectory_id
            prev_row = None

        # --- insert node if unseen ---
        if text not in seen_nodes:
            node_type = "final" if final_object == 1 else "standard"
            reward = total_reward if final_object == 1 else None

            write_cur.execute("""
                INSERT INTO nodes (id, node_type, image, reward)
                VALUES (?, ?, ?, ?)
            """, (text, node_type, image, reward))

            seen_nodes.add(text)

        # --- create edge ---
        edge_id = f"{trajectory_id}_{step}"

        if prev_row is None:
            # root -> first node
            write_cur.execute("""
                INSERT INTO edges (
                    id,
                    source,
                    target,
                    trajectory_id,
                    iteration,
                    logprobs_forward,
                    logprobs_backward
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                edge_id,
                "#",
                text,
                trajectory_id,
                iteration,
                logpf,
                logpb
            ))
        else:
            prev_text = prev_row[3]

            write_cur.execute("""
                INSERT INTO edges (
                    id,
                    source,
                    target,
                    trajectory_id,
                    iteration,
                    logprobs_forward,
                    logprobs_backward
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                edge_id,
                prev_text,
                text,
                trajectory_id,
                iteration,
                logpf,
                logpb
            ))

        prev_row = row

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()