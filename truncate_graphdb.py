import sqlite3

def truncate_dag(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    cursor = conn.cursor()

    # Temporary table to track nodes to remove
    cursor.execute("""
        CREATE TEMP TABLE nodes_to_remove AS
        SELECT n.id
        FROM nodes n
        LEFT JOIN edges e_in ON n.id = e_in.target
        LEFT JOIN edges e_out ON n.id = e_out.source
        GROUP BY n.id
        HAVING COUNT(DISTINCT e_in.source) = 1
           AND COUNT(DISTINCT e_out.target) = 1
           AND n.node_type = 'standard'
    """)

    removed = True
    while removed:
        # Count how many nodes to remove in this iteration
        cursor.execute("SELECT COUNT(*) FROM nodes_to_remove")
        removed_count = cursor.fetchone()[0]
        if removed_count == 0:
            removed = False
            break

        # Merge edges: from parent -> node -> child into parent -> child
        cursor.execute("""
            INSERT INTO edges (id, source, target, trajectory_id, iteration, logprobs_forward, logprobs_backward)
            SELECT
                e_in.trajectory_id || '_' || e_in.id AS id,
                e_in.source AS source,
                e_out.target AS target,
                e_in.trajectory_id,
                e_in.iteration,
                e_in.logprobs_forward + e_out.logprobs_forward AS logprobs_forward,
                e_in.logprobs_backward + e_out.logprobs_backward AS logprobs_backward
            FROM edges e_in
            JOIN edges e_out ON e_in.target = e_out.source
            JOIN nodes_to_remove n ON n.id = e_in.target
        """)

        # Remove the old edges through the removed nodes
        cursor.execute("""
            DELETE FROM edges
            WHERE source IN (SELECT id FROM nodes_to_remove)
               OR target IN (SELECT id FROM nodes_to_remove)
        """)

        # Remove the nodes themselves
        cursor.execute("""
            DELETE FROM nodes
            WHERE id IN (SELECT id FROM nodes_to_remove)
        """)

        # Recompute nodes_to_remove for next iteration
        cursor.execute("""
            DELETE FROM nodes_to_remove
        """)
        cursor.execute("""
            INSERT INTO nodes_to_remove
            SELECT n.id
            FROM nodes n
            LEFT JOIN edges e_in ON n.id = e_in.target
            LEFT JOIN edges e_out ON n.id = e_out.source
            GROUP BY n.id
            HAVING COUNT(DISTINCT e_in.source) = 1
               AND COUNT(DISTINCT e_out.target) = 1
               AND n.node_type = 'standard'
        """)

    conn.commit()
    conn.close()
    print("DAG truncation complete.")

if __name__ == "__main__":
    truncate_dag("traindata1/traindata1.db")
