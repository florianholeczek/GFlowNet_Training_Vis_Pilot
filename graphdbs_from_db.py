
def create_tables(conn):
    """Create both nodes and edges tables in the same database"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            node_type TEXT,
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
        INSERT OR IGNORE INTO nodes (id, node_type, reward)
        VALUES ('#', 'start', NULL)
    """)
    conn.commit()


def create_indexes(conn):
    """Create indexes for efficient truncation queries"""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
    conn.commit()


def truncate_graph(conn):
    """
    Identify nodes to keep, then rebuild edges by following chains through removable nodes.
    Much faster than iterative removal.
    """
    cursor = conn.cursor()

    # Step 1: Identify removable nodes
    # A node is removable if it has exactly 1 unique predecessor AND 1 unique successor
    print("Step 1: Identifying removable nodes...")
    cursor.execute("""
            CREATE TEMP TABLE IF NOT EXISTS removable_nodes AS
            WITH node_connections AS (
                SELECT 
                    n.id,
                    COUNT(DISTINCT e_in.source) as num_predecessors,
                    COUNT(DISTINCT e_out.target) as num_successors
                FROM nodes n
                LEFT JOIN edges e_in ON n.id = e_in.target
                LEFT JOIN edges e_out ON n.id = e_out.source
                GROUP BY n.id
            )
            SELECT id
            FROM node_connections
            WHERE num_predecessors = 1 AND num_successors = 1 AND type != "final"
        """)

    removable_count = cursor.execute("SELECT COUNT(*) FROM removable_nodes").fetchone()[0]
    print(f"  Found {removable_count} removable nodes")

    if removable_count == 0:
        print("No nodes to remove. Exiting.")
        return

    # Step 2: Create lookup table mapping each removable node to its unique successor
    print("Step 2: Creating successor lookup table...")
    cursor.execute("""
            CREATE TEMP TABLE IF NOT EXISTS node_successors AS
            SELECT DISTINCT
                e.source as node_id,
                e.target as successor_id
            FROM edges e
            WHERE e.source IN (SELECT id FROM removable_nodes)
        """)

    # Step 3: Iteratively bypass removable nodes
    print("Step 3: Bypassing chains...")
    iteration = 0
    max_iterations = 1000  # Safety limit

    while iteration < max_iterations:
        iteration += 1
        cursor.execute("DROP TABLE IF EXISTS edges_to_delete")

        # Materialize edges to update/delete
        cursor.execute("""
            CREATE TEMP TABLE edges_to_delete AS
            SELECT e.source, e.target, e.trajectory_id
            FROM edges e
            WHERE e.source NOT IN (SELECT id FROM removable_nodes)
              AND e.target IN (SELECT id FROM removable_nodes)
        """)

        cursor.execute("SELECT COUNT(*) FROM edges_to_delete")
        edges_to_update = cursor.fetchone()[0]

        if edges_to_update == 0:
            print(f"  Completed in {iteration - 1} iterations")
            cursor.execute("DROP TABLE edges_to_delete")
            break

        print(f"  Iteration {iteration}: Updating {edges_to_update} edges...")

        # Insert bypass edges
        cursor.execute("""
            INSERT INTO edges (
                id,
                source,
                target,
                trajectory_id,
                iteration,
                logprobs_forward,
                logprobs_backward
            )
            SELECT 
                e.id,
                e.source,
                ns.successor_id AS target,
                e.trajectory_id,
                e.iteration,
                e.logprobs_forward + COALESCE(e_next.logprobs_forward, 0),
                e.logprobs_backward + COALESCE(e_next.logprobs_backward, 0)
            FROM edges e
            INNER JOIN edges_to_delete etd
                ON e.source = etd.source
               AND e.target = etd.target
               AND e.trajectory_id = etd.trajectory_id
            INNER JOIN node_successors ns
                ON e.target = ns.node_id
            LEFT JOIN edges e_next
                ON e_next.source = e.target
               AND e_next.target = ns.successor_id
               AND e_next.trajectory_id = e.trajectory_id
        """)

        # Delete only the original edges
        cursor.execute("""
            DELETE FROM edges
            WHERE (source, target, trajectory_id) IN (
                SELECT source, target, trajectory_id
                FROM edges_to_delete
            )
        """)

        cursor.execute("DROP TABLE edges_to_delete")
        conn.commit()

    if iteration >= max_iterations:
        print(f"  Warning: Reached maximum iterations ({max_iterations})")

    # Step 4: Handle edges between removable nodes (internal chain edges)
    print("Step 4: Cleaning up internal chain edges...")
    cursor.execute("""
            DELETE FROM edges
            WHERE source IN (SELECT id FROM removable_nodes)
               OR target IN (SELECT id FROM removable_nodes)
        """)

    # Step 5: Remove nodes from nodes table
    print("Step 5: Removing nodes...")
    nodes_deleted = cursor.execute("""
            DELETE FROM nodes
            WHERE id IN (SELECT id FROM removable_nodes)
        """).rowcount
    print(f"  Deleted {nodes_deleted} nodes")

    # Step 6: Clean up temporary tables
    print("Step 6: Cleaning up temporary tables...")
    cursor.execute("DROP TABLE IF EXISTS removable_nodes")
    cursor.execute("DROP TABLE IF EXISTS node_successors")

    conn.commit()
    print("Compression complete!")

    # Print summary statistics
    remaining_nodes = cursor.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    remaining_edges = cursor.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    print(f"\nSummary:")
    print(f"  Remaining nodes: {remaining_nodes}")
    print(f"  Remaining edges: {remaining_edges}")


def print_stats(conn):
    """Print graph statistics"""
    node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    node_types = conn.execute("""
        SELECT node_type, COUNT(*) 
        FROM nodes 
        GROUP BY node_type
    """).fetchall()

    print("\n=== Graph Statistics ===")
    print(f"Total nodes: {node_count}")
    print(f"Total edges: {edge_count}")
    print("Node types:")
    for node_type, count in node_types:
        print(f"  {node_type}: {count}")


def create_graph_dbs(conn):

    create_tables(conn)
    insert_root_node(conn)

    read_cur = conn.cursor()  # Cursor for reading
    write_cur = conn.cursor()  # Cursor for writing

    seen_nodes = set(["#"])

    print("Building initial graph...")
    read_cur.execute(f"""
        SELECT
            final_id,
            step,
            final_object,
            text,
            iteration,
            total_reward,
            logprobs_forward,
            logprobs_backward
        FROM trajectories
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
        ) = row

        # --- new trajectory starts ---
        if trajectory_id != current_trajectory:
            current_trajectory = trajectory_id
            prev_row = None

        # --- insert or update node ---
        if text not in seen_nodes:
            node_type = "final" if final_object == 1 else "standard"
            reward = total_reward if final_object == 1 else None

            write_cur.execute("""
                INSERT OR IGNORE INTO nodes (id, node_type, reward)
                VALUES (?, ?, ?)
            """, (text, node_type, reward))

            seen_nodes.add(text)
        elif final_object == 1:
            # Node exists but this is a final occurrence - update it
            write_cur.execute("""
                UPDATE nodes 
                SET node_type = 'final', reward = ?
                WHERE id = ?
            """, (total_reward, text))

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
    print("Initial graph built successfully")

    print_stats(conn)

    # Create indexes for efficient truncation
    print("\nCreating indexes...")
    create_indexes(conn)

    # Perform truncation
    print("\nStarting truncation...")
    truncate_graph(conn)

    print_stats(conn)

    print("\nDone!")


if __name__ == "__main__":
    import sqlite3
    # Connect to your database
    conn = sqlite3.connect("debugdata/data.db")
    conn.execute("DROP TABLE IF EXISTS edges")
    conn.execute("DROP TABLE IF EXISTS nodes")
    conn.commit()

    try:
        create_graph_dbs(conn)
    finally:
        conn.close()