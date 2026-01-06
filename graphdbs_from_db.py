
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
    print("\nIdentifying nodes to keep...")

    # Identify all nodes that should be KEPT:
    # 1. start/final nodes
    # 2. nodes with != 1 parent OR != 1 child
    conn.execute("""
        CREATE TEMP TABLE nodes_to_keep AS
        WITH parent_counts AS (
            SELECT target AS node_id, COUNT(*) AS parent_count
            FROM edges
            GROUP BY target
        ),
        child_counts AS (
            SELECT source AS node_id, COUNT(*) AS child_count
            FROM edges
            GROUP BY source
        )
        SELECT DISTINCT n.id
        FROM nodes n
        LEFT JOIN parent_counts pc ON n.id = pc.node_id
        LEFT JOIN child_counts cc ON n.id = cc.node_id
        WHERE 
            n.node_type IN ('start', 'final')
            OR COALESCE(pc.parent_count, 0) != 1
            OR COALESCE(cc.child_count, 0) != 1
    """)

    kept_count = conn.execute("SELECT COUNT(*) FROM nodes_to_keep").fetchone()[0]
    total_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    removable_count = total_count - kept_count

    print(f"Nodes to keep: {kept_count}")
    print(f"Nodes to remove: {removable_count}")

    if removable_count == 0:
        print("No nodes to remove, graph is already truncated.")
        conn.execute("DROP TABLE nodes_to_keep")
        return

    print("\nRebuilding edges by following chains...")

    # Create new edges table to build into
    conn.execute("""
        CREATE TEMP TABLE edges_new (c
            id TEXT,
            source TEXT,
            target TEXT,
            trajectory_id INTEGER,
            iteration INTEGER,
            logprobs_forward REAL,
            logprobs_backward REAL
        )
    """)

    # For each edge that starts from a kept node, follow the chain
    # until we reach another kept node, accumulating logprobs
    conn.execute("""
        CREATE TEMP TABLE edges_to_process AS
        SELECT 
            e.id,
            e.source,
            e.target,
            e.trajectory_id,
            e.iteration,
            e.logprobs_forward,
            e.logprobs_backward
        FROM edges e
        WHERE e.source IN (SELECT id FROM nodes_to_keep)
    """)

    # Process each edge by following chains
    cursor = conn.cursor()
    edges_to_process = cursor.execute("SELECT * FROM edges_to_process").fetchall()

    processed = 0
    for edge_id, source, target, traj_id, iteration, logpf, logpb in edges_to_process:
        # Follow the chain from target until we hit a kept node
        current_target = target
        total_logpf = logpf
        total_logpb = logpb

        # Keep following while current target is not in nodes_to_keep
        while True:
            is_kept = conn.execute("""
                SELECT 1 FROM nodes_to_keep WHERE id = ?
            """, (current_target,)).fetchone()

            if is_kept:
                # We've reached a kept node, create the edge
                conn.execute("""
                    INSERT INTO edges_new (
                        id, source, target, trajectory_id, iteration,
                        logprobs_forward, logprobs_backward
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (edge_id, source, current_target, traj_id, iteration,
                      total_logpf, total_logpb))
                break

            # Current target is removable, follow to next node
            next_edge = conn.execute("""
                SELECT target, logprobs_forward, logprobs_backward
                FROM edges
                WHERE source = ?
            """, (current_target,)).fetchone()

            if not next_edge:
                # Dead end - shouldn't happen in valid graph
                print(f"Warning: Dead end at node {current_target}")
                break

            next_target, next_logpf, next_logpb = next_edge
            total_logpf += next_logpf
            total_logpb += next_logpb
            current_target = next_target

        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{len(edges_to_process)} edges...")

    print(f"Processed all {processed} edges")


    conn.execute("BEGIN")

    conn.execute("ALTER TABLE edges RENAME TO edges_old")
    conn.execute("ALTER TABLE edges_new RENAME TO edges")
    conn.execute("""
        DELETE FROM nodes 
        WHERE id NOT IN (SELECT id FROM nodes_to_keep)
    """)
    conn.execute("DROP TABLE nodes_to_keep")
    conn.execute("DROP TABLE edges_to_process")
    conn.execute("DROP TABLE edges_old")  # drop backup

    # Commit changes
    conn.commit()
    print("Truncation complete!")


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
