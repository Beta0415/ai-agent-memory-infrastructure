from __future__ import annotations

import os

import pandas as pd
import psycopg
import streamlit as st
from dotenv import load_dotenv
from psycopg.rows import dict_row

from ai_agent_memory_infrastructure.memory_graph import memory_connections


def _mentions(database_url: str) -> list[tuple[str, str, str, str]]:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT edge.memory_id, memories.content,
                       entity.entity_id, entity.name
                FROM graph_edges AS edge
                JOIN memories ON memories.id = edge.memory_id
                JOIN graph_entities AS entity
                    ON entity.entity_id = edge.from_entity_id
                UNION
                SELECT edge.memory_id, memories.content,
                       entity.entity_id, entity.name
                FROM graph_edges AS edge
                JOIN memories ON memories.id = edge.memory_id
                JOIN graph_entities AS entity
                    ON entity.entity_id = edge.to_entity_id
                """
            )
            rows = cursor.fetchall()
    return [
        (
            str(row["memory_id"]),
            str(row["content"]),
            str(row["entity_id"]),
            str(row["name"]),
        )
        for row in rows
    ]


load_dotenv()

st.set_page_config(
    page_title="Show Graph Connection (Table)",
    page_icon="🧠",
    layout="wide",
)

st.title("Show Graph Connection (Table)")
if st.button("Back To Chat"):
    st.switch_page("app.py")

api_key = os.getenv("OPENAI_API_KEY", "").strip()
database_url = os.getenv(
    "DATABASE_URL", "postgresql://localhost:5432/agent_memory"
)
if not api_key:
    st.info("Add Your Key To `.env` As `OPENAI_API_KEY=...`")
    st.stop()

try:
    mentions = _mentions(database_url)
except Exception as exc:
    st.error(f"Graph Unavailable: {exc}")
    st.stop()

if not mentions:
    st.info("No Graph Connection Yet. Click Build Graph On The Chat Page")
    st.stop()

view = memory_connections(mentions, broad_limit=20)
contents = dict(view.memories)
if not view.groups:
    st.info("No Graph Connection Yet. Click Build Graph On The Chat Page")
    st.stop()

st.caption(
    "Each table is one entity. "
    "It lists every memory that names that entity. "
    "A memory can appear in more than one table."
)
for _entity_id, name, memory_ids in view.groups:
    st.subheader(f"{name} ({len(memory_ids)})")
    rows = sorted(contents[memory_id] for memory_id in memory_ids)
    table = pd.DataFrame(
        [[index, content] for index, content in enumerate(rows, start=1)],
        columns=["Index", "Memory"],
    )
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Index": st.column_config.NumberColumn(width="small"),
            "Memory": st.column_config.TextColumn(width="large"),
        },
    )
