from __future__ import annotations

import os

import pandas as pd
import psycopg
import streamlit as st
from dotenv import load_dotenv
from psycopg.rows import dict_row


def _episodes(
    database_url: str,
) -> list[tuple[str, tuple[str, ...], tuple[str, ...], tuple[float, float, float] | None]]:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT episode.episode_id, episode.title, episode.summary,
                       episode.cohesion, episode.minimum_cohesion,
                       episode.cohesion_stddev,
                       memories.id AS memory_id, memories.content,
                       entity.name AS entity_name
                FROM memory_episodes AS episode
                JOIN memory_episode_members AS member
                    ON member.episode_id = episode.episode_id
                JOIN memories ON memories.id = member.memory_id
                LEFT JOIN memory_episode_entities AS link
                    ON link.episode_id = episode.episode_id
                LEFT JOIN graph_entities AS entity
                    ON entity.entity_id = link.entity_id
                ORDER BY episode.title, episode.episode_id, memories.content,
                         entity.name
                """
            )
            rows = cursor.fetchall()
    grouped: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id not in grouped:
            order.append(episode_id)
            grouped[episode_id] = {
                "title": str(row["title"]),
                "summary": str(row["summary"]),
                "entities": set(),
                "memories": {},
                "cohesion": row["cohesion"],
                "minimum_cohesion": row["minimum_cohesion"],
                "cohesion_stddev": row["cohesion_stddev"],
            }
        item = grouped[episode_id]
        entities = item["entities"]
        memories = item["memories"]
        assert isinstance(entities, set)
        assert isinstance(memories, dict)
        if row["entity_name"] is not None:
            entities.add(str(row["entity_name"]))
        memories[str(row["memory_id"])] = str(row["content"])
    episodes: list[
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[float, float, float] | None,
        ]
    ] = []
    for episode_id in order:
        item = grouped[episode_id]
        entities = item["entities"]
        memories = item["memories"]
        assert isinstance(entities, set)
        assert isinstance(memories, dict)
        scores = None
        if item["cohesion"] is not None:
            scores = (
                float(item["cohesion"]),
                float(item["minimum_cohesion"]),
                float(item["cohesion_stddev"]),
            )
        episodes.append(
            (
                str(item["title"]),
                str(item["summary"]),
                tuple(sorted(entities, key=str.casefold)),
                tuple(memories[memory_id] for memory_id in memories),
                scores,
            )
        )
    return episodes


load_dotenv()

st.set_page_config(
    page_title="Show Episodes",
    page_icon="🧠",
    layout="wide",
)

st.title("Show Episodes")
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
    episodes = _episodes(database_url)
except Exception as exc:
    st.error(f"Episodes Unavailable: {exc}")
    st.stop()

if not episodes:
    st.info("No Episodes Yet. Click Build Episodes On The Chat Page")
    st.stop()

st.caption(
    "Each table is one episode. "
    "It lists the atomic memories in that episode. "
    "A memory can appear in more than one episode."
)
for title, summary, entity_names, memories, scores in episodes:
    st.subheader(f"{title} ({len(memories)})")
    if summary:
        st.markdown(summary)
    if entity_names:
        st.caption("Joined By " + ", ".join(entity_names))
    if scores is not None:
        cohesion, minimum, deviation = scores
        st.caption(
            f"Cohesion {cohesion:.2f} · "
            f"Minimum {minimum:.2f} · "
            f"Deviation {deviation:.2f}"
        )
    table = pd.DataFrame(
        [[index, content] for index, content in enumerate(memories, start=1)],
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
