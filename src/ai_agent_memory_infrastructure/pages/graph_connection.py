from __future__ import annotations

import math
import os

import plotly.graph_objects as go
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


_CLUSTER_COLORS = (
    "#8ec5ff",
    "#f0b429",
    "#7dcea0",
    "#f5a3c7",
    "#c9b6ff",
    "#f2c14e",
    "#79d2c4",
    "#ffb085",
    "#b7d67d",
    "#9ec1ff",
)


def _on_sphere(index: int, count: int, radius: float) -> list[float]:
    if count <= 1:
        return [0.0, 0.0, 0.0]
    if count == 2:
        side = -radius if index == 0 else radius
        return [side, 0.0, 0.0]
    height = 1 - (index / (count - 1)) * 2
    ring = math.sqrt(max(1 - height * height, 0.0))
    angle = math.pi * (3 - math.sqrt(5)) * index
    return [
        math.cos(angle) * ring * radius,
        height * radius,
        math.sin(angle) * ring * radius,
    ]


def _layout(memories, links, hubs=()):
    node_ids = [memory_id for memory_id, _content in memories]
    parent = {memory_id: memory_id for memory_id in node_ids}

    def find(memory_id: str) -> str:
        while parent[memory_id] != memory_id:
            parent[memory_id] = parent[parent[memory_id]]
            memory_id = parent[memory_id]
        return memory_id

    def union(left: str, right: str) -> None:
        parent[find(left)] = find(right)

    for left, right, _names in links:
        union(left, right)
    for _entity_id, _name, memory_ids in hubs:
        present = [memory_id for memory_id in memory_ids if memory_id in parent]
        for memory_id in present[1:]:
            union(present[0], memory_id)
    grouped: dict[str, list[str]] = {}
    for memory_id in node_ids:
        grouped.setdefault(find(memory_id), []).append(memory_id)
    groups = sorted(grouped.values(), key=len, reverse=True)
    clusters = [group for group in groups if len(group) > 1]
    singles = [group[0] for group in groups if len(group) == 1]
    position: dict[str, list[float]] = {}
    group_of: dict[str, int] = {}
    cursor = 0.0
    for group_index, group in enumerate(clusters):
        count = len(group)
        radius = 2.2 * math.sqrt(count)
        center = cursor + radius
        cursor += (radius * 2) + 5
        for index, memory_id in enumerate(group):
            group_of[memory_id] = group_index
            point = _on_sphere(index, count, radius)
            position[memory_id] = [
                center + point[0],
                point[1],
                point[2],
            ]
    single_columns = max(1, math.ceil(math.sqrt(len(singles))))
    for index, memory_id in enumerate(singles):
        group_of[memory_id] = len(clusters) + index
        position[memory_id] = [
            cursor + (index % single_columns) * 2.2,
            0.0,
            (index // single_columns) * 2.2,
        ]
    for index, (entity_id, _name, memory_ids) in enumerate(hubs):
        present = [memory_id for memory_id in memory_ids if memory_id in position]
        if not present:
            continue
        center = [
            sum(position[memory_id][axis] for memory_id in present) / len(present)
            for axis in range(3)
        ]
        offset = (index - (len(hubs) - 1) / 2) * 1.8
        position[f"hub:{entity_id}"] = [
            center[0] + offset,
            center[1],
            center[2],
        ]
    clustered = {memory_id for group in clusters for memory_id in group}
    return position, group_of, clustered


def _figure(memories, links, positions, group_of, clustered, hubs=()):
    figure = go.Figure()
    for left, right, names in links:
        entity_label = " · ".join(names)
        midpoint = [
            (positions[left][axis] + positions[right][axis]) / 2
            for axis in range(3)
        ]
        figure.add_trace(
            go.Scatter3d(
                x=[positions[left][0], positions[right][0]],
                y=[positions[left][1], positions[right][1]],
                z=[positions[left][2], positions[right][2]],
                mode="lines",
                line={"color": "#d5dde6", "width": 5},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[midpoint[0]],
                y=[midpoint[1]],
                z=[midpoint[2]],
                mode="markers",
                marker={"size": 5, "color": "#f4d35e", "symbol": "diamond"},
                customdata=[["edge", entity_label]],
                hovertext=[entity_label],
                hoverinfo="text",
                showlegend=False,
            )
        )
    for entity_id, _name, memory_ids in hubs:
        hub_id = f"hub:{entity_id}"
        if hub_id not in positions:
            continue
        point = positions[hub_id]
        for memory_id in memory_ids:
            if memory_id not in positions:
                continue
            figure.add_trace(
                go.Scatter3d(
                    x=[positions[memory_id][0], point[0]],
                    y=[positions[memory_id][1], point[1]],
                    z=[positions[memory_id][2], point[2]],
                    mode="lines",
                    line={"color": "#f4d35e", "width": 3},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    node_x = []
    node_y = []
    node_z = []
    node_color = []
    node_custom = []
    node_hover = []
    for memory_id, text in memories:
        node_x.append(positions[memory_id][0])
        node_y.append(positions[memory_id][1])
        node_z.append(positions[memory_id][2])
        color_index = group_of[memory_id] % len(_CLUSTER_COLORS)
        node_color.append(
            _CLUSTER_COLORS[color_index] if memory_id in clustered else "#8b93a1"
        )
        node_custom.append(["memory", text])
        node_hover.append(text)
    figure.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker={"size": 9, "color": node_color},
            customdata=node_custom,
            hovertext=node_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    for entity_id, name, memory_ids in hubs:
        hub_id = f"hub:{entity_id}"
        if hub_id not in positions:
            continue
        point = positions[hub_id]
        figure.add_trace(
            go.Scatter3d(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                mode="markers",
                marker={"size": 16, "color": "#fff4c2", "symbol": "diamond"},
                customdata=[["hub", name]],
                hovertext=[name],
                hoverinfo="text",
                showlegend=False,
            )
        )
    figure.update_layout(
        height=760,
        paper_bgcolor="#0e1117",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={
            "bgcolor": "#161b22",
            "xaxis": {"visible": False, "showbackground": False},
            "yaxis": {"visible": False, "showbackground": False},
            "zaxis": {"visible": False, "showbackground": False},
            "aspectmode": "data",
        },
        uirevision="graph-connection",
    )
    return figure


def _selected_detail(event) -> tuple[str, str] | None:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None
    custom = points[0].get("customdata")
    if not isinstance(custom, list) or len(custom) < 2:
        return None
    return str(custom[0]), str(custom[1])


load_dotenv()

st.set_page_config(
    page_title="Show Graph Connection (Visual)",
    page_icon="🧠",
    layout="wide",
)

st.title("Show Graph Connection (Visual)")
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
used_labels: set[str] = set()
hub_ids = {entity_id for entity_id, _name, _memory_ids in view.hubs}


def _menu_label(name: str, count: int) -> str:
    label = f"{name} ({count})"
    if label not in used_labels:
        used_labels.add(label)
        return label
    index = 2
    while f"{label} {index}" in used_labels:
        index += 1
    renamed = f"{label} {index}"
    used_labels.add(renamed)
    return renamed


menu: list[tuple[str, set[str], tuple]] = []
for entity_id, name, memory_ids in view.groups:
    shown_hub = (
        ((entity_id, name, memory_ids),) if entity_id in hub_ids else ()
    )
    menu.append((_menu_label(name, len(memory_ids)), set(memory_ids), shown_hub))
menu_by_label = {label: (members, hubs) for label, members, hubs in menu}
choice = st.selectbox(
    "Group",
    ["All Groups", *[label for label, _members, _hubs in menu]],
)
shown_memories = view.memories
shown_links = view.links
shown_hubs = view.hubs
if choice != "All Groups":
    members, shown_hubs = menu_by_label[choice]
    shown_memories = tuple(
        item for item in view.memories if item[0] in members
    )
    shown_links = tuple(
        item
        for item in view.links
        if item[0] in members and item[1] in members
    )
positions, group_of, clustered = _layout(
    shown_memories, shown_links, shown_hubs
)
figure = _figure(
    shown_memories,
    shown_links,
    positions,
    group_of,
    clustered,
    shown_hubs,
)
event = st.plotly_chart(
    figure,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="graph_connection_pick",
)
chosen = _selected_detail(event)
if chosen is None:
    st.caption(
        "Click a memory point to read it. "
        "Click a hub point, or a diamond on a line, to see the shared entity."
    )
elif chosen[0] == "edge":
    st.subheader("Connection")
    st.write(chosen[1])
elif chosen[0] == "hub":
    st.subheader("Hub")
    st.write(chosen[1])
else:
    st.subheader("Memory")
    st.write(chosen[1])
st.caption(
    "Each point is an atomic memory. "
    "A line joins memories that share an entity. "
    "Separate colors are separate groups."
)
st.caption(
    "The Group menu lists every entity shared by two or more memories. "
    "A group of more than 20 memories is drawn as a hub. "
    "A smaller group is drawn with a line between each pair."
)
