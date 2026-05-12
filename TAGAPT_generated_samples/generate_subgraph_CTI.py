# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : generate_subgraph_CTI.py
# Description: make dot and pdf file for all generated IAG
# Fixed for cross-platform (Linux/Kaggle) compatibility
"""
import argparse
import re
import os
import shutil
import sys

# Try importing graphviz; if not available, skip PDF generation
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("[WARN] graphviz not installed. DOT files will be created but PDF rendering skipped.")
    print("       Install with: pip install graphviz")


def graph_construct(fname, result_dir):
    """Construct DOT/PDF visualization from IAG text files.

    Parameters
    ----------
    fname : str
        Path to directory containing .txt graph files
    result_dir : str
        Path to output directory for DOT/PDF files
    """
    # Ensure output dir exists
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    txt_files = [f for f in os.listdir(fname) if f.endswith('.txt')]
    if not txt_files:
        print(f"[WARN] No .txt files found in {fname}")
        return

    processed = 0
    skipped = 0

    for path in sorted(txt_files):
        graph_txt = os.path.join(fname, path)
        graph_name_dot = path[:-4] + ".dot"
        graph_name_dot_path = os.path.join(result_dir, graph_name_dot)

        try:
            with open(graph_txt, "r", encoding="utf-8", errors="ignore") as f:
                graph_lines = f.readlines()
        except Exception as e:
            print(f"[WARN] Cannot read {graph_txt}: {e}")
            skipped += 1
            continue

        # Parse entity list
        entity_list = []
        entity_count = 0
        for line in graph_lines:
            line_stripped = line.strip()
            if re.match(r"^[A-Z]{2}", line_stripped):
                # Escape special characters for DOT format
                safe_line = line_stripped.replace("\\", "\\\\\\\\")
                safe_line = safe_line.replace(":", "\\\\")
                safe_line = safe_line.replace("/", "\\\\")
                entity_list.append(str(entity_count) + "_" + safe_line)
                entity_count += 1

        # Parse relation list
        relation_list = []
        for line in graph_lines:
            line_stripped = line.strip()
            if re.match(r"^\d+\s\d+\s[A-Z]+", line_stripped):
                relation_list.append(line_stripped)

        # Skip empty/malformed graphs
        if entity_count == 0:
            print(f"  [SKIP] {path}: no entities found")
            skipped += 1
            continue
        if not relation_list:
            print(f"  [SKIP] {path}: no edges found")
            skipped += 1
            continue

        print(f"  {path}: {entity_count} entities, {len(relation_list)} edges")

        if not HAS_GRAPHVIZ:
            # Write a basic DOT file manually
            with open(graph_name_dot_path, 'w') as f:
                f.write(f'digraph "{graph_name_dot_path}" {{\n')
                f.write('rankdir="LR";\n')
                for entity in entity_list:
                    f.write(f'  "{entity}";\n')
                for rel in relation_list:
                    parts = rel.split(" ", 2)
                    if len(parts) >= 3:
                        sub, obj = int(parts[0]), int(parts[1])
                        verb = parts[2].split("-")[0]
                        if sub < len(entity_list) and obj < len(entity_list):
                            f.write(f'  "{entity_list[sub]}" -> "{entity_list[obj]}" [label="{verb}"];\n')
                f.write('}\n')
            processed += 1
            continue

        # Build graphviz DOT
        g = Digraph(graph_name_dot_path, filename=graph_name_dot)
        g.body.extend([
            'rankdir="LR"', 'size="9"', 'fixedsize="false"',
            'splines="true"', 'nodesep=0.3', 'ranksep=0',
            'fontsize=10', 'overlap="scalexy"', 'engine= "neato"'
        ])

        # Draw entities
        for entity in entity_list:
            parts = entity.split("_", 1)
            if len(parts) < 2:
                continue
            etype = parts[1].split("-")[0].replace("*", "")

            if parts[1].startswith(("MP*", "TP*")):
                g.node(entity, shape='rectangle', style='filled', fillcolor="lightblue:red")
            elif etype in ("MP", "TP"):
                g.node(entity, shape='rectangle')
            elif etype == "SO":
                g.node(entity, shape='diamond')
            elif etype in ("MF", "SF", "TF"):
                g.node(entity, shape='ellipse')
            elif etype == "R":
                g.node(entity, shape='house')

        # Draw edges
        edge_count = 1
        for relation in relation_list:
            parts = relation.split(" ", 2)
            if len(parts) < 3:
                continue
            try:
                sub = int(parts[0])
                obj = int(parts[1])
            except ValueError:
                continue

            if sub >= len(entity_list) or obj >= len(entity_list):
                print(f"    [WARN] Edge {sub}->{obj} out of bounds (max={len(entity_list)-1}), skipping")
                continue

            verb_stage = parts[2].strip()
            verb_parts = verb_stage.split("-")
            verb = verb_parts[0]
            stage = verb_parts[1:]

            # Draw labeled edge
            if verb in ("RD", "WR", "EX", "UK", "CD", "FR", "IJ", "ST", "RF"):
                g.edge(entity_list[sub], entity_list[obj],
                       label=str(edge_count) + ': ' + verb)
                edge_count += 1

            # Draw stage-colored edges
            stage_colors = {"1": "red", "2": "blue", "3": "yellow", "4": "green", "5": "pink"}
            for s, color in stage_colors.items():
                if s in stage:
                    g.edge(entity_list[sub], entity_list[obj],
                           label=str(s), color=color)

        # Save DOT file
        g.save(graph_name_dot_path)

        # Try to render PDF (may fail if graphviz binary not installed)
        try:
            g.render(graph_name_dot_path, view=False)
        except Exception as e:
            print(f"    [WARN] PDF render failed for {path}: {e}")

        processed += 1

    print(f"\n  Visualization complete: {processed} processed, {skipped} skipped")


def make_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"'{directory_path}' created")
    else:
        print(f"'{directory_path}' exists")
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize instantiation graph')
    parser.add_argument('--graph_path_txt', type=str,
                        default='./generated_100_Asg/generated-graph-instance-result',
                        help='the instantiation graph', required=True)
    parser.add_argument('--graph_txt_path_2', type=str,
                        default='./generated_100_Asg/generated-graph-instance-result-visualization',
                        help='the visualization result', required=True)
    args = parser.parse_args()

    make_directory(args.graph_txt_path_2)
    graph_construct(args.graph_path_txt, args.graph_txt_path_2)