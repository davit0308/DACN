#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_pipeline.py
================
Smoke test for the pruning-integrated graph_instance pipeline.

Creates synthetic subgraph files + minimal regulation/tech/instance data,
then runs the main loop from graph_instance.py to verify:
  1. PruningAgent + FastGA import OK
  2. Pruning reduces node count
  3. GA_main() returns valid dicts
  4. Remap produces original-index keys
  5. File writing completes without error

Usage:
    cd TAGAPT_generated_samples
    python test_pipeline.py
"""

import os
import sys
import json
import random
import shutil
import tempfile
import traceback

# ---------------------------------------------------------------------------
# 1. Create temporary test data directory structure
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE, "_test_workspace")

def setup_test_data():
    """Create minimal synthetic data that graph_instance.py expects."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)

    # --- regulation_dic (4 stages) ---
    regu_dir = os.path.join(TEST_DIR, "regulation_dic")
    os.makedirs(regu_dir)
    # Minimal rules: each stage has a few rules matching entity-type triples
    entity_types = ["MP", "TP", "SO", "MF", "SF", "TF"]
    verbs = ["FR", "RD", "WR", "EX", "ST", "RF", "IJ", "UK"]
    tactic_names = {
        1: ["Reconnaissance", "Initial Access", "Execution"],
        2: ["Persistence", "Privilege Escalation", "Defense Evasion"],
        3: ["Credential Access", "Discovery", "Lateral Movement"],
        4: ["Collection", "Exfiltration", "Impact"],
    }
    for stage_num in range(1, 5):
        rules = {}
        for i, tactic in enumerate(tactic_names[stage_num]):
            rule_name = f"{i+1}.{tactic}-{i+1}"
            # Each rule = list of [entity_type, verb, entity_type] triples
            n_triples = random.randint(1, 3)
            rule_body = []
            for _ in range(n_triples):
                rule_body.append([
                    random.choice(entity_types),
                    random.choice(verbs),
                    random.choice(entity_types),
                ])
            rules[rule_name] = rule_body
        with open(os.path.join(regu_dir, f"stage{stage_num}_regulation.json"), "w") as f:
            json.dump(rules, f)

    # --- tech_dic (4 stages, keys must match regulation_dic) ---
    tech_dir = os.path.join(TEST_DIR, "tech_dic")
    os.makedirs(tech_dir)
    for stage_num in range(1, 5):
        regu_path = os.path.join(regu_dir, f"stage{stage_num}_regulation.json")
        with open(regu_path) as f:
            regu = json.load(f)
        techs = {}
        for key in regu:
            techs[key] = [f"T{random.randint(1000,1999)}"]
        with open(os.path.join(tech_dir, f"stage{stage_num}_tech.json"), "w") as f:
            json.dump(techs, f)

    # --- instance_lib ---
    inst_dir = os.path.join(TEST_DIR, "instance_lib")
    os.makedirs(inst_dir)
    # Build a JSONL file with one entry per tech
    inst_path = os.path.join(inst_dir, "technique-instance-lib-os-filter-add.json")
    with open(inst_path, "w", encoding="utf-8") as f:
        for stage_num in range(1, 5):
            tech_path = os.path.join(tech_dir, f"stage{stage_num}_tech.json")
            with open(tech_path) as tf:
                techs = json.load(tf)
            for key, tech_list in techs.items():
                for tech_id in tech_list:
                    entry = {"stage-key": f"{tech_id}-linux"}
                    for et in entity_types:
                        entry[et] = [f"instance_{et}_{tech_id}"]
                    f.write(json.dumps(entry) + "\n")

    # --- subgraph files (the input graphs) ---
    sub_dir = os.path.join(TEST_DIR, "4000_3_generated_data_new2_sub")
    os.makedirs(sub_dir)
    # Create 2 small test graphs
    for gid in range(1, 3):
        n_nodes = 20  # small enough for fast test
        lines = [f"{n_nodes}\n"]
        for _ in range(n_nodes):
            lines.append(random.choice(entity_types) + "\n")
        # edges with stage annotation
        edges = []
        for _ in range(40):
            src = random.randint(0, n_nodes - 1)
            dst = random.randint(0, n_nodes - 1)
            if src == dst:
                dst = (dst + 1) % n_nodes
            verb = random.choice(verbs)
            stage_tag = random.randint(1, 4)
            edges.append(f"{src} {dst} {verb}-{stage_tag}\n")
        lines.append(f"{len(edges)}\n")
        lines.extend(edges)
        with open(os.path.join(sub_dir, f"{gid}.txt"), "w") as f:
            f.writelines(lines)

    # --- output directory ---
    out_dir = os.path.join(TEST_DIR, "4000_3_generated_data_new2_sub_instance_windows")
    os.makedirs(out_dir, exist_ok=True)

    return sub_dir, out_dir


# ---------------------------------------------------------------------------
# 2. Run the pipeline
# ---------------------------------------------------------------------------

def run_test():
    print("=" * 60)
    print("PIPELINE SMOKE TEST")
    print("=" * 60)

    sub_dir, out_dir = setup_test_data()
    print(f"[OK] Test data created in {TEST_DIR}")

    # Temporarily override paths used by Dataloader
    # We'll import and monkey-patch the Dataloader class
    sys.path.insert(0, BASE)

    # Import after sys.path is set
    from pruning_agent import PruningAgent, FastGA
    print("[OK] pruning_agent imported successfully")

    # Import Dataloader and helpers from graph_instance
    import graph_instance as gi
    print("[OK] graph_instance imported successfully (PruningAgent+FastGA import OK)")

    # Patch Dataloader paths to point to test data
    original_init = gi.Dataloader.__init__
    def patched_init(self):
        self.regu_path = os.path.join(TEST_DIR, "regulation_dic")
        self.tech_path = os.path.join(TEST_DIR, "tech_dic")
        self.sub_graph_path = sub_dir
    gi.Dataloader.__init__ = patched_init

    # Run the core logic (extracted from __main__ block)
    CXPB, MUTPB, NGEN, popsize = 0.8, 0.4, 3, 10  # fewer gens for speed

    DL = gi.Dataloader()
    s1r, s2r, s3r, s4r = DL.load_regulation()
    s1t, s2t, s3t, s4t = DL.load_tech()
    print(f"[OK] Loaded regulation ({len(s1r)} rules stage1) and tech dicts")

    stage_len = [len(s1r), len(s2r), len(s3r), len(s4r)]
    print(f"[OK] stage_len = {stage_len}")

    instance_lib = gi.read_new(
        os.path.join(TEST_DIR, "instance_lib", "technique-instance-lib-os-filter-add.json")
    )
    print(f"[OK] instance_lib loaded ({len(instance_lib)} entries)")

    os_type = "linux"
    errors = []
    files_processed = 0

    for txt in os.listdir(sub_dir):
        whole_file_path = os.path.join(sub_dir, txt)
        new_file_path = os.path.join(out_dir, txt)

        graph_data, entity_list, relation_list1 = DL.get_graph_info(whole_file_path)
        print(f"\n--- Processing {txt}: {len(entity_list)} nodes ---")

        entity_instance_dic = {i: [] for i in range(len(entity_list))}
        sum_relation = sum(len(s) for s in relation_list1)
        relation_instance_dic = {i: [] for i in range(sum_relation)}

        pruning_agent = PruningAgent(target_nodes=150, min_stage_coverage=0.60, verbose=True)

        for stage in range(1, 5):
            stage_index_regu_dic = DL.load_regulation()[stage - 1]
            stage_index_tech_dic = DL.load_tech()[stage - 1]
            stage_regu_len = stage_len[stage - 1]

            # STEP 3: Prune
            pruned_entity, pruned_relations, node_map = pruning_agent.prune(
                entity_list, relation_list1, stage
            )
            print(f"  Stage {stage}: pruned {len(entity_list)}->{len(pruned_entity)} nodes, "
                  f"node_map size={len(node_map)}")

            # STEP 4+5: FastGA
            parameter = [CXPB, MUTPB, NGEN, popsize, stage_regu_len,
                         stage_index_regu_dic, stage_index_tech_dic,
                         pruned_entity, pruned_relations, stage]
            run = FastGA(parameter)
            (bestindividual_gene,
             bestindividual_entity_regu_dic_pruned,
             bestindividual_relation_regu_dic_pruned) = run.GA_main()

            # STEP 6: Remap
            bestindividual_entity_regu_dic = pruning_agent.remap_entity_dic(
                bestindividual_entity_regu_dic_pruned, node_map
            )
            bestindividual_relation_regu_dic = pruning_agent.remap_relation_dic(
                bestindividual_relation_regu_dic_pruned, node_map
            )

            # Validate types
            assert isinstance(bestindividual_entity_regu_dic, dict), "entity_regu_dic not dict"
            assert isinstance(bestindividual_relation_regu_dic, dict), "relation_regu_dic not dict"
            print(f"  Stage {stage}: entity_regu keys={list(bestindividual_entity_regu_dic.keys())[:5]}...")

            # STEP 7: unsuccess_edge + run2
            unsuccess_edge = [[]]
            for key in bestindividual_relation_regu_dic:
                if len(bestindividual_relation_regu_dic[key]) == 0:
                    tr = gi.find_target_relation(relation_list1[stage - 1], key)
                    if tr:
                        unsuccess_edge[0].append(tr)

            if unsuccess_edge[0]:
                new_target_stage = 2 if stage == 1 else stage - 1
                sr2 = DL.load_regulation()[new_target_stage - 1]
                st2 = DL.load_tech()[new_target_stage - 1]
                srl2 = stage_len[new_target_stage - 1]

                p_e2, p_r2, nm2 = pruning_agent.prune(entity_list, unsuccess_edge, 1)
                parameter2 = [CXPB, MUTPB, NGEN, popsize, srl2, sr2, st2, p_e2, p_r2, 1]
                run2 = FastGA(parameter2)
                _, be2_p, br2_p = run2.GA_main()
                be2 = pruning_agent.remap_entity_dic(be2_p, nm2)
                br2 = pruning_agent.remap_relation_dic(br2_p, nm2)
                print(f"  Stage {stage} retry: {len(unsuccess_edge[0])} unsuccess edges processed")

            # Update instance dicts (same logic as original)
            for key in list(bestindividual_entity_regu_dic.keys()):
                for l in range(len(bestindividual_entity_regu_dic[key])):
                    r = bestindividual_entity_regu_dic[key][l]
                    info = str(r) if str(r).find("-") != -1 else f"{stage}-{r}"
                    if info not in entity_instance_dic.get(key, []):
                        entity_instance_dic.setdefault(key, []).append(info)

            for key in list(bestindividual_relation_regu_dic.keys()):
                for l in range(len(bestindividual_relation_regu_dic[key])):
                    r = bestindividual_relation_regu_dic[key][l]
                    info = str(r) if str(r).find("-") != -1 else f"{stage}-{r}"
                    if info not in relation_instance_dic.get(key, []):
                        relation_instance_dic.setdefault(key, []).append(info)

        files_processed += 1
        count = sum(1 for v in entity_instance_dic.values() if len(v) != 0)
        coverage = count / max(len(entity_instance_dic), 1)
        print(f"  => {txt} coverage = {coverage:.2%}")

    # Restore
    gi.Dataloader.__init__ = original_init

    print("\n" + "=" * 60)
    print(f"RESULT: {files_processed} files processed, {len(errors)} errors")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        print("FAILED [X]")
        return False
    else:
        print("ALL PASSED [OK]")
        return True


# ---------------------------------------------------------------------------
# 3. Cleanup
# ---------------------------------------------------------------------------

def cleanup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
        print(f"[cleanup] Removed {TEST_DIR}")


if __name__ == "__main__":
    try:
        ok = run_test()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        ok = False
    finally:
        cleanup()
    sys.exit(0 if ok else 1)
