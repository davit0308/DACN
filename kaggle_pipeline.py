import os
import shutil
import subprocess
import time
import re as _re

# ============================================================================
# KAGGLE PIPELINE — TAGAPT + PruningAgent + FastGA
# ============================================================================
# Pipeline:
#   CTIs_trans/multirow/new_100asg.txt  (nhieu graph noi nhau, tach bang #N)
#     -> split thanh file rieng le     -> graph_txt/1.txt, 2.txt, ...
#     -> Find_hub_process_test.py      -> 4000_3_generated_data_new2_sub/ (co stage annotation)
#     -> graph_instance.py + FastGA    -> 4000_3_generated_data_new2_sub_instance_windows/
#     -> generate_subgraph_CTI.py      -> result-visualization/
# ============================================================================

prev_run_dir = "/kaggle/input/datasets/avtnguyn/tagapt/DACN/TAGAPT_generated_samples"
pruning_dir  = "/kaggle/input/datasets/avtnguyn/tagapt/DACN/Pruning_Agent"
working_dir  = "/kaggle/working/TAGAPT_Run"

MAX_GRAPHS = 3  # gioi han so do thi de tranh tran RAM

# ======================================================================
# BUOC 1: KIEM TRA DU LIEU DAU VAO
# ======================================================================
print("=" * 60)
print("BUOC 1: KIEM TRA DU LIEU DAU VAO")
print("=" * 60)

if not os.path.exists(prev_run_dir):
    raise FileNotFoundError(f"Khong tim thay thu muc: {prev_run_dir}")
print(f"[OK] Thanh qua tai: {prev_run_dir}")

# ======================================================================
# BUOC 2: COPY SANG WORKING (co quyen ghi)
# ======================================================================
print("\n" + "=" * 60)
print("BUOC 2: COPY SANG MOI TRUONG LAM VIEC")
print("=" * 60)

if not os.path.exists(working_dir):
    print("Dang copy ma nguon va du lieu...")
    shutil.copytree(prev_run_dir, working_dir)
    print(f"[OK] Da copy xong -> {working_dir}")
else:
    print("Thu muc lam viec da ton tai, su dung lai.")

# Dam bao pruning_agent.py co trong working dir
pruning_dst = os.path.join(working_dir, "pruning_agent.py")
if not os.path.exists(pruning_dst):
    candidates = [
        os.path.join(pruning_dir, "pruning_agent.py"),
        os.path.join(prev_run_dir, "pruning_agent.py"),
        os.path.join(prev_run_dir, "..", "Pruning_Agent", "pruning_agent.py"),
    ]
    for src in candidates:
        if os.path.exists(src):
            shutil.copy(src, pruning_dst)
            print(f"[OK] Da copy pruning_agent.py tu {src}")
            break
    else:
        raise FileNotFoundError(
            "Khong tim thay pruning_agent.py!\n"
            "Hay upload file nay len Kaggle dataset."
        )
else:
    print(f"[OK] pruning_agent.py da co san")

# Dam bao edge_validator.py co trong working dir
validator_dst = os.path.join(working_dir, "edge_validator.py")
if not os.path.exists(validator_dst):
    validator_candidates = [
        os.path.join(pruning_dir, "edge_validator.py"),
        os.path.join(prev_run_dir, "edge_validator.py"),
        os.path.join(prev_run_dir, "..", "Pruning_Agent", "edge_validator.py"),
    ]
    for src in validator_candidates:
        if os.path.exists(src):
            shutil.copy(src, validator_dst)
            print(f"[OK] Da copy edge_validator.py tu {src}")
            break
    else:
        raise FileNotFoundError(
            "Khong tim thay edge_validator.py!\n"
            "Hay upload file nay len Kaggle dataset."
        )
else:
    print(f"[OK] edge_validator.py da co san")

os.chdir(working_dir)
print(f"[OK] Working directory: {os.getcwd()}")

# ======================================================================
# BUOC 3: CAI THU VIEN
# ======================================================================
print("\n" + "=" * 60)
print("BUOC 3: CAI DAT THU VIEN")
print("=" * 60)

subprocess.run(["pip", "install", "-q", "networkx"], check=True)
print("[OK] Thu vien da san sang")

# ======================================================================
# BUOC 4: CHUAN BI DO THI
# ======================================================================
print("\n" + "=" * 60)
print("BUOC 4: CHUAN BI DO THI")
print("=" * 60)

# -- Helper functions --
def is_graph_file(filepath):
    """File co dung format do thi TAGAPT? (dong 1 = so node, dong 2 = entity type)"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            line1 = f.readline().strip()
            if not line1.isdigit():
                return False
            line2 = f.readline().strip()
            return bool(_re.match(r'^[A-Z]{2}', line2))
    except:
        return False

def has_stage_annotation(filepath):
    """File co edge voi stage annotation? (vi du: 0 1 FR-1)"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if _re.match(r'\d+\s+\d+\s+[A-Z]+-\d+', line.strip()):
                    return True
    except:
        pass
    return False

def split_multirow(multirow_path, output_dir, max_files=3):
    """Split file multirow (cac graph noi nhau bang #N) thanh file rieng le."""
    with open(multirow_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    graphs = _re.split(r'^#\d+\s*\n', content, flags=_re.MULTILINE)
    graphs = [g.strip() for g in graphs if g.strip()]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    count = 0
    for i, graph_text in enumerate(graphs[:max_files]):
        out_path = os.path.join(output_dir, f"{i+1}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(graph_text + '\n')
        count += 1
    return count, len(graphs)

def print_graph_info(directory, label=""):
    """In thong tin cac file do thi trong thu muc."""
    if not os.path.exists(directory):
        print(f"  {label}(thu muc khong ton tai)")
        return
    files = sorted([f for f in os.listdir(directory) if f.endswith('.txt')])
    for f in files[:MAX_GRAPHS]:
        fpath = os.path.join(directory, f)
        with open(fpath, 'r') as fh:
            first_line = fh.readline().strip()
        sz = os.path.getsize(fpath)
        staged = " [co stage]" if has_stage_annotation(fpath) else ""
        print(f"  {f}: {first_line} nodes, {sz} bytes{staged}")

# -- 4a. In cay thu muc de debug --
print("\n[DEBUG] Cau truc thu muc lam viec:")
for root, dirs, files in os.walk(working_dir):
    depth = root.replace(working_dir, "").count(os.sep)
    if depth > 2:
        dirs.clear()
        continue
    indent = "  " * depth
    basename = os.path.basename(root) or "TAGAPT_Run"
    print(f"{indent}{basename}/  ({len(files)} files, {len(dirs)} dirs)")
    if depth <= 1:
        for f in sorted(files)[:5]:
            print(f"{indent}  {f}")
        if len(files) > 5:
            print(f"{indent}  ... va {len(files)-5} file khac")
print()

graph_txt_dir = os.path.join(working_dir, "graph_txt")
sub_graph_dir = os.path.join(working_dir, "4000_3_generated_data_new2_sub")

# -- 4b. Kiem tra: da co subgraph voi stage annotation chua? --
subgraph_ready = False
if os.path.exists(sub_graph_dir):
    staged_files = [f for f in os.listdir(sub_graph_dir)
                    if f.endswith('.txt')
                    and is_graph_file(os.path.join(sub_graph_dir, f))
                    and has_stage_annotation(os.path.join(sub_graph_dir, f))]
    if staged_files:
        print(f"[OK] Da co {len(staged_files)} subgraph files (co stage annotation)")
        # Gioi han
        for f in sorted(staged_files)[MAX_GRAPHS:]:
            os.remove(os.path.join(sub_graph_dir, f))
        print_graph_info(sub_graph_dir)
        subgraph_ready = True

# -- 4c. Neu chua co subgraph, kiem tra graph_txt (file rieng le) --
graph_txt_ready = False
if not subgraph_ready:
    if os.path.exists(graph_txt_dir):
        gfiles = [f for f in os.listdir(graph_txt_dir)
                  if f.endswith('.txt') and is_graph_file(os.path.join(graph_txt_dir, f))]
        if gfiles:
            print(f"[OK] Da co {len(gfiles)} graph files tai graph_txt/")
            print_graph_info(graph_txt_dir)
            graph_txt_ready = True

# -- 4d. Neu chua co graph_txt, tim multirow file de split --
if not subgraph_ready and not graph_txt_ready:
    print("[INFO] Chua co graph_txt/ hoac subgraph/, dang tim multirow file de split...")
    multirow_file = None
    search_dirs = [
        os.path.join(working_dir, "CTIs_trans", "multirow"),
        os.path.join(working_dir, "CTIs_trans"),
    ]
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.txt'):
                fpath = os.path.join(d, f)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
                    first_line = fh.readline().strip()
                if first_line.startswith('#'):
                    multirow_file = fpath
                    break
        if multirow_file:
            break

    if multirow_file is None:
        # Fallback: tim bat ky .txt nao co dung format graph
        print("[INFO] Khong tim thay multirow file, dang quet toan bo thu muc...")
        for root, dirs, files in os.walk(working_dir):
            for f in files:
                if f.endswith('.txt') and is_graph_file(os.path.join(root, f)):
                    # Tim thay file graph don le, copy vao graph_txt
                    if not os.path.exists(graph_txt_dir):
                        os.makedirs(graph_txt_dir)
                    shutil.copy(os.path.join(root, f), graph_txt_dir)
                    graph_txt_ready = True
                    if len(os.listdir(graph_txt_dir)) >= MAX_GRAPHS:
                        break
            if graph_txt_ready:
                break

    if multirow_file and not graph_txt_ready:
        print(f"[OK] Tim thay multirow file: {multirow_file}")
        count, total = split_multirow(multirow_file, graph_txt_dir, MAX_GRAPHS)
        print(f"[OK] Da split {count}/{total} do thi -> graph_txt/")
        print_graph_info(graph_txt_dir)
        graph_txt_ready = True

    if not graph_txt_ready and not subgraph_ready:
        print("\n[LOI] Khong tim thay du lieu do thi nao!")
        print("Can mot trong cac nguon sau:")
        print("  1. CTIs_trans/multirow/*.txt  (multirow bat dau bang #0)")
        print("  2. graph_txt/*.txt            (file rieng le)")
        print("  3. 4000_3_generated_data_new2_sub/*.txt (co stage annotation)")
        print("\nHay chay buoc sinh do thi truoc (MaskGAF.py / trans_gendata_CTI.py)")
        raise FileNotFoundError("Khong tim thay du lieu do thi!")

# -- 4e. Chay Find_hub_process_test.py de gan stage annotation --
if graph_txt_ready and not subgraph_ready:
    print("\n[INFO] Chay Find_hub_process_test.py de gan stage annotation...")

    # Find_hub doc tu ./graph_txt, ghi ra ./graph_txt_sub
    graph_txt_sub = os.path.join(working_dir, "graph_txt_sub")
    if os.path.exists(graph_txt_sub):
        shutil.rmtree(graph_txt_sub)
    os.makedirs(graph_txt_sub)

    hub_result = subprocess.run(
        ["python", "Find_hub_process_test.py"],
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    print(f"  Find_hub return code: {hub_result.returncode}")
    if hub_result.stdout:
        # In 20 dong cuoi cua stdout
        lines = hub_result.stdout.strip().split('\n')
        for line in lines[-20:]:
            print(f"  > {line}")

    if hub_result.returncode != 0:
        print(f"[WARN] Find_hub_process_test.py failed!")
        if hub_result.stderr:
            print(f"  stderr: {hub_result.stderr[-500:]}")
        print("[FALLBACK] Copy graph_txt/ truc tiep (khong co stage annotation)")
        if os.path.exists(sub_graph_dir):
            shutil.rmtree(sub_graph_dir)
        shutil.copytree(graph_txt_dir, sub_graph_dir)
    else:
        print("[OK] Find_hub_process_test.py hoan thanh!")
        # Copy graph_txt_sub -> 4000_3_generated_data_new2_sub
        if os.path.exists(sub_graph_dir):
            shutil.rmtree(sub_graph_dir)

        if os.path.exists(graph_txt_sub) and os.listdir(graph_txt_sub):
            shutil.copytree(graph_txt_sub, sub_graph_dir)
        else:
            print("[WARN] graph_txt_sub/ rong, dung graph_txt/ truc tiep")
            shutil.copytree(graph_txt_dir, sub_graph_dir)

    # Gioi han lai MAX_GRAPHS
    if os.path.exists(sub_graph_dir):
        all_files = sorted([f for f in os.listdir(sub_graph_dir) if f.endswith('.txt')])
        for f in all_files[MAX_GRAPHS:]:
            os.remove(os.path.join(sub_graph_dir, f))
        print(f"\n[OK] Subgraph dir san sang: {min(len(all_files), MAX_GRAPHS)} files")
        print_graph_info(sub_graph_dir)

# ======================================================================
# BUOC 5: KIEM TRA TICH HOP PRUNING + EDGE VALIDATOR
# ======================================================================
print("\n" + "=" * 60)
print("BUOC 5: KIEM TRA TICH HOP PRUNING + EDGE VALIDATOR")
print("=" * 60)

gi_path = os.path.join(working_dir, "graph_instance.py")
with open(gi_path, 'r') as f:
    gi_content = f.read()

need_patch = False
if "from pruning_agent import" not in gi_content or "FastGA" not in gi_content:
    print("[WARN] graph_instance.py CHUA co import PruningAgent!")
    need_patch = True
if "from edge_validator import" not in gi_content or "EdgeConstraintValidator" not in gi_content:
    print("[WARN] graph_instance.py CHUA co import EdgeConstraintValidator!")
    need_patch = True

if need_patch:
    print("       Dang tu dong patch...")
    lines = gi_content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('import json'):
            # Insert both imports after 'import json'
            insert_lines = []
            if 'from pruning_agent import' not in gi_content:
                insert_lines.append('from pruning_agent import PruningAgent, FastGA')
            if 'from edge_validator import' not in gi_content:
                insert_lines.append('from edge_validator import EdgeConstraintValidator')
            for j, new_line in enumerate(insert_lines):
                lines.insert(i + 1 + j, new_line)
            break
    gi_content = '\n'.join(lines)
    gi_content = gi_content.replace('run = GA(parameter)\n', 'run = FastGA(parameter)\n')
    gi_content = gi_content.replace('run2 = GA(parameter2)\n', 'run2 = FastGA(parameter2)\n')

    # Fix Windows paths for Linux (Kaggle)
    gi_content = gi_content.replace(r'r".\regulation_dic"', 'os.path.join(".", "regulation_dic")')
    gi_content = gi_content.replace(r'r".\tech_dic"', 'os.path.join(".", "tech_dic")')
    gi_content = gi_content.replace(r'r".\4000_3_generated_data_new2_sub"', 'os.path.join(".", "4000_3_generated_data_new2_sub")')
    gi_content = gi_content.replace(r'r".\instance_lib\technique-instance-lib-os-filter-add.json"', 'os.path.join(".", "instance_lib", "technique-instance-lib-os-filter-add.json")')
    gi_content = gi_content.replace(r'r".\4000_3_generated_data_new2_sub_instance_windows"', 'os.path.join(".", "4000_3_generated_data_new2_sub_instance_windows")')
    # Fix path concatenation with backslash
    gi_content = gi_content.replace('self.regu_path + "\\\\" + regulation', 'os.path.join(self.regu_path, regulation)')
    gi_content = gi_content.replace('self.tech_path + "\\\\" + tech', 'os.path.join(self.tech_path, tech)')
    gi_content = gi_content.replace('DL.sub_graph_path + "\\\\" + txt', 'os.path.join(DL.sub_graph_path, txt)')
    gi_content = gi_content.replace('new_instance_path+"\\\\"+ txt', 'os.path.join(new_instance_path, txt)')
    gi_content = gi_content.replace('new_instance_path + "\\\\" + txt', 'os.path.join(new_instance_path, txt)')
    gi_content = gi_content.replace('new_instance_path+"\\\\\"+txt', 'os.path.join(new_instance_path, txt)')

    # Fix get_relation_list regex: old regex REQUIRES stage annotation (-1, -2, etc.)
    # which fails when files have "0 1 FR" without stage.
    # Replace with robust version that makes stage annotation optional.
    old_regex_block = r"""            pattern = r'(\d+)\s+(\d+)\s+([A-Z]+)+(-\d+)'"""
    if old_regex_block in gi_content:
        print("       [PATCH] Fixing get_relation_list regex...")
        # Replace the entire get_relation_list method body
        old_method = '''    def get_relation_list(self,data):
        relation_list = [[],[],[],[]]
        i1 = 0
        for line in data:
            relation = []
            pattern = r'(\d+)\s+(\d+)\s+([A-Z]+)+(-\d+)'
            matches = re.findall(pattern, line)
            for match in matches:
                num1, num2, text, stage = match
                stage_true = int(stage[-1])-1
                relation.append(num1)
                relation.append(num2)
                relation.append(text)
                relation.append(i1)
                i1+=1
                relation_list[stage_true].append(relation)
        return relation_list'''
        new_method = '''    def get_relation_list(self,data):
        relation_list = [[],[],[],[]]
        i1 = 0
        for line in data:
            line = line.strip()
            m = re.match(r'^(\\d+)\\s+(\\d+)\\s+([A-Z]{2})(?:(-[\\d-]+))?$', line)
            if not m:
                continue
            num1, num2, verb, stage_str = m.groups()
            relation = [num1, num2, verb, i1]
            i1 += 1
            if stage_str:
                stages = [int(s) for s in stage_str.split('-') if s.strip()]
                for stage_num in stages:
                    if 1 <= stage_num <= 4:
                        relation_list[stage_num - 1].append(list(relation))
            else:
                for s in range(4):
                    relation_list[s].append(list(relation))
        return relation_list'''
        gi_content = gi_content.replace(old_method, new_method)

    with open(gi_path, 'w') as f:
        f.write(gi_content)
    print("[OK] Da patch xong graph_instance.py (imports + GA + Linux paths + regex)")
else:
    print("[OK] graph_instance.py da tich hop PruningAgent + FastGA + EdgeConstraintValidator")

# Verify pruning_agent.py
with open(pruning_dst, 'r') as f:
    pa_content = f.read()
for c in ["class PruningAgent", "class FastGA", "def prune(", "def GA_main("]:
    status = "[OK]" if c in pa_content else "[FAIL]"
    print(f"  {status} pruning_agent: {c}")

# Verify edge_validator.py
with open(validator_dst, 'r') as f:
    ev_content = f.read()
for c in ["class EdgeConstraintValidator", "def validate_edge(", "def validate_instance(",
          "def filter_relation_list(", "def get_fallback_instance("]:
    status = "[OK]" if c in ev_content else "[FAIL]"
    print(f"  {status} edge_validator: {c}")

# ======================================================================
# BUOC 6: CHAY PIPELINE
# ======================================================================
print("\n" + "=" * 60)
print("BUOC 6: CHAY PIPELINE GRAPH INSTANCE (PruningAgent + FastGA)")
print("=" * 60)

t0 = time.time()
result = subprocess.run(
    ["python", "graph_instance.py"],
    cwd=working_dir,
    capture_output=True,
    text=True,
    timeout=600,
)
elapsed = time.time() - t0

print(f"\n--- stdout (last 3000 chars) ---")
stdout_tail = result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout
print(stdout_tail)

if result.stderr:
    print(f"\n--- stderr (last 2000 chars) ---")
    print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

print(f"\nReturn code: {result.returncode}")
print(f"Thoi gian: {elapsed:.1f}s")

if result.returncode != 0:
    print("\n[FAIL] graph_instance.py that bai! Kiem tra stderr o tren.")
else:
    print("\n[OK] graph_instance.py hoan thanh!")
    out_dir = os.path.join(working_dir, "4000_3_generated_data_new2_sub_instance_windows")
    if os.path.exists(out_dir):
        out_files = os.listdir(out_dir)
        print(f"[OK] Output: {len(out_files)} file tai {out_dir}")
        for f in out_files[:5]:
            print(f"  {f}: {os.path.getsize(os.path.join(out_dir, f))} bytes")
    else:
        print("[WARN] Thu muc output chua duoc tao")

# ======================================================================
# BUOC 7: VISUALIZATION (tuy chon)
# ======================================================================
print("\n" + "=" * 60)
print("BUOC 7: VISUALIZATION (tuy chon)")
print("=" * 60)

result_dir = os.path.join(working_dir, "4000_3_generated_data_new2_sub_instance_windows")
vis_dir = os.path.join(working_dir, "result-visualization")

if os.path.exists(result_dir) and os.listdir(result_dir):
    # Filter out empty/tiny files (< 10 bytes = only whitespace/newlines)
    valid_files = [f for f in os.listdir(result_dir)
                   if os.path.getsize(os.path.join(result_dir, f)) > 10]
    if not valid_files:
        print("[SKIP] Tat ca output files deu rong (< 10 bytes), khong co gi de visualize")
    else:
        print(f"  {len(valid_files)} valid IAG files (> 10 bytes) de visualize")
        try:
            r2 = subprocess.run(
                ["python", "generate_subgraph_CTI.py",
                 "--graph_path_txt", result_dir,
                 "--graph_txt_path_2", vis_dir],
                cwd=working_dir,
                capture_output=True, text=True, timeout=120,
            )
            print(r2.stdout[-2000:] if len(r2.stdout) > 2000 else r2.stdout)
            if r2.returncode == 0 and os.path.exists(vis_dir):
                print(f"[OK] Visualization: {len(os.listdir(vis_dir))} files tai {vis_dir}")
            else:
                print(f"[WARN] Visualization failed: {r2.stderr[:500]}")
        except Exception as e:
            print(f"[WARN] Visualization skipped: {e}")
else:
    print("[SKIP] Khong co output de visualize")

print("\n" + "=" * 60)
print("HOAN TAT!")
print("=" * 60)
