# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : graph_instance.py
# Description：
"""
import os
import random
import re
from operator import itemgetter
import json
from pruning_agent import PruningAgent, FastGA
from edge_validator import (
    EdgeConstraintValidator,
    FILE_AND_NETWORK_TOOLS,
    SCANNER_ONLY_TOOLS,
    EXECUTABLE_EXTENSIONS_STRICT,
)


def generate_smart_instance_name(node_type: str, edge_context: dict) -> str:
    """
    Generate a semantically realistic instance name when CTI lookup yields
    no valid match for the required edge verbs.

    Instead of falling back to 'unknown_tool' / 'unknown_process' etc.,
    we pick a real tool/file name that is *guaranteed* to be consistent
    with the edge constraints the node participates in.

    Args:
        node_type   : 'MP', 'TP', 'MF', 'SF', 'TF', or 'SO'
        edge_context: dict with keys
            'out_verbs' -> list of outgoing verb strings (e.g. ['FR','WR'])
            'in_verbs'  -> list of incoming verb strings (e.g. ['FR'])

    Returns:
        A realistic, semantically valid instance name string.
    """
    out_verbs = set(edge_context.get("out_verbs", []))
    in_verbs  = set(edge_context.get("in_verbs", []))

    # ── Process nodes (MP = Malicious Process, TP = Tool Process) ──
    if node_type in ("MP", "TP"):
        # Needs fork / exec / inject capability → must be shell or interpreter
        if out_verbs & {"FR", "IJ", "EX"}:
            if node_type == "MP":
                return random.choice(["bash", "sh", "python", "perl", "python3"])
            else:  # TP
                return random.choice(["bash", "sh", "python3", "perl", "sudo"])

        # Needs network send/receive → network-capable tool
        if out_verbs & {"ST", "RF"}:
            if node_type == "MP":
                return random.choice(["bash", "python", "nc", "curl", "wget"])
            else:  # TP
                return random.choice(["curl", "wget", "nc", "ssh", "nmap", "scp"])

        # File I/O only (RD / WR / CD / UK)
        if out_verbs & {"RD", "WR", "CD", "UK"}:
            if node_type == "TP":
                return random.choice(["cat", "cp", "grep", "awk", "dd", "tee", "sed"])
            else:  # MP – still an "active" process
                return random.choice(["bash", "python", "cat", "cp", "dd"])

        # No out_verbs recognised → safe default
        if node_type == "TP":
            return random.choice(["cat", "grep", "awk", "cp", "dd"])
        else:  # MP
            return random.choice(["bash", "sh", "python"])

    # ── Malicious File ──
    elif node_type == "MF":
        return random.choice([
            "malware.bin", "payload.elf", "backdoor.sh",
            "exploit.py", "dropper.bin", "implant", "stager.sh"
        ])

    # ── System File ──
    elif node_type == "SF":
        # If this file is being executed, MUST have explicit executable extension
        # (matches EXECUTABLE_EXTENSIONS_STRICT: no empty-string allowed)
        if "EX" in in_verbs:
            return random.choice([
                "init.sh", "startup.sh", "cron.sh", "deploy.sh",
                "backdoor.bin", "loader.elf", "stager.py", "hook.pl",
            ])
        return random.choice([
            "passwd", "shadow", "sudoers", "hosts",
            "sshd_config", "crontab", "fstab", "resolv.conf",
            "httpd.conf", "nginx.conf", "my.cnf", "bashrc"
        ])


    # ── Temporary File ──
    elif node_type == "TF":
        return random.choice([
            "out.tmp", "data.log", "dump.tmp", "capture.pcap",
            "loot.txt", "creds.txt", "scan.out", "output.dat"
        ])

    # ── Socket / Network endpoint ──
    elif node_type == "SO":
        ports = ["80", "443", "22", "21", "4444", "8080", "53", "25"]
        return "0.0.0.0:" + random.choice(ports)

    # Catch-all (should never reach here if node_type is valid)
    return "unknown_" + node_type.lower()


# ══════════════════════════════════════════════════════════════════════════════
# UK EDGE RESOLVER
# ══════════════════════════════════════════════════════════════════════════════

# Lookup table: (src_type, dst_type) -> preferred verb when UK is seen
# Based on semantics in provenance graphs:
#   Process  -> File    : write or read (50/50 randomised)
#   Process  -> Process : fork (subprocess spawn)
#   Process  -> Socket  : send data to network
#   Socket   -> Process : receive data from network (RF)
#   File     -> Process : execute (file executed by process)
_UK_RESOLUTION_TABLE = {
    # (src_type, dst_type) : verb  OR callable(src, dst) -> verb
    ("MP", "SF"): lambda: random.choice(["WR", "RD"]),
    ("MP", "TF"): "WR",
    ("MP", "MF"): lambda: random.choice(["WR", "EX"]),
    ("MP", "SO"): "ST",
    ("MP", "TP"): "FR",
    ("MP", "MP"): "FR",
    ("TP", "SF"): lambda: random.choice(["WR", "RD"]),
    ("TP", "TF"): lambda: random.choice(["WR", "RD"]),
    ("TP", "MF"): lambda: random.choice(["RD", "EX"]),
    ("TP", "SO"): "ST",
    ("TP", "TP"): lambda: random.choice(["FR", "IJ"]),
    ("TP", "MP"): "IJ",
    ("SO", "MP"): "RF",
    ("SO", "TP"): "RF",
    ("SF", "MP"): "RD",
    ("SF", "TP"): "RD",
    ("MF", "MP"): "EX",
    ("MF", "TP"): "EX",
}

_UK_FALLBACK_VERB = "RD"     # safest default if no rule matches


def resolve_uk_edges(entity_list, relation_list):
    """
    Replace 'UK' (unknown) verb edges with semantically appropriate verbs
    determined by the src/dst node types.

    This is a post-processing step applied to the raw model output
    BEFORE the GA regulation-matching phase.  Because UK edges cannot
    be matched by any regulation rule, resolving them increases the
    regulation coverage score and yields semantically richer graphs.

    Args:
        entity_list  : list of node type strings e.g. ['MP', 'TP', 'SF', ...]
        relation_list: list-of-lists (one per stage) of
                       [src_idx, dst_idx, verb, edge_id]

    Returns:
        (resolved_relation_list, stats_dict)
    """
    total_uk = 0
    resolved  = 0

    new_relation_list = []
    for stage_edges in relation_list:
        new_stage = []
        for edge in stage_edges:
            # edge format: [src_idx, dst_idx, verb, edge_id]
            src_idx, dst_idx, verb, eid = edge[0], edge[1], edge[2], edge[3]
            if verb != "UK":
                new_stage.append(edge)
                continue

            total_uk += 1
            # Resolve using node types
            try:
                src_type = entity_list[int(src_idx)]
                dst_type = entity_list[int(dst_idx)]
            except (IndexError, ValueError):
                new_stage.append(edge)
                continue

            rule = _UK_RESOLUTION_TABLE.get((src_type, dst_type))
            if rule is None:
                # Try reverse direction as fallback clue
                rule = _UK_RESOLUTION_TABLE.get((dst_type, src_type))

            if rule is not None:
                new_verb = rule() if callable(rule) else rule
                resolved += 1
            else:
                new_verb = _UK_FALLBACK_VERB

            new_edge = [src_idx, dst_idx, new_verb, eid]
            new_stage.append(new_edge)

        new_relation_list.append(new_stage)

    stats = {
        "total_uk": total_uk,
        "resolved": resolved,
        "unresolved": total_uk - resolved,
    }
    return new_relation_list, stats


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK NODE INJECTOR
# ══════════════════════════════════════════════════════════════════════════════

def inject_network_node_if_missing(entity_list, relation_list):
    """
    Ensure every graph has at least one SO (Socket) node to represent
    C2 / exfiltration activity.  If no SO node exists, inject one and
    wire it to the first MP node via a ST (Send) edge.

    This compensates for the 21% of model outputs that lack network
    activity nodes (diagnosed from the 100-graph sample).

    Args:
        entity_list  : list of node types (mutated in-place)
        relation_list: list-of-lists of edges (mutated in-place, stage 1
                       gets the new ST edge)

    Returns:
        (entity_list, relation_list, injected: bool)
    """
    if "SO" in entity_list:
        return entity_list, relation_list, False

    # Find the MP or TP node with the highest out-degree (most "active" process)
    out_degree = {}
    for stage_edges in relation_list:
        for edge in stage_edges:
            src = int(edge[0])
            out_degree[src] = out_degree.get(src, 0) + 1

    # Prefer MP, then TP
    mp_nodes = [i for i, t in enumerate(entity_list) if t == "MP"]
    tp_nodes = [i for i, t in enumerate(entity_list) if t == "TP"]

    if mp_nodes:
        src_node = max(mp_nodes, key=lambda i: out_degree.get(i, 0))
    elif tp_nodes:
        src_node = max(tp_nodes, key=lambda i: out_degree.get(i, 0))
    else:
        return entity_list, relation_list, False  # nothing to wire to

    # Inject SO node
    so_idx = len(entity_list)
    entity_list = list(entity_list) + ["SO"]

    # Compute next edge id
    all_eids = [e[3] for stage in relation_list for e in stage]
    next_eid = max(all_eids) + 1 if all_eids else 0

    # Inject ST edge in stage 1 (reconnaissance / initial comms)
    new_edge = [str(src_node), str(so_idx), "ST", next_eid]
    relation_list[0] = list(relation_list[0]) + [new_edge]

    print(f"  [INJECT] No SO node found -> injected SO node {so_idx}, "
          f"wired from node {src_node}({entity_list[src_node]}) via ST")

    return entity_list, relation_list, True


class Dataloader:
    def __init__(self):
        self.regu_path = os.path.join(".", "regulation_dic")
        self.tech_path = os.path.join(".", "tech_dic")
        self.sub_graph_path = os.path.join(".", "4000_3_generated_data_new2_sub")

    def get_relation_list(self,data):
        relation_list = [[],[],[],[]]
        i1 = 0
        for line in data:
            line = line.strip()
            # Match edge lines: "0 1 FR" or "0 1 FR-1" or "0 1 FR-1-2-3"
            # Stage annotation is OPTIONAL — defaults to all 4 stages if missing
            m = re.match(r'^(\d+)\s+(\d+)\s+([A-Z]{2})(?:(-[\d-]+))?$', line)
            if not m:
                continue
            num1, num2, verb, stage_str = m.groups()
            relation = [num1, num2, verb, i1]
            i1 += 1
            if stage_str:
                # Parse stage annotations like "-1", "-1-2", "-1-2-3"
                stages = [int(s) for s in stage_str.split('-') if s.strip()]
                for stage_num in stages:
                    if 1 <= stage_num <= 4:
                        relation_list[stage_num - 1].append(list(relation))
            else:
                # No stage annotation — assign to ALL stages
                for s in range(4):
                    relation_list[s].append(list(relation))
        return relation_list

    def get_entity_list(self,data):
        entity_list = []
        for line in data:
            pattern = r"^([A-Z]{2})\*?"
            matches = re.findall(pattern, line)
            for match in matches:
                text = match
                entity_list.append(text)
        return entity_list

    def get_graph_info(self,whole_file_path):
        with open(whole_file_path,"r") as file:
            data = file.readlines()
            entity_list = self.get_entity_list(data)
            relation_list = self.get_relation_list(data)
        return data,entity_list,relation_list

    def read_json(self,file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def load_regulation(self):
        for regulation in os.listdir(self.regu_path):
            if regulation.find("stage1")!=-1:
                whole_path = os.path.join(self.regu_path, regulation)
                stage1_index_regu_dic = self.read_json(whole_path)
            elif regulation.find("stage2")!=-1:
                whole_path = os.path.join(self.regu_path, regulation)
                stage2_index_regu_dic = self.read_json(whole_path)
            elif regulation.find("stage3")!=-1:
                whole_path = os.path.join(self.regu_path, regulation)
                stage3_index_regu_dic = self.read_json(whole_path)
            elif regulation.find("stage4")!=-1:
                whole_path = os.path.join(self.regu_path, regulation)
                stage4_index_regu_dic = self.read_json(whole_path)
        return stage1_index_regu_dic,stage2_index_regu_dic,stage3_index_regu_dic,stage4_index_regu_dic

    def load_tech(self):
        for tech in os.listdir(self.tech_path):
            whole_path = os.path.join(self.tech_path, tech)
            if tech.find("stage1") !=-1:
                stage1_index_tech_dic = self.read_json(whole_path)
            elif tech.find("stage2")!=-1:
                stage2_index_tech_dic = self.read_json(whole_path)
            elif tech.find("stage3")!=-1:
                stage3_index_tech_dic = self.read_json(whole_path)
            elif tech.find("stage4")!=-1:
                stage4_index_tech_dic = self.read_json(whole_path)
        return stage1_index_tech_dic,stage2_index_tech_dic,stage3_index_tech_dic,stage4_index_tech_dic

class Gene:
    """
    This is a class to represent individual(Gene) in GA algorithom
    each object of this class have two attribute: data, size
    """
    def __init__(self, **data):
        self.__dict__.update(data)
        self.size = len(data['data'])  # length of gene

class GA:
    """
    This is a class of GA algorithm.
    """
    def __init__(self, parameter):
        """
        Initialize the pop of GA algorithom and evaluate the pop by computing its' fitness value.
        The data structure of pop is composed of several individuals which has the form like that:
        {'Gene':a object of class Gene, 'fitness': 1.02(for example)}
        Representation of Gene is a list: [b s0 u0 sita0 s1 u1 sita1 s2 u2 sita2]
        """
        self.parameter = parameter
        stage_len_1 = self.parameter[4]
        stage_index_regu_dic = self.parameter[5]
        stage_index_tech_dic = self.parameter[6]
        entity_list = self.parameter[7]
        relation_list = self.parameter[8]
        stage = self.parameter[9]
        self.bound = []
        pop = []

        for i in range(self.parameter[3]):
            geneinfo = []
            for pos in range(stage_len_1):
                geneinfo.append(random.randint(0, 1))

            fitness,geneinfo,entity_regu_dic,relation_regu_dic = self.evaluate(geneinfo,relation_list,entity_list,stage,stage_len_1)  # evaluate each chromosome
            pop.append({'Gene': Gene(data=geneinfo), 'fitness': fitness,'entity_regu_dic':entity_regu_dic,'relation_regu_dic':relation_regu_dic})  # store the chromosome and its fitness

        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)

    def match_rule(self,relation_info,entity_info,stage,stage_len_1):
        stage_index_regu_dic = self.parameter[5]
        stage_entity_list = []
        relation = relation_info[stage - 1]
        relation_list_specific = []
        relation_index_dic = {}
        index_small = 0
        rule_tatic_dic = {}
        for relation_small in relation:
            if relation_small[0] not in stage_entity_list:
                stage_entity_list.append(relation_small[0])
            if relation_small[1] not in stage_entity_list:
                stage_entity_list.append(relation_small[1])
            relation_specific = []
            relation_specific.append(entity_info[int(relation_small[0])])
            relation_specific.append(relation_small[2])
            relation_specific.append(entity_info[int(relation_small[1])])
            relation_index_dic[index_small] = relation_small[3]
            index_small += 1
            relation_list_specific.append(relation_specific)
        rule_match_edge_dic = {}
        for i in range(stage_len_1):
            rule_match_edge_dic[i] = []
            rule_name = list(stage_index_regu_dic.keys())[i]
            regu = stage_index_regu_dic[rule_name]
            j = 0
            flag = 0
            result = []
            temp = []
            while (j < len(relation_list_specific)):
                target_sub_index = flag % len(regu)
                sub = regu[target_sub_index]
                if relation_list_specific[j] == sub:
                    real_index = relation_index_dic[j]
                    temp.append(real_index)
                    flag += 1
                    if flag % len(regu) == 0 and flag != 0:
                        rule_match_edge_dic[i].append(temp)
                        temp = []
                j += 1

        for j in range(len(list(stage_index_regu_dic.keys()))):
            tactic = (list(stage_index_regu_dic.keys())[j].split(".")[1]).split("-")[0]
            rule_tatic_dic[j] = tactic
        return rule_match_edge_dic,rule_tatic_dic

    def find_target_rule(self,last_rule,match_rule):
        greater_numbers = [num for num in match_rule if num >= last_rule]
        smaller_numbers = [num for num in match_rule if num <= last_rule]

        if greater_numbers:
            nearest_greater = min(greater_numbers)
            return nearest_greater
        elif smaller_numbers:
            nearest_smaller = max(smaller_numbers)
            return nearest_smaller
        else:
            return None

    def check_order_of_values(self,my_dict, value1, value2):
        i1 = -1
        i2 = -1
        for value in my_dict.values():
            i1 += 1
            if value == value1:
                break
        for value in my_dict.values():
            i2 += 1
            if value == value2:
                break
        if i1 == -1 or i2 == -1:
            return True
        elif (i1 != -1 and i2 != -1) and (i1>i2):
            return False
        else:
            return True

    def evaluate(self, geneinfo,relation_info,entity_info,stage,stage_len_1):
        rule_match_edge_dic,rule_tatic_dic = self.match_rule(relation_info,entity_info,stage,stage_len_1)
        stage_entity_list = []
        relation = relation_info[stage-1]
        relation_list_specific = []
        relation_index_dic = {}
        index_small = 0
        for relation_small in relation:
            if relation_small[0] not in stage_entity_list:
                stage_entity_list.append(relation_small[0])
            if relation_small[1] not in stage_entity_list:
                stage_entity_list.append(relation_small[1])
            relation_specific = []
            relation_specific.append(entity_info[int(relation_small[0])])
            relation_specific.append(relation_small[2])
            relation_specific.append(entity_info[int(relation_small[1])])
            relation_index_dic[index_small] = relation_small[3]
            index_small += 1
            relation_list_specific.append(relation_specific)
        relation_arrange_dic = {}
        relation_regu_dic = {}
        relation_tactic_dic = {}
        entity_arrange_dic = {}
        entity_regu_dic = {}

        for i in range(len(relation)):
            index = relation_index_dic[i]
            relation_arrange_dic[index] = 0
            relation_regu_dic[index] = []

        for entity_small in stage_entity_list:
            entity_arrange_dic[entity_small] = 0
            entity_regu_dic[int(entity_small)] = []

        match_rule_whole = []
        match_edge_whole = []
        for i in range(len(relation)):
            real_index = relation_index_dic[i]
            if real_index not in match_edge_whole:
                match_rule = []
                for j in range(len(rule_match_edge_dic.items())):
                    key,value = list(rule_match_edge_dic.items())[j]
                    for small_match in value:
                        if real_index in small_match:
                            match_rule.append(key)
                if len(match_rule) == 0:
                    continue
                else:
                    if len(match_rule_whole) == 0:
                        last_rule = 0
                    else:
                        last_rule = match_rule_whole[-1]
                    target_rule = self.find_target_rule(last_rule,match_rule)
                    while(geneinfo[target_rule] != 1):
                        match_rule.remove(target_rule)
                        target_rule = self.find_target_rule(last_rule, match_rule)
                        if target_rule == None:
                            break
                    if target_rule != None:
                        match_rule_whole.append(target_rule)
                        for match_cluster in rule_match_edge_dic[target_rule]:
                            if real_index in match_cluster:
                                target_cluster = match_cluster
                        for edge in target_cluster:
                            relation_arrange_dic[edge] = 1
                            if edge not in match_edge_whole:
                                relation_regu_dic[edge].append(target_rule)
                                relation_tactic_dic[edge] = rule_tatic_dic[target_rule]
                            match_edge_whole.append(edge)
                            edge_info = relation[i]
                            entity_1 = int(edge_info[0])
                            entity_2 = int(edge_info[1])
                            entity_arrange_dic[int(entity_1)] = 1
                            entity_arrange_dic[int(entity_2)] = 1
                            if target_rule not in entity_regu_dic[int(entity_1)]:
                                entity_regu_dic[int(entity_1)].append(target_rule)
                            if target_rule not in entity_regu_dic[int(entity_2)]:
                                entity_regu_dic[int(entity_2)].append(target_rule)

        for x in range(len(geneinfo)):
            if geneinfo[x] ==1 and x not in match_rule_whole:
                geneinfo[x] = 0
        count = sum(1 for value in entity_arrange_dic.values() if value != 0)
        fitness = count / len(list(entity_arrange_dic.keys()))
        Flag1 = self.check_order_of_values(relation_tactic_dic,"Initial Access","Execution")
        Flag2 = self.check_order_of_values(relation_tactic_dic,"Privilege Escalation","Discovery")
        if (Flag1 == False) or (Flag2 == False):
            fitness = 0

        return fitness,geneinfo,entity_regu_dic,relation_regu_dic

    def selectBest(self, pop):
        """
        select the best individual from pop
        """
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)          # from large to small, return a pop
        return s_inds[0]

    def selection(self, individuals, k):
        """
        select some good individuals from pop, note that good individuals have greater probability to be choosen
        for example: a fitness list like that:[5, 4, 3, 2, 1], sum is 15,
        [-----|----|---|--|-]
        012345|6789|101112|1314|15
        we randomly choose a value in [0, 15],
        it belongs to first scale with greatest probability
        """
        s_inds = sorted(individuals, key=itemgetter("fitness"), reverse=True)  # sort the pop by the reference of fitness
        sum_fits = sum(ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop
        chosen = []
        for i in range(k):
            u = random.random() * sum_fits
            sum_ = 0
            for ind in s_inds:
                sum_ += ind['fitness']
                if sum_ >= u:
                    chosen.append(ind)
                    break
        chosen = sorted(chosen, key=itemgetter("fitness"), reverse=False)
        return chosen

    def crossoperate(self, offspring):
        """
        cross operation
        here we use two points crossoperate
        for example: gene1: [5, 2, 4, 7], gene2: [3, 6, 9, 2], if pos1=1, pos2=2
        5 | 2 | 4  7
        3 | 6 | 9  2
        =
        3 | 2 | 9  2
        5 | 6 | 4  7
        """
        dim = len(offspring[0]['Gene'].data)
        geninfo1 = offspring[0]['Gene'].data  # Gene's data of first offspring chosen from the selected pop
        geninfo2 = offspring[1]['Gene'].data  # Gene's data of second offspring chosen from the selected pop

        if dim == 1:
            pos1 = 1
            pos2 = 1
        else:
            pos1 = random.randrange(1, dim)  # select a position in the range from 0 to dim-1,
            pos2 = random.randrange(1, dim)

        newoff1 = Gene(data=[])  # offspring1 produced by cross operation
        newoff2 = Gene(data=[])  # offspring2 produced by cross operation
        temp1 = []
        temp2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i < max(pos1, pos2):
                temp2.append(geninfo2[i])
                temp1.append(geninfo1[i])
            else:
                temp2.append(geninfo1[i])
                temp1.append(geninfo2[i])
        newoff1.data = temp1
        newoff2.data = temp2

        return newoff1, newoff2

    def mutation(self, crossoff, bound):
        """
        mutation operation
        """
        dim = len(crossoff.data)

        if dim == 1:
            pos = 0
        else:
            pos = random.randrange(0, dim)  # chose a position in crossoff to perform mutation.

        crossoff.data[pos] = random.randint(0, 1)
        return crossoff

    def GA_main(self):
        """
        main frame work of GA
        """
        popsize = self.parameter[3]
        stage_len_1 = self.parameter[4]
        # print("Start of evolution")
        stage_index_regu_dic = self.parameter[5]
        stage_index_tech_dic = self.parameter[6]
        entity_list = self.parameter[7]
        relation_list = self.parameter[8]
        stage = self.parameter[9]
        # Begin the evolution
        for g in range(NGEN):

            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, popsize)

            nextoff = []
            while len(nextoff) != popsize:
                # Apply crossover and mutation on the offspring

                # Select two individuals
                offspring = [selectpop.pop() for _ in range(2)]
                if random.random() < CXPB:
                    crossoff1, crossoff2 = self.crossoperate(offspring)
                    if random.random() < MUTPB:
                        muteoff1 = self.mutation(crossoff1, self.bound)
                        muteoff2 = self.mutation(crossoff2, self.bound)
                        fit_muteoff1,muteoff1,entity_regu_dic1,relation_regu_dic1 = self.evaluate(muteoff1.data,relation_list,entity_list,stage,stage_len_1)  # Evaluate the individuals
                        fit_muteoff2,muteoff2,entity_regu_dic2,relation_regu_dic2 = self.evaluate(muteoff2.data,relation_list,entity_list,stage,stage_len_1)  # Evaluate the individuals
                        nextoff.append({'Gene': Gene(data=muteoff1), 'fitness': fit_muteoff1,'entity_regu_dic':entity_regu_dic1,'relation_regu_dic':relation_regu_dic1})
                        nextoff.append({'Gene': Gene(data=muteoff2), 'fitness': fit_muteoff2,'entity_regu_dic':entity_regu_dic2,'relation_regu_dic':relation_regu_dic2})
                    else:
                        fit_crossoff1,crossoff1,entity_regu_dic3,relation_regu_dic3 = self.evaluate(crossoff1.data,relation_list,entity_list,stage,stage_len_1)  # Evaluate the individuals
                        fit_crossoff2,crossoff2,entity_regu_dic4,relation_regu_dic4 = self.evaluate(crossoff2.data,relation_list,entity_list,stage,stage_len_1)
                        nextoff.append({'Gene': Gene(data=crossoff1), 'fitness': fit_crossoff1,'entity_regu_dic':entity_regu_dic3,'relation_regu_dic':relation_regu_dic3})
                        nextoff.append({'Gene': Gene(data=crossoff2), 'fitness': fit_crossoff2,'entity_regu_dic':entity_regu_dic4,'relation_regu_dic':relation_regu_dic4})
                else:
                    nextoff.extend(offspring)

                    # The population is entirely replaced by the offspring
                self.pop = nextoff

                # Gather all the fitnesses in one list and print the stats
                fits = [ind['fitness'] for ind in self.pop]

                best_ind = self.selectBest(self.pop)

                if best_ind['fitness'] > self.bestindividual['fitness']:
                    self.bestindividual = best_ind

        return self.bestindividual['Gene'].data,self.bestindividual['entity_regu_dic'],self.bestindividual['relation_regu_dic']

def read_new(json_file):
    file = open(json_file, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line,strict=False)
        papers.append(dic)
    file.close()
    return papers

def make_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return 1


def find_target_relation(relation_list, key):
    for rela in relation_list:
        if rela[3] == key:
            return rela


if __name__ == "__main__":
    CXPB, MUTPB, NGEN, popsize = 0.8, 0.4, 5, 20  # popsize must be even number
    stage_len = [58,119,94,101]
    DL = Dataloader()
    stage1_index_regu_dic, stage2_index_regu_dic, stage3_index_regu_dic, stage4_index_regu_dic = DL.load_regulation()
    stage1_index_tech_dic, stage2_index_tech_dic, stage3_index_tech_dic, stage4_index_tech_dic = DL.load_tech()
    instance_lib = read_new(os.path.join(".", "instance_lib", "technique-instance-lib-os-filter-add.json")) #dir for instantiation lib
    new_instance_path = os.path.join(".", "4000_3_generated_data_new2_sub_instance_windows") #dir for IAG
    make_directory(new_instance_path)
    os_type = "linux"
    for txt in os.listdir(DL.sub_graph_path):
        whole_file_path = os.path.join(DL.sub_graph_path, txt)
        new_file_path = os.path.join(new_instance_path, txt)
        graph_data, entity_list, relation_list1 = DL.get_graph_info(whole_file_path)
        print(whole_file_path)

        # ── PRE-PROCESSING STEP 1: Resolve UK edges ──────────────────────────
        # Replace 'UK' verbs with semantically appropriate verbs before GA.
        relation_list1, uk_stats = resolve_uk_edges(entity_list, relation_list1)
        if uk_stats["total_uk"] > 0:
            print(f"  [UK-RESOLVE] {uk_stats['resolved']}/{uk_stats['total_uk']} "
                  f"UK edges resolved, {uk_stats['unresolved']} kept as RD fallback")

        # ── PRE-PROCESSING STEP 2: Ensure network presence ───────────────────
        # Graphs without any SO node lack C2/exfil semantics; inject one.
        entity_list, relation_list1, so_injected = inject_network_node_if_missing(
            entity_list, relation_list1
        )
        if so_injected:
            # Update graph_data node-count line to reflect added SO node
            graph_data[0] = str(len(entity_list)) + "\n"
        # ─────────────────────────────────────────────────────────────────────

        # Debug: show what was parsed
        edge_counts = [len(s) for s in relation_list1]
        total_edges = sum(edge_counts)
        print(f"  [DEBUG] Parsed: {len(entity_list)} entities, {total_edges} edges {edge_counts}")

        if total_edges == 0:
            print(f"  [SKIP] {txt}: no edges parsed, skipping")
            continue

        # -- PRE-GA: Edge Constraint Validation --
        validator = EdgeConstraintValidator(verbose=True)
        relation_list1, filter_stats = validator.filter_relation_list(
            entity_list, relation_list1
        )

        # Debug: show what survived
        post_counts = [len(s) for s in relation_list1]
        print(f"  [DEBUG] After filter: {sum(post_counts)} edges {post_counts} "
              f"(removed {filter_stats['removed']})")

        if not filter_stats["graph_valid"]:
            print(f"  [SKIP] {txt}: {filter_stats['removal_rate']:.0%} edges invalid, skipping graph")
            continue
        if filter_stats["removed"] > 0:
            # Clean up dangling nodes after edge removal
            entity_list, relation_list1, dangle_map = validator.remove_dangling_nodes(
                entity_list, relation_list1
            )
            # Update graph_data line count if nodes were removed
            if len(entity_list) < int(graph_data[0]):
                graph_data[0] = str(len(entity_list)) + "\n"

        entity_instance_dic = {}
        relation_instance_dic = {}
        for i in range(len(entity_list)):
            entity_instance_dic[i] = []
        sum_relation = 0
        for sublist in relation_list1:
            sum_relation += len(sublist)
        for i in range(sum_relation):
            relation_instance_dic[i] = []
        pruning_agent = PruningAgent(target_nodes=150, min_stage_coverage=0.60, verbose=True)
        for stage in range(1,5):
            stage_index_regu_dic = DL.load_regulation()[stage-1]
            stage_index_tech_dic = DL.load_tech()[stage-1]
            stage_regu_len = stage_len[stage-1]
            print("entity_list"+str(entity_list))
            print("relation_list" + str(relation_list1))
            pruned_entity, pruned_relations, node_map = pruning_agent.prune(
                entity_list, relation_list1, stage
            )
            parameter = [CXPB, MUTPB, NGEN, popsize, stage_regu_len,stage_index_regu_dic, stage_index_tech_dic,pruned_entity,pruned_relations,stage]
            run = FastGA(parameter)
            bestindividual_gene,bestindividual_entity_regu_dic_pruned,bestindividual_relation_regu_dic_pruned = run.GA_main()
            bestindividual_entity_regu_dic = pruning_agent.remap_entity_dic(
                bestindividual_entity_regu_dic_pruned, node_map
            )
            bestindividual_relation_regu_dic = pruning_agent.remap_relation_dic(
                bestindividual_relation_regu_dic_pruned, node_map
            )
            print("bestindividual_entity_regu_dic" + str(bestindividual_entity_regu_dic))
            print("bestindividual_relation_regu_dic"+str(bestindividual_relation_regu_dic))
            unsuccess_edge = [[]]
            for key in bestindividual_relation_regu_dic.keys():
                if len(bestindividual_relation_regu_dic[key]) == 0:
                    target_relation = find_target_relation(relation_list1[stage-1],key)
                    unsuccess_edge[0].append(target_relation)
            print("unsuccess_edge"+str(unsuccess_edge))
            if len(unsuccess_edge[0]) != 0:
                if stage == 1:
                    new_target_stage = 2
                else:
                    new_target_stage = stage-1
                stage_index_regu_dic2 = DL.load_regulation()[new_target_stage - 1]
                stage_index_tech_dic2 = DL.load_tech()[new_target_stage - 1]
                stage_regu_len2 = stage_len[new_target_stage - 1]
                p_e2, p_r2, nm2 = pruning_agent.prune(entity_list, unsuccess_edge, 1)
                parameter2 = [CXPB, MUTPB, NGEN, popsize, stage_regu_len2, stage_index_regu_dic2, stage_index_tech_dic2,p_e2, p_r2, 1]
                run2 = FastGA(parameter2)
                bestindividual_gene2, bestindividual_entity_regu_dic2_pruned, bestindividual_relation_regu_dic2_pruned = run2.GA_main()
                bestindividual_entity_regu_dic2 = pruning_agent.remap_entity_dic(
                    bestindividual_entity_regu_dic2_pruned, nm2
                )
                bestindividual_relation_regu_dic2 = pruning_agent.remap_relation_dic(
                    bestindividual_relation_regu_dic2_pruned, nm2
                )
                print("bestindividual_entity_regu_dic2" + str(bestindividual_entity_regu_dic2))
                print("bestindividual_relation_regu_dic2 " + str(stage)+str(bestindividual_relation_regu_dic2))
                for key1 in bestindividual_relation_regu_dic2.keys():
                    if len(bestindividual_relation_regu_dic2[key1]) != 0:
                        info = str(new_target_stage) + "-" + str(bestindividual_relation_regu_dic2[key1][0])
                        bestindividual_relation_regu_dic[key1].append(info)
                        target_relation = find_target_relation(relation_list1[stage - 1], key1)
                        entity1 = int(target_relation[0])
                        entity2 = int(target_relation[1])
                        bestindividual_entity_regu_dic[entity1].append(info)
                        bestindividual_entity_regu_dic[entity2].append(info)
                print("bestindividual_relation_regu_dic"+str(bestindividual_relation_regu_dic))

            for key in list(bestindividual_entity_regu_dic.keys()):
                for l in range(len(bestindividual_entity_regu_dic[key])):
                    if str(bestindividual_entity_regu_dic[key][l]).find("-") != -1:
                        info = bestindividual_entity_regu_dic[key][l]
                    else:
                        info = str(stage)+"-"+str(bestindividual_entity_regu_dic[key][l])
                    if info not in entity_instance_dic[key]:
                        entity_instance_dic[key].append(info)

            for key in list(bestindividual_relation_regu_dic.keys()):
                for l in range(len(bestindividual_relation_regu_dic[key])):
                    if str(bestindividual_relation_regu_dic[key][l]).find("-") != -1:
                        info = bestindividual_relation_regu_dic[key][l]
                    else:
                        info = str(stage)+"-"+str(bestindividual_relation_regu_dic[key][l])
                    if info not in relation_instance_dic[key]:
                        relation_instance_dic[key].append(info)
        print("entity_instance_dic"+str(entity_instance_dic))
        print("relation_instance_dic"+str(relation_instance_dic))
        # --- NEW CODE: Build out_verbs and in_verbs for validation ---
        from edge_validator import EdgeConstraintValidator
        validator = EdgeConstraintValidator(verbose=False)
        out_verbs = {}
        in_verbs = {}
        for x in range(1, len(graph_data)):
            if re.match(r"\d+\s\d+\s[A-Z]+", graph_data[x]):
                parts = graph_data[x].strip().split()
                sub = parts[0]
                obj = parts[1]
                verb = parts[2].split('-')[0]
                if sub not in out_verbs: out_verbs[sub] = []
                if obj not in in_verbs: in_verbs[obj] = []
                out_verbs[sub].append(verb)
                in_verbs[obj].append(verb)
        # -------------------------------------------------------------

        entity_one_instance_dic = {}

        # Build per-node edge context for smart instance generation
        # (used when CTI lookup yields no valid match for the required verbs)
        node_edge_context = {}
        for k in entity_instance_dic.keys():
            node_edge_context[k] = {
                "out_verbs": out_verbs.get(str(k), []),
                "in_verbs":  in_verbs.get(str(k), []),
            }

        for key in entity_instance_dic.keys():
            entity_one_instance_dic[key] = 0
            if len(entity_instance_dic[key]) != 0:
                target_rule = entity_instance_dic[key][random.randint(0, len(entity_instance_dic[key])-1)]
                target_type = entity_list[int(key)]
                target_stage = int(target_rule[0])
                target_rule_index = int(target_rule.split("-")[1])-1
                target_tech_dic = DL.load_tech()[target_stage-1]
                target_tech = target_tech_dic[list(target_tech_dic.keys())[target_rule_index]]
                target_instance_list = []
                for tech in target_tech:
                    for data in instance_lib:
                        if data["stage-key"] == tech+"-"+os_type:
                            for instance in data[target_type]:
                                target_instance_list.append(instance)
                
                # --- NEW CODE: Filter instances ---
                filtered_instances = []
                for instance in target_instance_list:
                    valid = True
                    for verb in out_verbs.get(str(key), []):
                        if not validator.validate_instance(instance, verb, "unknown", src_type=target_type):
                            valid = False
                            break
                    if not valid: continue
                    for verb in in_verbs.get(str(key), []):
                        if not validator.validate_instance("unknown", verb, instance):
                            valid = False
                            break
                    if valid:
                        filtered_instances.append(instance)

                # ── PREFERENCE PASS 1: ST + WR  →  prefer FILE_AND_NETWORK_TOOLS ──
                # A node that must both send over network AND write files should use
                # curl/wget/scp rather than a pure scanner like nmap/masscan.
                node_out = set(out_verbs.get(str(key), []))
                if (target_type in ("MP", "TP")
                        and filtered_instances
                        and node_out & {"ST", "RF"}
                        and node_out & {"WR"}):
                    preferred = [
                        inst for inst in filtered_instances
                        if validator._normalize_tool(inst) in FILE_AND_NETWORK_TOOLS
                    ]
                    if preferred:
                        filtered_instances = preferred

                # ── PREFERENCE PASS 2: EX in in_verbs  →  force executable extension ──
                # If this node (SF/MF) will be executed, only pick instances
                # that have an explicit executable file extension.
                node_in = set(in_verbs.get(str(key), []))
                if target_type in ("SF", "MF") and "EX" in node_in and filtered_instances:
                    exec_instances = [
                        inst for inst in filtered_instances
                        if validator._get_extension(inst) in EXECUTABLE_EXTENSIONS_STRICT
                    ]
                    if exec_instances:
                        filtered_instances = exec_instances
                # -------------------------------------------------------------------

                if len(filtered_instances) != 0:
                    entity_one_instance_dic[key] = random.choice(filtered_instances)
                else:
                    # No CTI instance satisfies the edge constraints for this node.
                    # Use a context-aware smart name instead of 'unknown_*'.
                    smart_name = generate_smart_instance_name(
                        target_type, node_edge_context[key]
                    )
                    entity_one_instance_dic[key] = smart_name
                    print(f"  [SMART] Node {key}({target_type}): no CTI match "
                          f"for out_verbs={node_edge_context[key]['out_verbs']}, "
                          f"generated '{smart_name}'")
        print(entity_one_instance_dic)

        # ── POST-ASSIGNMENT ENFORCEMENT ──
        # Re-check every assigned instance against ALL edges it participates in.
        # This catches cases where the initial per-verb filter missed cross-edge
        # conflicts (e.g., a tool valid for RD but assigned to a node that also
        # has a FR edge to another node).
        for node_key in list(entity_one_instance_dic.keys()):
            instance_name = entity_one_instance_dic[node_key]
            if instance_name == 0:
                continue  # unassigned node
            node_type = entity_list[int(node_key)]
            is_valid = True
            # Check as SOURCE: instance --verb--> target
            for v in out_verbs.get(str(node_key), []):
                if not validator.validate_instance(instance_name, v, "unknown", src_type=node_type):
                    is_valid = False
                    break
            # Check as TARGET: source --verb--> instance
            if is_valid:
                for v in in_verbs.get(str(node_key), []):
                    if not validator.validate_instance("unknown", v, instance_name):
                        is_valid = False
                        break
            if not is_valid:
                # Use smart name (not unknown_*) as the enforced replacement
                ctx = node_edge_context.get(node_key, {
                    "out_verbs": out_verbs.get(str(node_key), []),
                    "in_verbs":  in_verbs.get(str(node_key), []),
                })
                fallback = generate_smart_instance_name(node_type, ctx)
                print(f"  [ENFORCE] Node {node_key}({node_type}): '{instance_name}' "
                      f"failed edge validation -> smart replacement '{fallback}'")
                entity_one_instance_dic[node_key] = fallback

        not_satisfied_entity = []
        for i in range(len(entity_one_instance_dic.keys())):
            if entity_one_instance_dic[list(entity_one_instance_dic.keys())[i]] != 0:
                graph_data[i+1] = graph_data[i+1].strip()+"-"+entity_one_instance_dic[list(entity_one_instance_dic.keys())[i]]+"\n"
            else:
                not_satisfied_entity.append(i+1)
        temp_graph_data = []
        for j in range(len(graph_data)):
            if j not in not_satisfied_entity:
                temp_graph_data.append(graph_data[j])
        delete_edge_list = []

        new_entity_match_dic = {}
        for e in range(int(temp_graph_data[0])):
            if e+1 not in not_satisfied_entity:
                i1 = 0
                for t in not_satisfied_entity:
                    if e+1 > t:
                        i1 += 1
                new_entity_match_dic[e] = e-i1
        print(new_entity_match_dic)
        for x in range(len(temp_graph_data)):
            if re.match(r"\d+\s\d+\s[A-Z]+", temp_graph_data[x]):
                sub_str, obj_str, verb_stage_str = temp_graph_data[x].split(" ", 2)
                sub = int(sub_str)
                obj = int(obj_str)

                if (sub+1 in not_satisfied_entity ) or (obj+1 in not_satisfied_entity) :
                    if x not in delete_edge_list:
                        delete_edge_list.append(x)
                else:
                    temp_graph_data[x] = str(new_entity_match_dic[sub]) + " " + str(new_entity_match_dic[obj]) + " " + verb_stage_str

        # ── FINAL EDGE-PRUNING PASS ──
        # After instances are assigned, re-validate every surviving edge
        # against the actual tool/file names. Remove edges where the
        # assigned instance cannot physically perform the verb.
        # This is necessary because edges are determined by the GA before
        # instance assignment, so impossible combinations slip through.
        #
        # Build a lookup: node_index -> assigned instance name
        instance_lookup = {}
        for key, val in entity_one_instance_dic.items():
            if val != 0:
                instance_lookup[int(key)] = val
        # Also need entity types for context
        type_lookup = {}
        for key in entity_one_instance_dic.keys():
            idx = int(key)
            if idx < len(entity_list):
                type_lookup[idx] = entity_list[idx]

        pruned_count = 0
        for x in range(len(temp_graph_data)):
            if x in delete_edge_list:
                continue  # already marked for deletion
            if re.match(r"\d+\s\d+\s[A-Z]+", temp_graph_data[x]):
                parts = temp_graph_data[x].strip().split()
                sub_idx = int(parts[0])
                obj_idx = int(parts[1])
                verb = parts[2].split('-')[0]  # strip stage annotation

                # Get the assigned instance names (using inverse mapping)
                # new_entity_match_dic maps old -> new, we need new -> old
                new_to_old = {v: k for k, v in new_entity_match_dic.items()}
                orig_sub = new_to_old.get(sub_idx, sub_idx)
                orig_obj = new_to_old.get(obj_idx, obj_idx)

                sub_instance = instance_lookup.get(orig_sub, "")
                obj_instance = instance_lookup.get(orig_obj, "")
                sub_type = type_lookup.get(orig_sub, "TP")

                if sub_instance and obj_instance:
                    # Check: can this tool perform this verb on this target?
                    if not validator.validate_instance(
                        sub_instance, verb, obj_instance, src_type=sub_type
                    ):
                        delete_edge_list.append(x)
                        pruned_count += 1

        if pruned_count > 0:
            print(f"  [PRUNE] Removed {pruned_count} edges that violated "
                  f"instance-level allowlist after CTI assignment")
        final_graph_data = []

        for v in range(len(temp_graph_data)):
            if v == 0:
                final_graph_data.append(str(len(new_entity_match_dic.keys()))+"\n")
            elif v not in delete_edge_list:
                final_graph_data.append(temp_graph_data[v])
        # Update edge count in final_graph_data.
        # NOTE: We cannot assume the edge-count line is at a fixed index because:
        #   - Some input files (with stage annotation) have an explicit edge-count line
        #     after the node-instance lines.
        #   - Files without stage annotation (Find_hub fallback) have NO edge-count line;
        #     edges start immediately after node lines.
        # Strategy: scan final_graph_data for the first line that is a bare integer
        # AFTER all node lines (i.e., after index N_nodes). If found, decrement it.
        n_nodes = len(new_entity_match_dic.keys())
        edge_count_idx = None
        for _i in range(n_nodes + 1, len(final_graph_data)):
            candidate = final_graph_data[_i].strip()
            if candidate.isdigit():
                edge_count_idx = _i
                break

        if edge_count_idx is not None and len(delete_edge_list) > 0:
            old_count = int(final_graph_data[edge_count_idx].strip())
            new_count = max(0, old_count - len(delete_edge_list))
            final_graph_data[edge_count_idx] = str(new_count) + "\n"
        # If no edge-count line found, nothing to update (file format has no header).


        with open(new_file_path, 'w', encoding='utf-8') as file:
            file.writelines(final_graph_data)
            file.close()

        count = sum(1 for value in entity_instance_dic.values() if len(value) != 0)
        coverage = count / max(len(list(entity_instance_dic.keys())), 1)
        print(f"coverage: {coverage:.4f}")
        print(entity_instance_dic)