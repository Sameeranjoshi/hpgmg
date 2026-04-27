#!/usr/bin/env python3
"""Parse HPGMG result files from paper_results_sc/{with,without}_convergence
and print summary tables: Total solve time, v-cycles, time per v-cycle."""

import re, os

def parse_results(filepath):
    """For each grid size, get MGSolve time and vcycles from the FIRST 'Running' block only."""
    data = {}
    with open(filepath) as f:
        text = f.read()

    grid_sections = re.split(r'={3,}\s*2\^(\d+)\s*=\s*\d+x\d+x\d+\s*grid\s*={3,}', text)

    for i in range(1, len(grid_sections), 2):
        log2 = int(grid_sections[i])
        section = grid_sections[i + 1]

        running_blocks = re.split(r'={3,}\s*Running\s+\d+\s+solves\s*={3,}', section)

        if len(running_blocks) < 2:
            continue

        first_run = running_blocks[1]

        mg_match = re.search(r'Total time in MGSolve\s+([\d.]+)', first_run)
        vc_match = re.search(r'number of v-cycles\s+(\d+)', first_run)
        bi_match = re.search(r'Bottom solver iterations\s+(\d+)', first_run)

        if mg_match and vc_match:
            mgsolve = float(mg_match.group(1))
            vcycles = int(vc_match.group(1))
            bottom = int(bi_match.group(1)) if bi_match else 0
            data[log2] = {
                'mgsolve': mgsolve,
                'vcycles': vcycles,
                'bottom_iters': bottom,
                'time_per_vcycle': mgsolve / vcycles if vcycles > 0 else 0,
            }
    return data


def print_tables(label, all_data, ordered_keys, display_names, log2_sizes):
    print(f"\n{'='*120}")
    print(f"  {label}")
    print(f"{'='*120}")

    header = f"{'Grid':>12}"
    for cn in ordered_keys:
        header += f" | {'TTS':>10} {'Iter':>5} {'1-cycle':>10}"
    print()
    # Sub-header with config names
    name_header = f"{'':>12}"
    for cn in ordered_keys:
        name_header += f" | {display_names[cn]:^27}"
    print(name_header)
    print(header)
    print("-" * len(header))
    for log2 in log2_sizes:
        gs = 1 << log2
        row = f"{gs:>5}^3 ({log2})"
        for cn in ordered_keys:
            d = all_data[cn].get(log2)
            if d:
                row += f" | {d['mgsolve']:>10.6f} {d['vcycles']:>5d} {d['time_per_vcycle']:>10.6f}"
            else:
                row += f" | {'N/A':>10} {'N/A':>5} {'N/A':>10}"
        print(row)
    print()


def process_folder(folder_path, label):
    config_order = ["6_6_100", "4_4_6", "6_6_6", "4_4_100"]
    display_names = {
        "6_6_100": "GH200(6/6/100)",
        "4_4_6":   "GH200(4/4/6)",
        "6_6_6":   "GH200(6/6/6)",
        "4_4_100": "GH200(4/4/100)",
    }
    log2_sizes = [4, 5, 6, 7, 8, 9]

    all_data = {}
    for config in config_order:
        # Try multiple naming patterns
        for pat in [f"glow_{config}.txt", f"glow_{config}_conv.txt", f"glow_{config}_no_conv.txt"]:
            fp = os.path.join(folder_path, pat)
            if os.path.exists(fp):
                all_data[config] = parse_results(fp)
                break
        else:
            print(f"WARNING: No file found for config {config} in {folder_path}")

    if not all_data:
        print(f"No results found in {folder_path}!")
        return

    ordered_keys = [c for c in config_order if c in all_data]
    print_tables(label, all_data, ordered_keys, display_names, log2_sizes)


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_results_sc")

if os.path.isdir(base_dir):
    process_folder(base_dir, "WITH convergence check")
else:
    print(f"Folder not found: {base_dir}")
