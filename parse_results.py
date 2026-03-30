#!/usr/bin/env python3
"""Parse HPGMG output file and generate summary table."""
import sys
import re

def parse_results(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    results = []
    current_grid = None
    solves = []
    in_richardson = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Grid header
        m = re.match(r'=+ 2\^(\d+) = (\d+)x(\d+)x(\d+) grid =+', line)
        if m:
            if current_grid and solves:
                results.append((current_grid, solves))
            log2 = int(m.group(1))
            dim = int(m.group(2))
            current_grid = {'log2': log2, 'dim': dim, 'levels': log2 - 1}
            solves = []
            in_richardson = False
            i += 1
            continue

        # Detect Richardson section
        if 'Richardson error analysis' in line:
            in_richardson = True
            i += 1
            continue

        # MGSolve time + v-cycles + bottom solver iterations
        m = re.match(r'Total time in MGSolve\s+([\d.]+) seconds', line)
        if m:
            solve_time = float(m.group(1))
            vcycles = 0
            bottom_iters = 0
            if i + 1 < len(lines):
                m2 = re.match(r'number of v-cycles\s+(\d+)', lines[i+1].strip())
                if m2:
                    vcycles = int(m2.group(1))
            if i + 2 < len(lines):
                m3 = re.match(r'Bottom solver iterations\s+(\d+)', lines[i+2].strip())
                if m3:
                    bottom_iters = int(m3.group(1))
            solves.append({'time': solve_time, 'vcycles': vcycles, 'bottom_iters': bottom_iters})
            i += 1
            continue

        # "done" lines in Richardson section only
        # Handles: iter= N norm=... rel=... done  AND  v-cycle= N norm=... rel=... done
        if in_richardson and 'done' in line:
            m = re.search(r'(?:iter|v-cycle)=\s*(\d+)\s+norm=([\d.e+-]+)\s+rel=([\d.e+-]+)\s+done', line)
            if m and current_grid:
                current_grid.setdefault('richardson_solves', [])
                current_grid['richardson_solves'].append({
                    'iters': int(m.group(1)),
                    'norm': float(m.group(2)),
                    'rel': float(m.group(3)),
                })
                i += 1
                continue

        # Richardson error value
        m = re.match(r'h=([\d.e+-]+)\s+\|\|error\|\|=([\d.e+-]+)', line)
        if m and current_grid:
            current_grid['h'] = float(m.group(1))
            current_grid['error'] = float(m.group(2))
            i += 1
            continue

        # Order
        m = re.match(r'order=([\d.]+)', line)
        if m and current_grid:
            current_grid['order'] = float(m.group(1))
            i += 1
            continue

        i += 1

    if current_grid and solves:
        results.append((current_grid, solves))

    return results


def print_table(results):
    print()
    print("HPGMG FP32 Results Summary (h solve only)")
    w = 130
    print("=" * w)
    fmt = "{:<8} {:>6} {:>8} {:>10} {:>13} {:>8} {:>13} {:>14} {:>11} {:>7} {:>9}"
    print(fmt.format(
        "Grid", "Levels", "V-cycles", "Bot.Iters", "||r||/||F||", "Stalled",
        "MGSolve(s)", "Time/Vcyc(s)", "||error||", "Order", "Converged"
    ))
    print("-" * w)

    for grid, solves in results:
        dim = grid['dim']
        levels = grid.get('levels', '?')

        # First solve = finest (h)
        if solves:
            s = solves[0]
            vcycles = s['vcycles']
            total_time = s['time']
            bottom_iters = s.get('bottom_iters', 0)
            time_per_vc = total_time / vcycles if vcycles > 0 else 0
            stalled = "YES" if vcycles >= 20 else "No"
        else:
            vcycles = 0
            total_time = 0
            time_per_vc = 0
            bottom_iters = 0
            stalled = "?"

        # Final rel from Richardson first solve (finest)
        rel = "?"
        if 'richardson_solves' in grid and grid['richardson_solves']:
            rel_val = grid['richardson_solves'][0]['rel']
            rel = f"{rel_val:.2e}"

        error = f"{grid['error']:.2e}" if 'error' in grid else "?"
        order = f"{grid['order']:.2f}" if 'order' in grid else "?"
        converged = "No" if stalled == "YES" else "Yes"

        print(fmt.format(
            f"{dim}^3", str(levels), str(vcycles), str(bottom_iters), rel,
            stalled, f"{total_time:.4f}", f"{time_per_vc:.4f}",
            error, order, converged
        ))

    print("=" * w)
    print()
    print("Notes:")
    print("  - Stalled = hit max v-cycles (20 for MGPCG, 100 for MGSolve)")
    print("  - Bot.Iters = total bottom solver iterations across all v-cycles")
    print("  - ||r||/||F|| = infinity norm of residual / infinity norm of RHS")
    print("  - Order should be ~2.0 for 2nd-order scheme (7-point Laplacian)")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_file>")
        sys.exit(1)
    results = parse_results(sys.argv[1])
    print_table(results)
