import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import pandas as pd

def parse_conjecture_file(filepath):
    rows = []
    current_stratum = None
    current_target  = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect stratum header
            if line.startswith('--- model=') or line.startswith('--- size='):
                parts = line.strip('-').strip().split('|')
                current_stratum = parts[0].strip()
                current_target  = parts[1].replace('target=', '').strip() if len(parts) > 1 else None
                n_str = parts[2].strip() if len(parts) > 2 else 'n=?'

            # Detect conjecture lines
            elif line.startswith('Conjecture'):
                match = re.match(
                    r'Conjecture (\d+)\. (.+?)\s+\[touches=(\d+), support=(\d+)\]',
                    line
                )
                if match and current_stratum:
                    rows.append({
                        'stratum':    current_stratum,
                        'target':     current_target,
                        'rank':       int(match.group(1)),
                        'formula':    match.group(2),
                        'touches':    int(match.group(3)),
                        'support':    int(match.group(4)),
                    })

    return pd.DataFrame(rows)

if __name__ == '__main__':
    from utils import test_utils
    test_utils.set_dir(test_utils.get_path())
    df = parse_conjecture_file("../../results/scaled_topologies/conjecture_output.txt")
    df.to_csv("../../results/scaled_topologies/conjectures_parsed.csv", index=False)

    # Quick summary — highest touch counts
    print(df[df.touches > 0].sort_values('touches', ascending=False).head(20))
    print(f"\nTotal conjectures parsed: {len(df)}")
    print(f"Conjectures with touches > 0: {(df.touches > 0).sum()}")