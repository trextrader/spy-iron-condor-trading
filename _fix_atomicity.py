"""Fix broken atomicity_fail print and add gate summary lines in right place."""
with open('kaggle/condor_brain_backtest_v45.py', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix the broken atomicity_fail block
fixed = 0
i = 0
while i < len(lines):
    # Find the line that starts the broken print (missing closing paren)
    if ("run_backtest._entry_dbg['atomicity_fail']" in lines[i]
            and lines[i].rstrip().endswith('"')
            and 'ATOMICITY FAIL' in lines[i]):
        # Check next two lines are the wrongly inserted P4 lines
        if (i + 2 < len(lines)
                and 'expiry_past_sim' in lines[i + 1]
                and 'max_pos_block' in lines[i + 2]):
            # Remove lines i+1 and i+2 (the wrongly inserted P4 lines)
            del lines[i + 1]
            del lines[i + 1]
            fixed += 1
            print(f"Fixed: removed 2 P4 lines after atomicity_fail at line {i+1}")
    i += 1

# Find credit_fail in gate summary and insert expiry/maxpos counts after it
# The summary credit_fail line looks like:
#   print(f"  {'credit_fail':<28s}: {d['credit_fail']:>8,}")
inserted = 0
i = 0
while i < len(lines):
    if ("d['credit_fail']" in lines[i]
            and 'print' in lines[i]
            and 'expiry_past_sim' not in lines[i]):
        # Check it's the summary line (not inside verbose_sim block)
        # and that expiry_past_sim not already after it
        if i + 1 < len(lines) and 'expiry_past_sim' not in lines[i + 1]:
            ind = len(lines[i]) - len(lines[i].lstrip())
            p = ' ' * ind
            new_lines = [
                p + 'print(f"  expiry_past_sim            : {d.get(\'expiry_past_sim\', 0):>8,}  ({d.get(\'expiry_past_sim\', 0)/bars*100:.1f}%)")\n',
                p + 'print(f"  max_pos_block              : {d.get(\'max_pos_block\', 0):>8,}  ({d.get(\'max_pos_block\', 0)/bars*100:.1f}%)")\n',
            ]
            lines.insert(i + 1, new_lines[1])
            lines.insert(i + 1, new_lines[0])
            inserted += 1
            print(f"Inserted gate summary lines after line {i+1}")
            break
    i += 1

with open('kaggle/condor_brain_backtest_v45.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Done. fixed={fixed} inserted={inserted}")
