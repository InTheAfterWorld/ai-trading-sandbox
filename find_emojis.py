"""Find emojis that were in committed version but removed in working directory."""
import subprocess
import re
import sys

# Use UTF-8 for stdout to handle emoji output
sys.stdout.reconfigure(encoding='utf-8')

# Broader emoji range
emoji_re = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F02F"
    "\U0001F0A0-\U0001F0FF"
    "\U00002702-\U000027B0"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U00002300-\U000023FF"
    "\U0001F900-\U0001F9FF"
    "\U00002705\U0000270A\U0001F44A\U0001F44B\U0001F44D\U0001F44E\U0001F44F"
    "\U0001F680\U00002764\U0001F499\U00002B50\U00002714\U00002716\U0000274C\U0000274E"
    "\U00002122\U00002192\U00002193\U000021C5\U000027A1\U00002139\U00002702"
    "\U0001F4A1\U0001F4C4\U0001F527\U0001F4BB\U0001F52C\U0001F4CA\U0001F4C8"
    "\U0001F4C9\U000026A0\U0001F6A8\U0001F4A4\U00002317\U00002318\U00002325"
    "]"
)

# Get modified files
result = subprocess.run(['git', 'diff', '--name-only'], capture_output=True, text=True, cwd='.', encoding='utf-8')
files = [f for f in result.stdout.strip().split('\n') if f and (f.endswith('.py') or f.endswith('.html'))]

print("Modified Python/HTML files:")
for f in files:
    print(f"  {f}")
print()

for f in files:
    try:
        comm_result = subprocess.run(['git', 'show', f'HEAD:{f}'], capture_output=True, cwd='.', encoding='utf-8', errors='replace')
        comm = comm_result.stdout
        work = open(f, encoding='utf-8', errors='replace').read()

        comm_lines_with_emoji = [(i+1, line.strip()[:140]) for i, line in enumerate(comm.split('\n')) if emoji_re.search(line)]
        work_lines_with_emoji = [(i+1, line.strip()[:140]) for i, line in enumerate(work.split('\n')) if emoji_re.search(line)]

        comm_emojis = set(emoji_re.findall(comm))
        work_emojis = set(emoji_re.findall(work))

        removed = comm_emojis - work_emojis

        if removed:
            print(f"=== {f} ===")
            print(f"  Removed emojis: {removed}")
            print(f"  Lines with emoji in COMMITTED version:")
            for ln, text in comm_lines_with_emoji:
                print(f"    line {ln}: {text}")
            print(f"  Lines with emoji in WORKING version:")
            for ln, text in work_lines_with_emoji:
                print(f"    line {ln}: {text}")
            print()
    except Exception as e:
        print(f"  Error reading {f}: {e}")
