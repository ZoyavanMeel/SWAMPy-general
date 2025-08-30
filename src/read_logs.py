import os
import subprocess

log_path = "logs/simulate/nCOV19_clinical_simulated"
logs = sorted(os.listdir(log_path))

dropped = []
for log in logs:
    grep = subprocess.run(
        ["grep", "find a match for the primer", os.path.join(log_path, log)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out = grep.stdout.decode().split("\n")
    if len(out) > 20:
        print(log)
    dropped.append(len(out))
print(sum(dropped) / len(dropped))