"""Quick utility: extract TUB suite summaries from logs/tub_clf_*.json."""
import json, glob, os

rows = []
for f in sorted(glob.glob('logs/tub_clf_*.json')):
    d = json.load(open(f))
    s = d['summary']
    b = s['acc_base_halfB']['mean']
    p = s['acc_proto_oracle']['mean']
    c = s['acc_tub_clf']['mean']
    t = s['acc_tcr_oracle_init']['mean']
    rows.append((d['dataset'], d['n_shot'], b, p, c, t))

# Group by shot
print(f"{'Dataset':10s} {'Shot':>4s} {'base':>7s} {'protoO':>7s} {'ΔP':>6s} {'TUBclf':>7s} {'ΔC':>6s} {'TIMinit':>8s} {'ΔA':>6s}")
print("-" * 80)
for ds, ns, b, p, c, t in rows:
    print(f"{ds:10s} {ns:4d} {b:7.2f} {p:7.2f} {p-b:+6.2f} {c:7.2f} {c-b:+6.2f} {t:8.2f} {t-b:+6.2f}")
