import json, os, glob, re, collections

def load(f): return json.load(open(f, encoding='utf-8', errors='replace'))

# ---- broken-turn detection (from salvage_rejected.py) ----
REPETITION_RE = re.compile(r"\b(\w{4,})\b(?:\s+\1\b){4,}", re.IGNORECASE)
JOB_PHRASES = ["Human Resources","hiring","job posting","engineer -","manager -","we are looking","candidate","years of experience","salary range"]
CS_PHRASES = ["API","SDK","JSON","YAML","deploy","kubernetes","docker","tensorflow","pytorch","git push","pull request","merge conflict","scikit-learn","k-fold","cross-validation"]
LANG_LIST_RE = re.compile(r"\b(Chinese|Korean|Japanese|Hindi|Tamil|Telugu|Bengali|Marathi|Spanish|French|German|Italian|Portuguese|Russian|Arabic|Hebrew|Vietnamese|Thai|Burmese|Lao|Khmer|Indonesian|Malay|Tagalog|Filipino|Turkish|Persian|Urdu|Punjabi|Gujarati|Kannada|Malayalam|Odia|Assamese|Nepali|Sinhala)\b", re.I)
LEAKS = ["iCALL","Vandrevala","follow ups occur automatically","Follow Up Phrases May Include","Based solely On User Input","trigger templates","instruction documents","emotionally financially physically mentally psychologically"]

def broken_flags(text):
    if len(text) < 600: return []
    f=[]
    if REPETITION_RE.search(text): f.append('repetition')
    if any(p.lower() in text.lower() for p in JOB_PHRASES): f.append('job')
    if sum(1 for p in CS_PHRASES if p in text)>=2: f.append('cs')
    if len(set(LANG_LIST_RE.findall(text)))>=4: f.append('langs')
    if re.search(r"[A-Za-z ]{300,}", text): f.append('unbroken')
    if any(k in text for k in LEAKS): f.append('leak')
    return f

def truncate_to_clean(msgs):
    last_good=-1
    for i,m in enumerate(msgs):
        if m['role']!='assistant': continue
        if not broken_flags(m.get('content','') or ''): last_good=i
    if last_good<0: return msgs[:2]
    return msgs[:last_good+1]

def audit(rows):
    out=[]
    for r in rows:
        msgs=r['messages']
        bad=[(i,m.get('content','') or '') for i,m in enumerate(msgs) if m['role']=='assistant' and broken_flags(m.get('content','') or '')]
        clean=truncate_to_clean(msgs)
        out.append(dict(id=r['conversation_id'],cat=r['category'],score=r['scores']['total'],
                        n_turns=len(msgs),bad_turns=bad,n_removed=len(msgs)-len(clean),
                        clean_msgs=clean))
    return out

flagged=[load(f) for f in glob.glob('formatted/flagged/*.json')]
fixes=[load(f) for f in glob.glob('formatted/flagged_fixes/*.json')]
fa=audit(flagged); fxa=audit(fixes)

fmap={a['id']:a for a in fa}; fxmap={a['id']:a for a in fxa}
allids=sorted(set(fmap)|set(fxmap))
print(f"{'id':42s} {'src':14s} {'scr':>3s} {'turns':>5s} {'bad':>3s} {'rmv':>3s} {'clean':>5s}  verdict")
rows=[]
for i in allids:
    f=fmap.get(i); x=fxmap.get(i)
    src='both' if f and x else ('flagged' if f else 'fixes')
    a=x or f
    verdict='CLEAN' if not a['bad_turns'] else ('TRUNC' if a['n_removed'] else 'BROKEN')
    rows.append((i,src,a['score'],a['n_turns'],len(a['bad_turns']),a['n_removed'],len(a['clean_msgs']),verdict))
    print(f"{i:42s} {src:14s} {a['score']:3d} {a['n_turns']:5d} {len(a['bad_turns']):3d} {a['n_removed']:3d} {len(a['clean_msgs']):5d}  {verdict}")
vc=collections.Counter(r[7] for r in rows)
print('\nVERDICT:',dict(vc))
print('CLEAN:',[r[0] for r in rows if r[7]=='CLEAN'])
print('TRUNC:',[r[0] for r in rows if r[7]=='TRUNC'])
print('BROKEN:',[r[0] for r in rows if r[7]=='BROKEN'])
json.dump(rows, open('/tmp/audit.json','w'), indent=1)