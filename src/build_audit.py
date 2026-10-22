"""Create two independent, blind, offline annotation workspaces from real audit cases."""
import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = '''# Anleitung für die unabhängige Annotation

Aaron annotiert 160 Fälle, Chris prüft 40 dieser Fälle unabhängig. Bitte zunächst
ohne Abstimmung annotieren. Die Dateien zeigen weder Modellvorhersagen noch
Benchmarklabels oder die Auswahlgruppe. Die gemeinsame Besprechung folgt erst
nach Abgabe beider unabhängiger Dateien.

Entscheidung: Match = dieselbe reale Entität; Non-match = unterschiedliche Entitäten;
Unklar = die vorhandenen Records reichen nicht für eine belastbare Entscheidung.
Ähnliche Produkte, Zubehör und unterschiedliche Varianten sind nicht automatisch
identisch. Fehlende Werte allein begründen weder Match noch Non-match. Bei Büchern
insbesondere Edition, Band und Identifier berücksichtigen. Begründungen müssen auf
konkrete Recordangaben verweisen. Wenn das Benchmarkverständnis selbst unklar ist,
Unklar markieren und den Definitionskonflikt notieren. Keine Modellantwort zur
Entscheidung heranziehen; externe Quellen nur mit Link und klarer Kennzeichnung.

Pro Fall eine Entscheidung, eine Hauptkategorie und kurze Begründung eintragen.
Kategorien: Produktvariante, Zubehör/Bundle, Identifier-Konflikt, Fehlende Angaben,
Preis-Konflikt, Abkürzung/Schreibweise, Sprache, Bibliografie, Schema, Sonstiges.

Die HTML-Datei lokal im Browser öffnen. Sie benötigt keinen Server und versendet
keine Daten. Browser-Speicherung ist nur eine Komfortfunktion: regelmäßig über
„Annotationen exportieren“ eine JSON-Datei herunterladen. Diese Datei lässt sich
über „Fortsetzen“ wieder importieren. Nicht nur den Browser schließen, ohne vorher
zu exportieren. Die HTML-Dateien selbst verändern sich beim Annotieren nicht.

Bitte Aaron- und Chris-Dateien getrennt zurückgeben. Danach werden fehlende
Annotationen, Übereinstimmung und Konflikte geprüft. Konflikte werden gemeinsam
adjudiziert, wobei beide Originalentscheidungen erhalten bleiben. Die zufällige
und die diagnostisch angereicherte Stichprobe werden getrennt ausgewertet; eine
gemeinsame Fehlerrate wäre irreführend. Leere Felder sind keine geprüften Fälle.
'''

TEMPLATE = r'''<!doctype html><html lang="de"><meta charset="utf-8">
<title>Blinde EM Annotation – __REVIEWER__</title>
<style>body{font:16px system-ui;margin:28px auto;max-width:1250px;padding:0 24px;color:#172338;background:#f5f7fb}header{display:flex;justify-content:space-between;align-items:center}button,select,input,textarea{font:inherit;padding:10px;border:1px solid #abb6c8;border-radius:6px}button{cursor:pointer;background:white}h1{font-size:24px}.records{display:grid;grid-template-columns:1fr 1fr;gap:16px}.record{background:white;border:1px solid #d8dfeb;padding:18px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:52vh;overflow:auto}textarea{width:95%;height:85px;background:white}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.muted{color:#4e5d72}label{display:block;margin:10px 0}#warning{color:#a32715}</style>
<header><h1>Entity Matching · __REVIEWER__</h1><button id="export">Annotationen exportieren</button></header>
<p class="muted">Unabhängig annotieren. Match = identische reale Entität, Non-match = verschiedene Entitäten, Unklar = unzureichende Evidenz. Keine Modellantworten konsultieren. Vor dem Schließen exportieren.</p>
<p id="warning"></p><div class="controls"><button id="prev">← Zurück</button><select id="jump" aria-label="Fall wählen"></select><button id="next">Weiter →</button><span id="progress"></span></div>
<div class="records"><section><h2>Record A</h2><div class="record" id="left"></div></section><section><h2>Record B</h2><div class="record" id="right"></div></section></div>
<div class="controls"><label>Entscheidung <select id="label"><option value="">Bitte wählen</option><option>Match</option><option>Non-match</option><option>Unklar</option></select></label><label>Hauptkategorie <select id="category"><option value="">Bitte wählen</option><option>Produktvariante</option><option>Zubehör/Bundle</option><option>Identifier-Konflikt</option><option>Fehlende Angaben</option><option>Preis-Konflikt</option><option>Abkürzung/Schreibweise</option><option>Sprache</option><option>Bibliografie</option><option>Schema</option><option>Sonstiges</option></select></label></div>
<label for="note">Begründung anhand der Records</label><textarea id="note"></textarea>
<p><label>Fortsetzen: exportierte JSON-Datei laden <input type="file" id="import" accept="application/json,.json"></label></p>
<script type="application/json" id="cases">__CASES__</script><script>
const cases=JSON.parse(document.getElementById('cases').textContent), reviewer=__REVIEWER_JSON__, key='decision-em-audit-v1-'+reviewer;
const el=id=>document.getElementById(id);let index=0,answers={};
try{answers=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){el('warning').textContent='Browser-Speicherung nicht verfügbar. Regelmäßig exportieren.'}
cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' · '+c.benchmark+' · '+c.audit_id;el('jump').append(o)});
function save(){const c=cases[index];answers[c.audit_id]={human_label:el('label').value,category:el('category').value,evidence_note:el('note').value};try{localStorage.setItem(key,JSON.stringify(answers))}catch(e){el('warning').textContent='Speicherung fehlgeschlagen. Jetzt exportieren.'}count()}
function count(){el('progress').textContent=cases.filter(c=>answers[c.audit_id]?.human_label&&answers[c.audit_id]?.category&&answers[c.audit_id]?.evidence_note?.trim()).length+' / '+cases.length+' vollständig'}
function show(){const c=cases[index],a=answers[c.audit_id]||{};el('jump').value=index;['left','right'].forEach(s=>{let r=JSON.parse(c['record_'+s]);el(s).textContent=Object.entries(r).map(([k,v])=>k+': '+(typeof v==='string'?v:JSON.stringify(v))).join('\n\n')});el('label').value=a.human_label||'';el('category').value=a.category||'';el('note').value=a.evidence_note||'';el('prev').disabled=index===0;el('next').disabled=index===cases.length-1;count()}
['label','category','note'].forEach(x=>el(x).addEventListener('input',save));el('prev').onclick=()=>{save();index--;show()};el('next').onclick=()=>{save();index++;show()};el('jump').onchange=()=>{save();index=Number(el('jump').value);show()};
el('export').onclick=()=>{save();const value={schema_version:1,reviewer,exported_at:new Date().toISOString(),annotations:cases.map(c=>({audit_id:c.audit_id,benchmark:c.benchmark,...(answers[c.audit_id]||{})}))};const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='audit_'+reviewer+'_'+new Date().toISOString().slice(0,10)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
el('import').onchange=async event=>{try{const data=JSON.parse(await event.target.files[0].text());if(data.reviewer!==reviewer||data.schema_version!==1)throw Error('Falsche Person oder Dateiversion');const ids=new Set(cases.map(c=>c.audit_id));if(!Array.isArray(data.annotations)||data.annotations.some(a=>!ids.has(a.audit_id)))throw Error('Unbekannte Fälle');if(!confirm('Aktuelle Einträge durch die importierte Datei ersetzen? Vorher gegebenenfalls exportieren.'))return;answers=Object.fromEntries(data.annotations.map(a=>[a.audit_id,a]));show();save()}catch(e){el('warning').textContent='Import fehlgeschlagen: '+e.message}};show();
</script></html>'''


def render(rows, reviewer):
    # Only explicitly whitelisted blind columns; no gold/model key is embedded.
    blind = [{k: r[k] for k in ['audit_id', 'benchmark', 'record_left', 'record_right']} for r in rows]
    data = json.dumps(blind, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return TEMPLATE.replace('__REVIEWER__', reviewer).replace('__REVIEWER_JSON__', json.dumps(reviewer)).replace('__CASES__', data)


def main():
    source = PROJECT/'artifacts/results'
    out = PROJECT/'artifacts/audit_for_reviewers'
    out.mkdir(parents=True, exist_ok=True)
    for reviewer, filename, n in [('Aaron', 'audit_blind_160.csv', 160), ('Chris', 'audit_second_reviewer_40.csv', 40)]:
        with (source/filename).open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n and len({r['audit_id'] for r in rows}) == n
        html = render(rows, reviewer)
        dest = out/('Annotation_'+reviewer+'.html')
        if dest.exists():
            assert dest.read_text() == html, 'Do not overwrite an existing annotation packet with different cases'
        else:
            dest.write_text(html)
    (out/'ANLEITUNG.md').write_text(INSTRUCTIONS)
    print('BLIND AUDIT PACKETS READY: Aaron 160, Chris 40; no human labels filled', flush=True)


if __name__ == '__main__':
    main()
