"""Build README visuals from the shipped demo; never fit, score or modify a run.

From the repository root:
  python docs/figures/readme/generate.py
  python docs/figures/readme/generate.py --check-run outputs/s26_gaussian_ewma
Requires the existing optional matplotlib dependency (tailrisk[plots]).
"""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import re
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tailrisk-readme-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BG, INK, MUTED = '#080d13', '#edf1ed', '#aab6c0'
MINT, BLUE, AMBER, LINE = '#85f5c6', '#85baff', '#ffc58a', '#35434e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                     'svg.fonttype': 'path', 'svg.hashsalt': 'tailrisk-readme-v1',
                     'text.color': INK, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED})

def canvas(height):
    fig = plt.figure(figsize=(6.4, height / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0,640), ylim=(height,0))
    ax.axis('off')
    return fig, ax

def text(ax, x, y, label, size=14, color=INK, weight='normal', **kw):
    return ax.text(x,y,label,fontsize=size,color=color,weight=weight,va='top',**kw)

def box(ax,x,y,w,h):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',
                              facecolor='#101923',edgecolor=LINE,linewidth=1))

def arrow(ax,x1,y1,x2,y2):
    ax.annotate('', xy=(x2,y2),xytext=(x1,y1),
                arrowprops={'arrowstyle':'->','color':MINT,'lw':1.5})

def save(fig, name, title, description):
    fig.savefig(OUT / (name+'.svg'), facecolor=BG, metadata={'Date':None,'Title':title,'Description':description})
    fig.savefig(OUT / (name+'.png'), dpi=200, facecolor=BG, metadata={'Title':title,'Description':description})
    plt.close(fig)


def workflow():
    fig, ax = canvas(715)
    text(ax,28,22,'Compare your risk model\nagainst standard baselines',22)
    box(ax,28,101,584,88)
    text(ax,44,119,'DATA',10,MINT)
    text(ax,119,116,'Synthetic benchmark OR local daily-return CSV',13)
    text(ax,44,157,'MODEL',10,MINT)
    text(ax,119,153,'Built-in models OR your model + adapter',13)
    text(ax,28,208,'Run from Python’s command line or ask WorkSwarm',14)
    stages = [
        ('Run','Fit on training data. Forecast next-session\nVaR and ES using past observations.'),
        ('Verify','Check saved predictions, data sources\nand reproducible model outputs.'),
        ('Evaluate','Score every model on the same dates.'),
        ('Report','See forecast scores, extreme\nlosses and failed checks.'),
    ]
    for i,(name,description) in enumerate(stages):
        y=260+i*112
        arrow(ax,320,y-26,320,y-4)
        box(ax,28,y,584,84)
        text(ax,44,y+13,f'0{i+1}',10,MINT)
        text(ax,75,y+30,name,17)
        text(ax,205,y+(30 if i==2 else 20),description,13,MUTED,linespacing=1.2)
    # A schematic report icon, not an additional numerical result plot.
    box(ax,512,606,82,63)
    ax.plot([522,583],[617,617],color=MUTED,lw=1)
    ax.plot([522,547],[625,625],color=LINE,lw=1)
    ax.plot([522,522,583],[632,659,659],color=LINE,lw=1)
    ax.plot([526,535,544,553,562,571,581],
            [649,644,650,637,642,632,635],color=MINT,lw=1.3)
    ax.plot([526,581],[654,646],color=BLUE,lw=1,ls='--')
    save(fig,'workflow','Compare your risk model against standard baselines',
         'Data and models enter a shared CLI: Run, Verify, Evaluate, Report. WorkSwarm is optional. Report thumbnail is schematic.')


def evidence(check_run):
    page = ROOT / 'demo/index.html'
    data = json.loads(re.search(r'<script id="evidence" type="application/json">(.*?)</script>',page.read_text(),re.S)[1])
    models=['example_ewma_gaussian','garch_student_t','historical_simulation']
    series=data['series']
    origins=[int(r[0]) for r in series[models[0]]]
    assert len(origins)==len(set(origins))==499
    assert origins==sorted(origins) and origins[0]==1000 and origins[-1]==1498
    metrics={r['model_id']:r for r in data['metrics'] if float(r['alpha'])==.025 and r['split']=='validation'}
    for m in models:
        assert [int(r[0]) for r in series[m]]==origins
        assert all(r[1]==s[1] for r,s in zip(series[m],series[models[0]]))
        assert all(all(math.isfinite(v) for v in row) and row[3]<=row[2]<0 for row in series[m])
        assert int(metrics[m]['n'])==499
        assert sum(r[1]<=r[2] for r in series[m])==int(metrics[m]['exceedance_count'])
    assert int(metrics[models[0]]['exceedance_count'])==12
    assert int(metrics[models[0]]['n_available'])==1437
    if check_run:
        run=Path(check_run)
        with (run/'forecasts.csv').open() as f:
            rows=[r for r in csv.DictReader(f) if r['series_id']=='example_garch_gaussian_validation' and float(r['alpha'])==.025]
        selected={m:{int(r['origin_date']):r for r in rows if r['model_id']==m and r['status']=='ok' and all(math.isfinite(float(r[k])) for k in ('var','es','realized_return')) and float(r['es'])<=float(r['var'])<0} for m in models}
        assert sorted(set.intersection(*(set(selected[m]) for m in models)))==origins
        for m in models:
            for o,y,v,e in series[m]:
                r=selected[m][int(o)]
                assert [float(r[k])*100 for k in ('realized_return','var','es')]==[y,v,e]
        with (run/'metrics.csv').open() as f:
            local={r['model_id']:r for r in csv.DictReader(f) if r['split']=='validation' and float(r['alpha'])==.025}
        assert local==metrics
        assert len(selected[models[0]])==1437
        assert sum(float(r['realized_return'])<=float(r['var']) for r in selected[models[0]].values())==43
        print('Local CSV cross-check: all 1,497 matched model/origin rows and primary metrics agree exactly with the demo; EWMA available 43/1,437.')
    return series[models[0]], metrics


def example(rows, metrics):
    fig,ax=canvas(750)
    text(ax,28,22,'Example: test an EWMA risk model',21)
    text(ax,28,65,'EWMA estimates volatility by giving more weight\nto recent returns.',14,MUTED,linespacing=1.4)
    box(ax,28,125,584,67)
    text(ax,44,137,'Your Python model → adapter → comparison with',13,MINT)
    text(ax,44,163,'historical simulation and GARCH-t',13)
    text(ax,28,216,'Synthetic data · 499 shared validation observations · α = 0.025',12)
    text(ax,28,249,'Signed decimal log return',12,MUTED)
    plot=fig.add_axes([.135,190/750,.82,245/750],facecolor=BG)
    x=[r[0] for r in rows]
    y,v,e=([r[j]/100 for r in rows] for j in (1,2,3))
    plot.plot(x,y,color=MUTED,lw=.8,label='Return',alpha=.85)
    plot.plot(x,v,color=BLUE,lw=1.6,label='VaR')
    plot.plot(x,e,color=MINT,lw=1.6,ls='--',label='ES')
    ids=[i for i in range(len(x)) if y[i]<=v[i]]
    plot.scatter([x[i] for i in ids],[y[i] for i in ids],s=28,color=AMBER,marker='o',zorder=4)
    plot.set_xlim(1000,1498)
    plot.set_xticks([1000,1200,1400])
    plot.set_yticks([-.04,-.02,0,.02])
    plot.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plot.tick_params(labelsize=11,length=0,pad=7)
    plot.grid(axis='y',color=LINE,lw=.6,alpha=.7)
    for spine in plot.spines.values(): spine.set_visible(False)
    plot.set_xlabel('Forecast origin (validation index)',fontsize=12,labelpad=9)
    leg=plot.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False,
                    fontsize=12,handlelength=2,columnspacing=2)
    for t in leg.get_texts(): t.set_color(INK)
    ax.scatter([34],[641],s=28,color=AMBER)
    text(ax,48,631,'Orange dots: returns at or below the predicted VaR threshold.',12,MUTED)
    count=len(ids)
    n=len(rows)
    alpha=float(metrics['example_ewma_gaussian']['alpha'])
    text(ax,28,674,f'{count} of {n} returns fell at or below VaR ({count/n:.1%}),',16,AMBER)
    text(ax,28,708,f'compared with the nominal {alpha:.1%}.',16)
    save(fig,'ewma-use-case','Example: test an EWMA risk model',
         'Synthetic Gaussian GARCH validation data, alpha 0.025, signed decimal log returns. EWMA: 12 of 499 shared observations at or below VaR (2.4%), compared with nominal 2.5%.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-run',type=Path,help='Cross-check embedded data against existing saved CSVs; never runs a model.')
    args=parser.parse_args()
    rows,metrics=evidence(args.check_run)
    workflow()
    example(rows,metrics)
    print('Wrote workflow and EWMA figures (SVG + 2× PNG); checked 499 origins and 12 EWMA exceedances.')
