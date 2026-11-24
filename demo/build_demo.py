"""Refresh embedded presentation data from saved artifacts; never fits or scores models.
Run from the repository root: python3 demo/build_demo.py
"""
from pathlib import Path
import base64, csv, json, re
import math
from statistics import NormalDist

def integrate(f, lo, hi, n=4000):
    h=(hi-lo)/n
    return h/3*(f(lo)+f(hi)+sum((4 if i%2 else 2)*f(lo+i*h) for i in range(1,n)))

class Law:
    def __init__(self, nu):
        self.nu=nu
        self.c=1/math.sqrt(2*math.pi) if nu is None else math.gamma((nu+1)/2)/(math.sqrt(nu*math.pi)*math.gamma(nu/2))
    def pdf(self, x):
        return self.c*math.exp(-x*x/2) if self.nu is None else self.c*(1+x*x/self.nu)**(-(self.nu+1)/2)
    def cdf(self, x):
        return NormalDist().cdf(x) if self.nu is None else .5-integrate(self.pdf,0,-x)
    def ppf(self, alpha):
        if self.nu is None: return NormalDist().inv_cdf(alpha)
        lo,hi=-30,0
        for _ in range(55):
            mid=(lo+hi)/2
            if self.cdf(mid)<alpha: lo=mid
            else: hi=mid
        return (lo+hi)/2
    def tail_mean_integral(self, q):
        # x=q-(u/(1-u))**2 maps the full negative tail to [0,1].
        def f(u):
            if u==1:
                return -2*self.c*self.nu**((self.nu+1)/2) if self.nu==1.5 else 0
            x=q-(u/(1-u))**2
            return x*self.pdf(x)*2*u/(1-u)**3
        return integrate(f,0,1,20000)
ROOT = Path(__file__).resolve().parents[1]
run = ROOT / 'outputs/s26_gaussian_ewma'
models = ['garch_student_t','example_ewma_gaussian','historical_simulation']
xs = [-12+i*17/1000 for i in range(1001)]
distributions = []
for name, nu in [('Normal',None),('Student-t',5),('Extreme tail',1.5)]:
    law = Law(nu)
    q = law.ppf(.025)
    scale = -2/q
    es = -scale*law.pdf(q)/.025 if nu is None else -scale*(nu+q*q)/(nu-1)*law.pdf(q)/.025
    # Independent numerical expectation check of the displayed analytic ES.
    numeric = law.tail_mean_integral(q)*scale/.025
    assert abs(es-numeric)<1e-7 and abs(law.cdf(-2/scale)-.025)<1e-10
    distributions.append(dict(name=name,nu=nu,xi=0 if nu is None else 1/nu,es=es,pdf=[round(law.pdf(x/scale)/scale,9) for x in xs]))
with (run/'metrics.csv').open() as f:
    metrics = [r for r in csv.DictReader(f) if r['split']=='validation']
with (run/'forecasts.csv').open() as f:
    forecasts = [r for r in csv.DictReader(f) if r['series_id']=='example_garch_gaussian_validation']
primary = [r for r in metrics if float(r['alpha'])==.025]
assert len(primary)==3 and all(int(r['n'])==499 for r in primary)
rows = {m:[r for r in forecasts if r['model_id']==m and float(r['alpha'])==.025] for m in models}
common = set.intersection(*(set(r['origin_date'] for r in rows[m] if r['status']=='ok') for m in models))
assert len(common)==499
series = {m:[[int(r['origin_date']), float(r['realized_return'])*100,float(r['var'])*100,float(r['es'])*100] for r in rows[m] if r['origin_date'] in common] for m in models}
last = max(int(x) for x in common)
curves = {m: sorted([[float(r['alpha']),float(r['var'])*100,float(r['es'])*100] for r in forecasts if r['model_id']==m and int(r['origin_date'])==last],key=lambda x:x[0]) for m in models}
assert all(len(c)==6 for c in curves.values())
figdir = ROOT/'evidence/stage2/s26/figures'
figs = ['realized_var_es_primary_alpha.png','coverage_rate_vs_alpha.png','curves_last_common_origin.png']
data = dict(distributions=distributions,metrics=metrics,series=series,curves=curves,figures=['data:image/png;base64,'+base64.b64encode((figdir/f).read_bytes()).decode() for f in figs])
page = ROOT/'demo/index.html'
html = page.read_text()
html,n = re.subn(r'(<script id="evidence" type="application/json">).*?(</script>)',lambda m:m[1]+json.dumps(data,separators=(',',':'))+m[2],html,flags=re.S)
assert n==1
page.write_text(html)
print('Embedded 3 analytic distributions, 499 matched origins per model, 18 risk-curve points, and 3 original audit plots.')
print('ES (%):', [round(d['es'],6) for d in distributions])
