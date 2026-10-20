#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pricing d'options européennes : Binomial (CRR), Trinomial (Boyle) vs Black-Scholes
Etude de la convergence et du temps de calcul, avec interface graphique simple.

Dépendances : numpy, scipy, matplotlib (Tkinter est fourni avec Python)
    pip install numpy scipy matplotlib
Lancement :
    python pricing_options.py
"""

import csv
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# =====================================================================
#  1. MOTEUR DE CALCUL
# =====================================================================

def black_scholes(S, K, T, r, sigma, q=0.0, option="call"):
    """Formule fermée de Black-Scholes-Merton (option européenne)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def _payoff(S, K, option):
    return np.maximum(S - K, 0.0) if option == "call" else np.maximum(K - S, 0.0)


def binomial_crr(S, K, T, r, sigma, N, q=0.0, option="call"):
    """Arbre binomial de Cox-Ross-Rubinstein (induction arrière vectorisée)."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    j = np.arange(N + 1)
    V = _payoff(S * u ** (2 * j - N), K, option)          # valeurs à l'échéance

    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])       # espérance actualisée
    return float(V[0])


def trinomial_boyle(S, K, T, r, sigma, N, q=0.0, option="call"):
    """Arbre trinomial de Boyle (u = exp(sigma*sqrt(2dt)), d = 1/u, m = 1)."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(2.0 * dt))
    a = np.exp((r - q) * dt / 2.0)
    b = np.exp(sigma * np.sqrt(dt / 2.0))
    pu = ((a - 1.0 / b) / (b - 1.0 / b)) ** 2
    pd = ((b - a) / (b - 1.0 / b)) ** 2
    pm = 1.0 - pu - pd
    disc = np.exp(-r * dt)

    j = np.arange(-N, N + 1)
    V = _payoff(S * u ** j, K, option)

    for i in range(N - 1, -1, -1):
        V = disc * (pu * V[2:] + pm * V[1:-1] + pd * V[:-2])
    return float(V[0])


def chrono(func, *args, min_time=0.02, **kwargs):
    """Mesure le temps moyen d'exécution (répète si l'appel est très court)."""
    n_rep, total, val = 0, 0.0, None
    while total < min_time and n_rep < 200:
        t0 = time.perf_counter()
        val = func(*args, **kwargs)
        total += time.perf_counter() - t0
        n_rep += 1
    return val, total / n_rep



def run_study(p, option, Ns, progress=None):
    """Étude de convergence et de temps. Retourne un dictionnaire de tableaux.

    p : dict avec S, K, T, r, sigma, q ;  Ns : liste de nombres de pas.
    Contient aussi CRR(2N) pour vérifier numériquement Tri(N) = CRR(2N).
    """
    args = (p["S"], p["K"], p["T"], p["r"], p["sigma"])
    ref = black_scholes(*args, p["q"], option)
    out = {k: [] for k in ("N", "crr", "tri", "crr2n", "t_crr", "t_tri")}
    for k, n in enumerate(Ns):
        n = int(n)
        v1, t1 = chrono(binomial_crr, *args, n, p["q"], option)
        v2, t2 = chrono(trinomial_boyle, *args, n, p["q"], option)
        v3 = binomial_crr(*args, 2 * n, p["q"], option)
        for key, val in zip(out, (n, v1, v2, v3, t1, t2)):
            out[key].append(val)
        if progress:
            progress(k + 1, len(Ns), n)
    d = {k: np.array(v) for k, v in out.items()}
    d["ref"] = ref
    d["err_crr"] = np.abs(d["crr"] - ref)
    d["err_tri"] = np.abs(d["tri"] - ref)
    d["ecart_eq"] = np.abs(d["tri"] - d["crr2n"])
    d["params"], d["option"] = dict(p), option
    return d


def _clean(*axes):
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=.25, which="both")
        ax.tick_params(labelsize=9)


def draw_convergence(fig, d):
    """Prix en fonction de N et erreur log-log."""
    Ns, ref = d["N"], d["ref"]
    eb, et = d["err_crr"] + 1e-16, d["err_tri"] + 1e-16
    fig.clear()
    a1, a2 = fig.subplots(1, 2)
    a1.plot(Ns, d["crr"], "o-", ms=3, color=C_BIN, label="Binomial CRR")
    a1.plot(Ns, d["tri"], "s-", ms=3, color=C_TRI, label="Trinomial")
    a1.axhline(ref, color=C_BS, ls="--", label="Black-Scholes")
    a1.set_title("Prix en fonction de N"); a1.set_xlabel("Nombre de pas N")
    a1.set_ylabel("Prix"); a1.legend(); a1.grid(alpha=.3)
    a2.loglog(Ns, eb, "o-", ms=3, color=C_BIN, label="Binomial CRR")
    a2.loglog(Ns, et, "s-", ms=3, color=C_TRI, label="Trinomial")
    a2.loglog(Ns, eb[0] * Ns[0] / Ns, ":", color="grey", label="pente −1 (ordre 1)")
    a2.set_title("Erreur absolue (échelle log-log)")
    a2.set_xlabel("N"); a2.set_ylabel("|prix − référence|")
    a2.legend(); _clean(a1, a2)
    fig.tight_layout()


def draw_time(fig, d):
    """Temps de calcul et courbe précision-coût."""
    Ns = d["N"]
    tb, tt = d["t_crr"] * 1000, d["t_tri"] * 1000
    eb, et = d["err_crr"] + 1e-16, d["err_tri"] + 1e-16
    fig.clear()
    b1, b2 = fig.subplots(1, 2)
    b1.loglog(Ns, tb, "o-", ms=3, color=C_BIN, label="Binomial CRR")
    b1.loglog(Ns, tt, "s-", ms=3, color=C_TRI, label="Trinomial")
    b1.set_title("Temps de calcul (log-log)")
    b1.set_xlabel("N"); b1.set_ylabel("Temps (ms)")
    b1.legend(); b1.grid(alpha=.3, which="both")
    b2.loglog(tb, eb, "o-", ms=3, color=C_BIN, label="Binomial CRR")
    b2.loglog(tt, et, "s-", ms=3, color=C_TRI, label="Trinomial")
    b2.set_title("Précision vs coût (efficacité)")
    b2.set_xlabel("Temps (ms)"); b2.set_ylabel("Erreur absolue")
    b2.legend(); _clean(b1, b2)
    fig.tight_layout()


def _fr(x, nd=6):
    """Nombre au format LaTeX français (virgule décimale)."""
    return f"{x:.{nd}f}".replace(".", "{,}")


def _sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"${m.replace('.', '{,}')}\\times10^{{{int(e)}}}$"


def pente_loglog(Ns, err, nmin=50):
    """Pente de la régression log-log de l'erreur sur N (N >= nmin)."""
    m = Ns >= nmin
    return float(np.polyfit(np.log(Ns[m]), np.log(err[m] + 1e-300), 1)[0])


def export_all(folder, d):
    """Écrit dans `folder` : CSV, tableaux LaTeX et figures (PDF + PNG)."""
    os.makedirs(folder, exist_ok=True)
    p, ref = d["params"], d["ref"]
    fichiers = []

    # --- CSV complet
    chemin = os.path.join(folder, "etude_convergence_temps.csv")
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["N", "BS", "CRR", "Trinomial", "err_CRR", "err_Trinomial",
                    "CRR_2N", "ecart_Tri(N)-CRR(2N)", "t_CRR_ms", "t_Trinomial_ms"])
        for i, n in enumerate(d["N"]):
            w.writerow([int(n), f"{ref:.10f}", f"{d['crr'][i]:.10f}", f"{d['tri'][i]:.10f}",
                        f"{d['err_crr'][i]:.3e}", f"{d['err_tri'][i]:.3e}",
                        f"{d['crr2n'][i]:.10f}", f"{d['ecart_eq'][i]:.3e}",
                        f"{d['t_crr'][i] * 1000:.5f}", f"{d['t_tri'][i] * 1000:.5f}"])
    fichiers.append(chemin)

    # --- Tableau LaTeX : convergence
    chemin = os.path.join(folder, "tab_convergence.tex")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rrrrr}\n\\hline\n")
        f.write("$N$ & $V_N^{\\mathrm{CRR}}$ & $E_{\\mathrm{CRR}}$ & "
                "$V_N^{\\mathrm{Boyle}}$ & $E_{\\mathrm{Boyle}}$ \\\\\n\\hline\n")
        for i, n in enumerate(d["N"]):
            f.write(f"{int(n)} & {_fr(d['crr'][i])} & {_sci(d['err_crr'][i])} & "
                    f"{_fr(d['tri'][i])} & {_sci(d['err_tri'][i])} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    fichiers.append(chemin)

    # --- Tableau LaTeX : temps
    chemin = os.path.join(folder, "tab_temps.tex")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rrrr}\n\\hline\n")
        f.write("$N$ & CRR (ms) & Boyle (ms) & Rapport Boyle/CRR \\\\\n\\hline\n")
        for i, n in enumerate(d["N"]):
            r = d["t_tri"][i] / d["t_crr"][i]
            f.write(f"{int(n)} & {_fr(d['t_crr'][i] * 1000, 3)} & "
                    f"{_fr(d['t_tri'][i] * 1000, 3)} & {_fr(r, 2)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    fichiers.append(chemin)

    # --- Tableau LaTeX : équivalence Tri(N) = CRR(2N)
    chemin = os.path.join(folder, "tab_equivalence.tex")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rrrr}\n\\hline\n")
        f.write("$N$ & $V_N^{\\mathrm{Boyle}}$ & $V_{2N}^{\\mathrm{CRR}}$ & $D_N$ \\\\\n\\hline\n")
        for i, n in enumerate(d["N"]):
            f.write(f"{int(n)} & {_fr(d['tri'][i], 8)} & {_fr(d['crr2n'][i], 8)} & "
                    f"{_sci(d['ecart_eq'][i])} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    fichiers.append(chemin)

    # --- Résumé texte (pentes, paramètres)
    chemin = os.path.join(folder, "resume.txt")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("Paramètres : " + ", ".join(f"{k}={v}" for k, v in p.items())
                + f", type={d['option']}\n")
        f.write(f"Prix Black-Scholes : {ref:.10f}\n")
        f.write(f"Pente log-log CRR (N>=50)      : {pente_loglog(d['N'], d['err_crr']):.4f}\n")
        f.write(f"Pente log-log Trinomial (N>=50): {pente_loglog(d['N'], d['err_tri']):.4f}\n")
        f.write(f"Écart max |Tri(N) - CRR(2N)|   : {d['ecart_eq'].max():.3e}\n")
        f.write(f"Rapport moyen temps Boyle/CRR  : {np.mean(d['t_tri'] / d['t_crr']):.2f}\n")
    fichiers.append(chemin)

    # --- Figures (PDF pour LaTeX + PNG)
    for nom, fn in (("fig_convergence", draw_convergence), ("fig_temps", draw_time)):
        fig = Figure(figsize=(10, 4.2), dpi=150)
        fn(fig, d)
        for ext in ("pdf", "png"):
            chemin = os.path.join(folder, f"{nom}.{ext}")
            fig.savefig(chemin)
            fichiers.append(chemin)
    return fichiers


def probas_valides(p, n):
    """Vérifie 0 <= p <= 1 pour CRR et 1/b <= a <= b pour Boyle (pas dt = T/n)."""
    dt = p["T"] / n
    u = np.exp(p["sigma"] * np.sqrt(dt))
    pc = (np.exp((p["r"] - p["q"]) * dt) - 1 / u) / (u - 1 / u)
    a = np.exp((p["r"] - p["q"]) * dt / 2)
    b = np.exp(p["sigma"] * np.sqrt(dt / 2))
    return bool(0 <= pc <= 1), bool(1 / b <= a <= b)


# =====================================================================
#  2. INTERFACE GRAPHIQUE
# =====================================================================

NAVY, GOLD, PALE, GREY, TRACK = "#13315C", "#B78A2A", "#EAEFF5", "#5F6673", "#D5DAE2"
BLUE = NAVY
C_BIN, C_TRI, C_BS = "#1F77B4", "#D9534F", "#2E8B57"

def scenarios(opt):
    """Scénarios proposés selon le type d'option (call ou put)."""
    base = dict(S="100", K="100", T="1", r="0.05", sigma="0.20", q="0")
    itm, otm = ("120", "80") if opt == "call" else ("80", "120")
    return {
        "Cas de référence (à la monnaie)": dict(base),
        "Dans la monnaie": dict(base, S=itm),
        "Hors de la monnaie": dict(base, S=otm),
        "Forte volatilité": dict(base, sigma="0.50"),
        "Courte maturité": dict(base, T="0.25"),
    }


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Évaluation d'options européennes")
        self.geometry("1260x780")
        self.minsize(1080, 680)
        self.configure(bg="white")
        self.last = None
        self.q = queue.Queue()
        self.v = {}
        self._style()

        # ---- bandeau de titre
        head = tk.Frame(self, bg=NAVY)
        head.pack(fill="x")
        tk.Label(head, text="Évaluation d'options européennes", bg=NAVY, fg="white",
                 font=("Segoe UI", 17, "bold")).pack(anchor="w", padx=22, pady=(14, 0))
        tk.Label(head, text="Black–Scholes–Merton  ·  Arbre binomial CRR  ·  Arbre trinomial de Boyle",
                 bg=NAVY, fg="#C9D3E3", font=("Segoe UI", 10)).pack(anchor="w", padx=22, pady=(0, 12))
        tk.Frame(self, bg=GOLD, height=3).pack(fill="x")

        # ---- barre d'état (en bas)
        bas = tk.Frame(self, bg=PALE)
        bas.pack(side="bottom", fill="x")
        self.status = tk.StringVar(value="Prêt.")
        tk.Label(bas, textvariable=self.status, bg=PALE, fg=NAVY, anchor="w",
                 font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True, padx=12, pady=5)
        self.prog = ttk.Progressbar(bas, length=180, mode="determinate")
        self.prog.pack(side="right", padx=12)

        # ---- panneau gauche
        side = tk.Frame(self, bg=PALE, width=290)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        self._titre(side, "Option")
        row = tk.Frame(side, bg=PALE)
        row.pack(fill="x", padx=18, pady=(2, 4))
        self.opt = tk.StringVar(value="call")
        for txt in ("call", "put"):
            ttk.Radiobutton(row, text=txt.capitalize(), value=txt, variable=self.opt,
                            style="Side.TRadiobutton",
                            command=self.changer_type).pack(side="left", padx=(0, 18))
        self.preset = tk.StringVar(value=list(scenarios("call"))[0])
        self.cb = ttk.Combobox(side, textvariable=self.preset, values=list(scenarios("call")),
                               state="readonly")
        self.cb.pack(fill="x", padx=18, pady=(4, 6))
        self.cb.bind("<<ComboboxSelected>>", self.appliquer_preset)

        self._titre(side, "Paramètres")
        self._champs(side, [("Spot S₀", "S", "100"), ("Strike K", "K", "100"),
                            ("Maturité T (ans)", "T", "1"), ("Taux sans risque r", "r", "0.05"),
                            ("Volatilité σ", "sigma", "0.20"), ("Dividende q", "q", "0")])

        self._titre(side, "Discrétisation")
        self._champs(side, [("Nombre de pas N", "N", "100"), ("N maximum", "Nmax", "1000"),
                            ("Nombre de points", "pts", "10")])

        btns = tk.Frame(side, bg=PALE)
        btns.pack(fill="x", padx=18, pady=(16, 0))
        self.b1 = ttk.Button(btns, text="Calculer", style="Accent.TButton", command=self.calculer)
        self.b2 = ttk.Button(btns, text="Convergence et temps", style="Accent.TButton", command=self.etude)
        self.b3 = ttk.Button(btns, text="Exporter les données", style="Second.TButton", command=self.exporter)
        for b in (self.b1, self.b2, self.b3):
            b.pack(fill="x", pady=4)
        self.bind("<Return>", lambda e: self.calculer())

        # ---- zone principale
        main = tk.Frame(self, bg="white")
        main.pack(side="left", fill="both", expand=True)
        self.nb = ttk.Notebook(main)
        self.nb.pack(fill="both", expand=True, padx=14, pady=12)

        t1 = ttk.Frame(self.nb)
        self.nb.add(t1, text="Résultats")
        cols = ("Méthode", "Prix", "Écart absolu vs BS", "Écart relatif (%)", "Temps (ms)")
        self.tree = self._table(t1, cols, height=6, width=170)
        self.tree.master.pack(fill="x", padx=10, pady=(18, 8))
        self.note = tk.StringVar(value="Saisissez les paramètres puis cliquez sur « Calculer ».")
        tk.Label(t1, textvariable=self.note, bg="white", fg=GREY, wraplength=820, justify="left",
                 font=("Segoe UI", 10)).pack(anchor="w", padx=14, pady=4)

        self.fig1, self.fig2 = Figure(figsize=(9, 4.6)), Figure(figsize=(9, 4.6))
        for fig, titre in ((self.fig1, "Convergence"), (self.fig2, "Temps de calcul")):
            fr = ttk.Frame(self.nb)
            self.nb.add(fr, text=titre)
            fig.canvas_tk = FigureCanvasTkAgg(fig, master=fr)
            fig.canvas_tk.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(fig.canvas_tk, fr).update()

        t4 = ttk.Frame(self.nb)
        self.nb.add(t4, text="Données")
        cols = ("N", "CRR", "Err. CRR", "Trinomial", "Err. Tri", "CRR(2N)",
                "|Tri(N)−CRR(2N)|", "t CRR (ms)", "t Tri (ms)")
        self.tdata = self._table(t4, cols, height=18, width=95)
        self.tdata.master.pack(fill="both", expand=True, padx=10, pady=10)

    # ---- petits composants
    def _titre(self, parent, txt):
        tk.Label(parent, text=txt.upper(), bg=PALE, fg=NAVY,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=18, pady=(16, 2))
        tk.Frame(parent, bg=GOLD, height=2, width=34).pack(anchor="w", padx=18, pady=(0, 6))

    def _champs(self, parent, champs):
        for txt, key, val in champs:
            ligne = tk.Frame(parent, bg=PALE)
            ligne.pack(fill="x", padx=18, pady=2)
            tk.Label(ligne, text=txt, bg=PALE, fg=GREY, font=("Segoe UI", 10),
                     anchor="w").pack(side="left")
            self.v[key] = tk.StringVar(value=val)
            ttk.Entry(ligne, textvariable=self.v[key], width=9,
                      justify="center").pack(side="right")

    def _table(self, parent, cols, height, width):
        cadre = tk.Frame(parent, bg="white")
        tv = ttk.Treeview(cadre, columns=cols, show="headings", height=height)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, anchor="center", width=width)
        tv.tag_configure("pair", background=PALE)
        sb = ttk.Scrollbar(cadre, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tv.pack(side="left", fill="both", expand=True)
        return tv

    def _remplir(self, tv, lignes):
        tv.delete(*tv.get_children())
        for i, vals in enumerate(lignes):
            tv.insert("", "end", values=vals, tags=("pair",) if i % 2 else ())

    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TFrame", background="white")
        st.configure("TNotebook", background="white", borderwidth=0)
        st.configure("TNotebook.Tab", padding=(18, 8), font=("Segoe UI", 10),
                     background=PALE, foreground=GREY, borderwidth=0)
        st.map("TNotebook.Tab", background=[("selected", NAVY)],
               foreground=[("selected", "white")])
        st.configure("TEntry", fieldbackground="white", padding=3)
        st.configure("TCombobox", padding=3)
        st.configure("Side.TRadiobutton", background=PALE, foreground=NAVY,
                     font=("Segoe UI", 10, "bold"))
        st.configure("Accent.TButton", background=NAVY, foreground="white",
                     font=("Segoe UI", 10, "bold"), padding=9, borderwidth=0)
        st.map("Accent.TButton", background=[("active", "#24508F"), ("disabled", "#9FB3C8")])
        st.configure("Second.TButton", background="white", foreground=NAVY,
                     font=("Segoe UI", 10, "bold"), padding=9, bordercolor=NAVY)
        st.map("Second.TButton", background=[("active", PALE), ("disabled", "#EEE")])
        st.configure("Treeview.Heading", background=NAVY, foreground="white",
                     font=("Segoe UI", 10, "bold"), padding=6)
        st.configure("Treeview", rowheight=28, font=("Segoe UI", 10), borderwidth=0)
        st.configure("Horizontal.TProgressbar", background=GOLD, troughcolor=TRACK)

    def changer_type(self):
        """Call ou put : met à jour la liste des scénarios et les paramètres."""
        liste = list(scenarios(self.opt.get()))
        self.cb.configure(values=liste)
        if self.preset.get() not in liste:
            self.preset.set(liste[0])
        self.appliquer_preset()

    def appliquer_preset(self, _evt=None):
        for k, val in scenarios(self.opt.get())[self.preset.get()].items():
            self.v[k].set(val)

    # ---- lecture + validation
    def lire(self):
        try:
            p = {k: float(self.v[k].get().replace(",", ".")) for k in ("S", "K", "T", "r", "sigma", "q")}
            n, nmax, pts = int(self.v["N"].get()), int(self.v["Nmax"].get()), int(self.v["pts"].get())
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez saisir des nombres valides.")
            return None
        if min(p["S"], p["K"], p["T"], p["sigma"]) <= 0 or n < 1 or nmax < 10 or pts < 3:
            messagebox.showerror("Erreur", "S, K, T, σ > 0 ; N ≥ 1 ; N max ≥ 10 ; Points ≥ 3.")
            return None
        return p, n, nmax, pts

    # ---- 1. calcul pour un N
    def calculer(self):
        lu = self.lire()
        if not lu:
            return
        p, N, _, _ = lu
        ok_c, ok_t = probas_valides(p, N)
        if not (ok_c and ok_t):
            messagebox.showerror("Probabilités invalides",
                                 "Pour ce N, les probabilités de transition ne sont pas dans [0, 1] "
                                 "(CRR : %s, Boyle : %s). Augmentez N ou réduisez |r−q|."
                                 % ("ok" if ok_c else "invalide", "ok" if ok_t else "invalide"))
            return
        opt = self.opt.get()
        a = (p["S"], p["K"], p["T"], p["r"], p["sigma"])
        bs, t0 = chrono(black_scholes, *a, p["q"], opt)
        vb, tb = chrono(binomial_crr, *a, N, p["q"], opt)
        vt, tt = chrono(trinomial_boyle, *a, N, p["q"], opt)
        v2, t2 = chrono(binomial_crr, *a, 2 * N, p["q"], opt)
        lignes = []
        for nom, v, t in (("Black-Scholes (référence)", bs, t0), (f"Binomial CRR (N={N})", vb, tb),
                          (f"Trinomial Boyle (N={N})", vt, tt), (f"Binomial CRR (2N={2 * N})", v2, t2)):
            e = abs(v - bs)
            lignes.append((nom, f"{v:.6f}", f"{e:.2e}",
                           f"{100 * e / bs:.4f}" if bs else "-", f"{t * 1000:.4f}"))
        self._remplir(self.tree, lignes)
        self.note.set(f"Écart |Trinomial(N) − CRR(2N)| = {abs(vt - v2):.2e} : "
                      "les deux prix semblent coïncider, ce qui est cohérent avec l'équivalence théorique.")
        self.status.set(f"Calcul terminé (N={N}).")
        self.nb.select(0)

    # ---- 2. étude (dans un thread pour ne pas figer la fenêtre)
    def etude(self):
        lu = self.lire()
        if not lu:
            return
        p, _, nmax, pts = lu
        Ns = [int(n) for n in np.unique(np.linspace(5, nmax, pts).astype(int))
              if all(probas_valides(p, int(n)))]
        if len(Ns) < 3:
            messagebox.showerror("Étude", "Trop peu de valeurs de N valides (probabilités hors de [0, 1]).")
            return
        for b in (self.b1, self.b2, self.b3):
            b.state(["disabled"])
        self.prog.configure(maximum=len(Ns), value=0)
        opt = self.opt.get()

        def travail():
            try:
                res = run_study(p, opt, Ns, lambda k, tot, n: self.q.put(("prog", k, n)))
                self.q.put(("fin", res))
            except Exception as e:                     # erreur remontée dans l'interface
                self.q.put(("err", str(e)))

        threading.Thread(target=travail, daemon=True).start()
        self.after(100, self._suivre)

    def _suivre(self):
        fini = False
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "prog":
                    self.prog.configure(value=msg[1])
                    self.status.set(f"Étude en cours... (N={msg[2]})")
                elif msg[0] == "fin":
                    self._afficher(msg[1]); fini = True
                else:
                    messagebox.showerror("Étude", msg[1]); fini = True
        except queue.Empty:
            pass
        if fini:
            for b in (self.b1, self.b2, self.b3):
                b.state(["!disabled"])
        else:
            self.after(100, self._suivre)

    def _afficher(self, d):
        self.last = d
        draw_convergence(self.fig1, d); self.fig1.canvas_tk.draw()
        draw_time(self.fig2, d); self.fig2.canvas_tk.draw()
        self._remplir(self.tdata, [
            (int(n), f"{d['crr'][i]:.6f}", f"{d['err_crr'][i]:.2e}", f"{d['tri'][i]:.6f}",
             f"{d['err_tri'][i]:.2e}", f"{d['crr2n'][i]:.6f}", f"{d['ecart_eq'][i]:.1e}",
             f"{d['t_crr'][i] * 1000:.3f}", f"{d['t_tri'][i] * 1000:.3f}")
            for i, n in enumerate(d["N"])])
        self.status.set(f"Terminé. Pentes log-log : CRR {pente_loglog(d['N'], d['err_crr']):.3f}, "
                        f"Trinomial {pente_loglog(d['N'], d['err_tri']):.3f}. "
                        f"Rapport de temps Boyle/CRR moyen : {np.mean(d['t_tri'] / d['t_crr']):.2f}.")
        self.nb.select(1)

    # ---- 3. export
    def exporter(self):
        if self.last is None:
            messagebox.showinfo("Export", "Lancez d'abord « Convergence et temps ».")
            return
        dossier = filedialog.askdirectory(title="Dossier d'export")
        if dossier:
            try:
                fichiers = export_all(dossier, self.last)
            except OSError as e:
                messagebox.showerror("Export", f"Écriture impossible : {e}")
                return
            self.status.set(f"{len(fichiers)} fichiers exportés dans {dossier}")
            messagebox.showinfo("Export", "Fichiers créés :\n" + "\n".join(os.path.basename(f) for f in fichiers))


if __name__ == "__main__":
    App().mainloop()
