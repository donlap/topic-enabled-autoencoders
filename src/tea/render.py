"""LaTeX rows of the tables of the paper."""

ORDER = ("WMF", "LightGCN", "SLIMElastic", "ADMMSLIM", "NCEPLRec", "RecVAE", "LDA", "ETM", "ECRTM", "Mult-VAE",
         "NMF", "NMF-ppSPA", "EASE", "RW", "Symmetric", "DAN", "RDLAE", "TEA")
RULE_AFTER = ("Mult-VAE", "NMF-ppSPA", "RDLAE")
DATASETS = ("books", "cds", "gowalla", "ml-1m", "retail", "steam")


def table1(cells, bold):
    """Return the rows of Table 1.

    cells maps a dataset and a method to its NDCG@10, recall@10, and TQ, where TQ is None without topics.
    bold maps a dataset and a metric to the methods printed in bold.
    """
    lines = []
    for method in ORDER:
        row = [method]
        for ds in DATASETS:
            for metric in ("ndcg", "recall", "tq"):
                v = cells[ds][method].get(metric)
                if v is None:
                    row.append("$-$")
                    continue
                s = f"{v:.3f}"
                row.append(f"\\textbf{{{s}}}" if method in bold[ds][metric] else s)
        lines.append(" & ".join(row) + " \\\\")
        if method in RULE_AFTER:
            lines.append("\\midrule")
    return "\n".join(lines) + "\n"


TABLE3 = (
    (("WMF", (("factors", "number of factors"), ("regularization", "regularization"))),
     ("LightGCN", (("learning_rate", "learning rate"), ("layers", "number of layers"))),
     ("SLIMElastic", (("alpha", "regularization"), ("l1_ratio", "L1 ratio"))),
     ("ADMMSLIM", (("lambda1", "L1 regularization"), ("lambda2", "L2 regularization"))),
     ("NCEPLRec", (("regularization", "regularization"), ("rank", "rank"))),
     ("RecVAE", (("latent_dim", "latent dimension"), ("gamma", "KL scale")))),
    (("LDA", (("max_iter", "number of iterations"), ("doc_topic_prior", "prior on topic proportions"),
              ("topic_word_prior", "prior on topics"))),
     ("ETM", (("learning_rate", "learning rate"), ("hidden_size", "hidden size"), ("dropout", "dropout probability"),
              ("kl_weight", "KL weight"))),
     ("ECRTM", (("learning_rate", "learning rate"), ("temperature", "temperature"), ("ecr_weight", "ECR weight"),
                ("kl_weight", "KL weight"))),
     ("Mult-VAE", (("learning_rate", "learning rate"), ("kl_weight", "KL weight"),
                   ("dropout", "dropout probability")))),
    (("NMF", (("alpha", "regularization"), ("l1_ratio", "L1 ratio"), ("init", "initialization"))),
     ("NMF-ppSPA", (("alpha", "regularization"), ("l1_ratio", "L1 ratio"), ("init", "initialization")))),
    (("EASE", (("lambda", "$\\lambda$"),)),
     ("RW", (("lambda", "$\\lambda$"),)),
     ("Symmetric", (("lambda", "$\\lambda$"),)),
     ("DAN", (("lambda", "$\\lambda$"), ("alpha", "$\\alpha$"), ("user_exponent", "$\\beta$ (the user exponent)"))),
     ("RDLAE", (("lambda", "$\\lambda$"), ("p", "$p$"), ("xi", "$\\xi$")))),
    (("TEA", (("alpha", "$\\alpha$"), ("p", "$p$"), ("lambda", "$\\lambda$"))),),
)


def _value(v):
    if v is None:
        return "$1/K$"
    if isinstance(v, str):
        return f"\\texttt{{{v}}}"
    return str(int(v)) if float(v).is_integer() else ("%.6f" % v).rstrip("0")


def table3(hyperparameters):
    """Return the rows of Table 3 from the selected hyperparameters of each method on each dataset."""
    lines = []
    for g, group in enumerate(TABLE3):
        if g:
            lines.append("\\midrule")
        for i, (method, params) in enumerate(group):
            if i:
                lines.append("\\addlinespace[2pt]")
            for j, (key, label) in enumerate(params):
                values = " & ".join(_value(hyperparameters[method][ds][key]) for ds in DATASETS)
                lines.append(f"{method if j == 0 else ''} & {label} & {values} \\\\")
    return "\n".join(lines) + "\n"


TABLE2 = (("books", "amazon-books", "books"), ("cds", "amazon-cds", "albums"), ("ml-1m", "ml-1m", "films"),
          ("retail", "online-retail-ii", "stock lines"), ("steam", "steam", "games"))


def table2(shown, K):
    """Return the rows of Table 2.

    shown maps a dataset to its displayed topics, each with a label, the titles of its top items, and n_eff.
    """
    lines = []
    for i, (ds, name, noun) in enumerate(TABLE2):
        if i:
            lines.append("\\midrule")
        which = f"the $K={K[ds]}$ topics" if K[ds] <= 5 else f"five of the $K={K[ds]}$ topics"
        lines.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\texttt{{{name}}}: {which}. Items are {noun}.}} \\\\")
        lines.append("\\midrule")
        for t in shown[ds]:
            lines.append(f"{t['label']} & {t['titles']} & ${t['neff']:.1f}$ \\\\")
    return "\n".join(lines) + "\n"
