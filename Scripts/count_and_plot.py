#!/usr/bin/env python3
"""
count_and_plot.py

Conta imagens por classe (subdiretórios) e gera um gráfico de barras.

- Cada subdiretório no `--root` é considerado uma classe.
- Detecta imagens aumentadas por prefixo (por padrão 'aug').
- Se `--include-aug` for passado, o gráfico mostra barras empilhadas: originais + aumentadas.

Exemplos (PowerShell):

python .\count_and_plot.py --root "C:\caminho\para\dataset" --include-aug --out counts.png --show
python .\count_and_plot.py --root . --out counts.png --csv counts.csv

"""
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import sys

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_EXTS = (".jpg", ".jpeg", ".png")


def count_images_in_dir(class_dir: Path, exts=DEFAULT_EXTS, aug_prefix: str = "aug") -> Tuple[int, int]:
    """Retorna (original_count, augmented_count) para a pasta dada.
    Um arquivo é considerado 'augmented' se seu nome (sem caminho) startswith(aug_prefix) (case-insensitive).
    """
    orig = 0
    aug = 0
    prefix_low = aug_prefix.lower()
    for p in class_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        name_low = p.name.lower()
        if name_low.startswith(prefix_low):
            aug += 1
        else:
            orig += 1
    return orig, aug


def scan_root(root: Path, exts=DEFAULT_EXTS, aug_prefix: str = "aug") -> Dict[str, Tuple[int, int]]:
    results = {}
    for child in sorted(root.iterdir()):
        if child.is_dir():
            orig, aug = count_images_in_dir(child, exts=exts, aug_prefix=aug_prefix)
            results[child.name] = (orig, aug)
    return results


def make_dataframe(counts: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for cls, (orig, aug) in counts.items():
        rows.append({"class": cls, "original": orig, "augmented": aug, "total": orig + aug})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(by="total", ascending=False).reset_index(drop=True)
    return df


def plot_counts(df: pd.DataFrame, include_aug: bool, out: Path = None, show: bool = False):
    if df.empty:
        print("Nenhuma classe encontrada ou nenhuma imagem presente.")
        return

    labels = df["class"].tolist()
    originals = df["original"].tolist()
    augmented = df["augmented"].tolist()

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 6))

    if include_aug:
        p1 = ax.bar(x, originals, label="original", color="#4C72B0")
        p2 = ax.bar(x, augmented, bottom=originals, label="augmented", color="#DD8452")
    else:
        p1 = ax.bar(x, originals, label="original", color="#4C72B0")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Quantidade de imagens (total)")
    ax.set_xlabel("Classe (nome da pasta)")
    ax.set_title("Distribuição de imagens por classe")
    ax.legend()
    plt.tight_layout()

    if out:
        fig.savefig(str(out), dpi=150)
        print(f"Gráfico salvo em: {out}")
    if show:
        plt.show()
    plt.close(fig)


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"CSV salvo em: {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Conta imagens por classe e plota distribuição com opção de incluir augmentations.")
    p.add_argument("--root", required=False, default=".", help="Diretório raiz que contém as pastas de classe")
    p.add_argument("--exts", default=",".join(DEFAULT_EXTS), help="Extensões separadas por vírgula para considerar (ex: .jpg,.png)")
    p.add_argument("--aug-prefix", default="aug", help="Prefixo usado para identificar imagens aumentadas (padrão: 'aug')")
    p.add_argument("--include-aug", action="store_true", help="Se passado, mostra a porção de imagens aumentadas empilhadas na mesma barra")
    p.add_argument("--out", default=None, help="Caminho para salvar o gráfico PNG (ex: counts.png)")
    p.add_argument("--csv", default=None, help="Caminho para salvar CSV com contagens por classe")
    p.add_argument("--show", action="store_true", help="Mostrar o gráfico na tela")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"Diretório inválido: {root}")
        sys.exit(1)

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())

    counts = scan_root(root, exts=exts, aug_prefix=args.aug_prefix)
    df = make_dataframe(counts)

    out_path = Path(args.out) if args.out else None
    csv_path = Path(args.csv) if args.csv else None

    plot_counts(df, include_aug=args.include_aug, out=out_path, show=args.show)

    if csv_path:
        write_csv(df, csv_path)


if __name__ == "__main__":
    main()
