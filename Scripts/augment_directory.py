#!/usr/bin/env python3
"""
augment_directory.py

Cria imagens aumentadas dentro do mesmo diretório de origem.

Uso:
  python augment_directory.py --dir /caminho/para/classe --target 8000 --ops rotate,hflip,noise --mode random

Suporta operações: rotate, hflip, vflip, brightness, contrast, blur, noise, crop
"""
import argparse
import os
import random
import uuid
from pathlib import Path
from typing import List

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


SUPPORTED_OPS = [
    "rotate",
    "hflip",
    "vflip",
    "brightness",
    "contrast",
    "blur",
    "noise",
    "crop",
]


def list_image_files(directory: Path, exts=(".jpg", ".jpeg", ".png")) -> List[Path]:
    files = [p for p in directory.iterdir() if p.suffix.lower() in exts and p.is_file()]
    return sorted(files)


def apply_rotate(img: Image.Image):
    angle = random.uniform(-30, 30)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)


def apply_hflip(img: Image.Image):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def apply_vflip(img: Image.Image):
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def apply_brightness(img: Image.Image):
    factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_contrast(img: Image.Image):
    factor = random.uniform(0.6, 1.4)
    return ImageEnhance.Contrast(img).enhance(factor)


def apply_blur(img: Image.Image):
    radius = random.uniform(0.0, 2.0)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_noise(img: Image.Image):
    arr = np.array(img).astype(np.float32)
    sigma = random.uniform(5, 25)
    noise = np.random.normal(0, sigma, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_crop(img: Image.Image):
    w, h = img.size
    if w < 20 or h < 20:
        return img
    crop_scale = random.uniform(0.8, 1.0)
    new_w = int(w * crop_scale)
    new_h = int(h * crop_scale)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), resample=Image.BICUBIC)


OP_FN = {
    "rotate": apply_rotate,
    "hflip": apply_hflip,
    "vflip": apply_vflip,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "blur": apply_blur,
    "noise": apply_noise,
    "crop": apply_crop,
}


def build_aug_name(orig: Path, ops_applied: List[str], index: int, prefix: str = "aug") -> str:
    base = orig.stem
    ops_tag = "-".join(ops_applied) if ops_applied else "none"
    return f"{prefix}_{base}_{ops_tag}_{index}{orig.suffix}"


def augment_directory(
    directory: Path,
    target_total: int,
    ops: List[str],
    mode: str = "random",
    seed: int = None,
    dry_run: bool = False,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    files = list_image_files(directory)
    n_existing = len(files)
    if n_existing == 0:
        raise SystemExit(f"Nenhuma imagem encontrada em {directory}")

    if target_total <= n_existing:
        print(f"Target {target_total} <= existentes {n_existing}. Nada a fazer.")
        return

    to_generate = target_total - n_existing
    print(f"Existentes: {n_existing}. Gerando: {to_generate} imagens para atingir {target_total}.")

    # Prevent name collisions: collect existing names
    existing_names = set(p.name for p in files)
    idx = 0
    tries = 0
    while idx < to_generate:
        tries += 1
        if tries > to_generate * 20:
            raise RuntimeError("Muitas tentativas para gerar nomes únicos — verifique permissões/nomes no diretório")

        src = random.choice(files)
        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            print(f"Falha ao abrir {src}: {e}. Pulando.")
            continue

        if mode == "random":
            # apply 1..min(3,len(ops)) random ops
            nops = random.randint(1, max(1, min(3, len(ops))))
            ops_applied = random.sample(ops, nops)
        else:
            ops_applied = ops.copy()

        out = img
        for op in ops_applied:
            fn = OP_FN.get(op)
            if fn:
                try:
                    out = fn(out)
                except Exception as e:
                    print(f"Erro aplicando {op} em {src}: {e}")

        # Build name and ensure uniqueness
        candidate_name = build_aug_name(src, ops_applied, uuid.uuid4().hex[:8])
        if candidate_name in existing_names:
            continue

        out_path = directory / candidate_name
        if dry_run:
            print(f"[DRY] Salvando {out_path} (ops: {ops_applied})")
        else:
            try:
                out.save(out_path, quality=95)
            except Exception as e:
                print(f"Falha ao salvar {out_path}: {e}")
                continue

        existing_names.add(candidate_name)
        idx += 1

    print(f"Pronto. Diretório {directory} agora tem {target_total} imagens (ou mais, se houveram outros arquivos).")


def parse_args():
    p = argparse.ArgumentParser(description="Augment images inside a class directory until a target total is reached.")
    p.add_argument("--dir", required=True, help="Diretório da classe a aumentar")
    p.add_argument("--target", type=int, required=True, help="Número total desejado (existentes + geradas)")
    p.add_argument(
        "--ops",
        type=str,
        default=",".join(SUPPORTED_OPS),
        help=f"Operações permitidas, separadas por vírgula. Suportadas: {', '.join(SUPPORTED_OPS)}",
    )
    p.add_argument("--mode", choices=("random", "all"), default="random", help="'random' aplica subset aleatório; 'all' aplica todas as ops escolhidas")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Mostra o que seria criado sem salvar")
    return p.parse_args()


def main():
    args = parse_args()
    directory = Path(args.dir)
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"Diretório inválido: {directory}")

    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    unknown = [o for o in ops if o not in SUPPORTED_OPS]
    if unknown:
        raise SystemExit(f"Operações desconhecidas: {unknown}. Suportadas: {SUPPORTED_OPS}")

    augment_directory(directory, args.target, ops, mode=args.mode, seed=args.seed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
