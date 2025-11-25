# Augmentation script

Este repositório contém `augment_directory.py`, um script para gerar imagens aumentadas diretamente no mesmo diretório de origem de uma classe.

Principais pontos:
- As imagens geradas são salvas no mesmo diretório da classe.
- Você escolhe as operações (`--ops`) a aplicar.
- Define-se o número total desejado (`--target`) — o script gera apenas o que falta.
- Nomes das imagens geradas seguem o padrão `aug_<origem>_<ops>_<id>.<ext>`.

Operações suportadas:
- `rotate` — rotação aleatória entre -30 e 30 graus
- `hflip` — flip horizontal
- `vflip` — flip vertical
- `brightness` — ajuste de brilho
- `contrast` — ajuste de contraste
- `blur` — Gaussian blur
- `noise` — ruído gaussiano
- `crop` — crop aleatório seguido de resize para o tamanho original

Instalação

No Windows (PowerShell), crie e ative um ambiente e instale dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "C:\Users\IA\Desktop\citology pipeline Train\Dataset\pre-processado\3 Classes\requirements.txt"
```

Exemplos de uso

- Gerar até 8000 imagens na pasta de uma classe com operações aleatórias (modo `random`):

```powershell
python "C:\Users\IA\Desktop\citology pipeline Train\Dataset\pre-processado\3 Classes\augment_directory.py" --dir "C:\caminho\para\classe" --target 8000 --ops rotate,hflip,noise --mode random
```

- Aplicar todas as operações escolhidas (modo `all`) até atingir 12000 imagens:

```powershell
python "C:\Users\IA\Desktop\citology pipeline Train\Dataset\pre-processado\3 Classes\augment_directory.py" --dir "C:\caminho\para\classe" --target 12000 --ops rotate,hflip,brightness,contrast,noise --mode all
```

- Testar sem salvar (`--dry-run`):

```powershell
python "C:\Users\IA\Desktop\citology pipeline Train\Dataset\pre-processado\3 Classes\augment_directory.py" --dir "C:\caminho\para\classe" --target 5000 --ops rotate,hflip --dry-run
```

Notas

- O script preserva extensões `.jpg`, `.jpeg` e `.png`.
- Se `--target` for menor ou igual ao número existente de imagens, nada é gerado.
- Ajuste as faixas de parâmetros (ângulos, brilho, ruído) diretamente no arquivo se desejar valores diferentes.

Se quiser, posso:
- Adicionar um modo que re-use uma lista específica de imagens origem (por exemplo, só imagens com determinado sufixo).
- Incluir processamento paralelo para acelerar a geração.
- Gerar um pequeno relatório CSV com quais augmentações foram aplicadas a cada arquivo.
