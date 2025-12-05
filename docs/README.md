# Documentação do Projeto: Citology Pipeline Train

- Objetivo: classificar lesões cervicais a partir de imagens, com pipelines de treino (Baseline, Robusto, Fine-Tuning e QAT) e inferência por tiles.
- Estrutura de diretórios:
  - `Notebooks/`
    - `cancer_training_pipeline.ipynb`
    - `QAT_cancer_training_pipeline.ipynb`
    - `cervical_cancer_classification.ipynb`
  - `Scripts/`
    - `create_labels.py`
    - `prepare_dataset.py`
    - `augment_directory.py`
    - `count_and_plot.py`
    - `rename.py`

## Construção do Dataset
- Geração de rótulos: varre subpastas (cada subpasta = classe) e cria `labels.csv` com colunas `Image_path` e `Label`.
  - Função: `create_label_file` em `Scripts/create_labels.py:4`
  - Saída: CSV com linhas por imagem; cada linha tem caminho absoluto e nome da classe.
- Divisão estratificada em treino/validação/teste: 70%/15%/15% respeitando a distribuição de classes; converte caminhos para relativos e renomeia colunas para `image_path` e `lesion_type`.
  - Função: `create_and_split_dataset` em `Scripts/prepare_dataset.py:5`
  - Parâmetros: `test_size=0.3`, `val_size=0.5` dentro do bloco 30% temporário; `random_state=42`.
  - Detalhes relevantes: normaliza separadores, garante `image_path` relativo ao diretório base.
- Contagem e gráfico por classe: escaneia diretório raiz (cada subpasta = classe), contabiliza originais vs aumentadas (prefixo `aug`), gera gráfico PNG e opcionalmente CSV.
  - Entradas: `--root`, `--include-aug`, `--out`, `--csv`.
  - Funções: `scan_root` (`Scripts/count_and_plot.py:49`), `plot_counts` (`Scripts/count_and_plot.py:69`).
- Augmentação de diretórios: gera imagens aumentadas no mesmo diretório até atingir um total alvo; operações suportadas: `rotate`, `hflip`, `vflip`, `brightness`, `contrast`, `blur`, `noise`, `crop`.
  - Função principal: `augment_directory` em `Scripts/augment_directory.py:108`
  - Nome de saída: `aug_<origem>_<ops>_<id>.<ext>`.

## Resoluções de Imagem
- Tamanho usado no treino: `IMG_HEIGHT=224`, `IMG_WIDTH=224`.
  - Referências:
    - `Notebooks/cancer_training_pipeline.ipynb:96–102`
    - `Notebooks/QAT_cancer_training_pipeline.ipynb:117–120`
    - `Notebooks/cervical_cancer_classification.ipynb:138–145`
- Resolução de entrada da rede (MobileNetV2): `input_shape=(224, 224, 3)`.
  - Construção do backbone:
    - `Notebooks/cancer_training_pipeline.ipynb:177–182`
    - `Notebooks/QAT_cancer_training_pipeline.ipynb:227–231`
    - `Notebooks/cervical_cancer_classification.ipynb:240–246`
- Resolução dos tiles na inferência: extração de patches grandes de `768x1024` e redimensionamento para `224x224` antes de alimentar a rede.
  - Definições: `PATCH_HEIGHT=768`, `PATCH_WIDTH=1024` em
    - `Notebooks/cancer_training_pipeline.ipynb:875–876`
    - `Notebooks/QAT_cancer_training_pipeline.ipynb:3301–3302`

## Pré‑processamento de Inferência (por Notebook)
- Pipeline clássico (`Notebooks/cancer_training_pipeline.ipynb`):
  - Carregamento do modelo e definição do input size: `load_inference_model` (`Notebooks/cancer_training_pipeline.ipynb:1286–1297`).
  - Extração de tiles válidos: percorre a imagem em janelas de `768x1024`, calcula desvio padrão no canal cinza e seleciona tiles com `np.std(patch_gray) > CONTENT_THRESHOLD` (`CONTENT_THRESHOLD=10.0`).
    - Definições e uso:
      - `CONTENT_THRESHOLD=10.0` (`Notebooks/cancer_training_pipeline.ipynb:879`)
      - `extract_valid_patches` (`Notebooks/cancer_training_pipeline.ipynb:1303–1323`), seleção: `np.std(patch_gray)` (`Notebooks/cancer_training_pipeline.ipynb:1341–1342`).
  - Pré‑processamento do patch para Keras/TF: BGR→RGB, resize para `_IMG_SIZE` (224x224), normalização `float32` em `[0,1]`, adiciona dimensão de batch.
    - Função: `preprocess_patch` (`Notebooks/cancer_training_pipeline.ipynb:908–923`).
  - Pipeline de inferência por tiles: cria lote, redimensiona e opcionalmente salva tiles com nomes contendo dimensões; garante `expected_shape=(1, 224, 224, 3)`.
    - Função: `process_and_infer` (`Notebooks/cancer_training_pipeline.ipynb:929–1061`), uso de `expected_shape` em (`Notebooks/cancer_training_pipeline.ipynb:999`).
- Pipeline QAT/TFLite (`Notebooks/QAT_cancer_training_pipeline.ipynb`):
  - Carregamento do modelo (Keras ou TFLite) e definição dos detalhes de tensor (dtype, shape, scale/zero‑point): `load_inference_model` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3340–3394`).
  - Pré‑processamento condicional pelo dtype de entrada:
    - `uint8`: mantém faixa `[0,255]`, adiciona batch.
    - `int8`: aplica quantização simétrica com `scale` e `zero_point` do input.
    - `float32`: normaliza para `[0,1]`.
    - Função: `preprocess_patch` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3396–3446`).
  - Desquantização da saída TFLite: `(output - zero_point) * scale`.
    - Função: `dequantize_output` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3448–3457`).
  - Pipeline de tiles e inferência completa: extração `768x1024`, resize `224x224`, montagem de lote, chamada do intérprete/keras, salvamento opcional e relatório.
    - Função: `process_and_infer` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3463–3668`).

## Funções Principais (por Notebook)
- `Notebooks/cancer_training_pipeline.ipynb`
  - `make_generators(train_df, val_df, test_df, ..., img_size=(224,224))` (`Notebooks/cancer_training_pipeline.ipynb:156–175`): cria geradores com/sem augmentation; `rescale=1./255` nos três splits.
  - `build_model(input_shape=(224,224,3), ...)` (`Notebooks/cancer_training_pipeline.ipynb:177–192`): MobileNetV2 como base (congelável), topo com GAP, `Dropout`, `Dense(128)`, `Dense(num_classes)` softmax.
  - `train_and_save(model, train_gen, val_gen, ...)` (`Notebooks/cancer_training_pipeline.ipynb:236–271`): treina com `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`; salva melhor checkpoint e um modelo final.
  - `plot_history(history, title)` (`Notebooks/cancer_training_pipeline.ipynb:319–338`): plota curvas de loss e accuracy.
  - `load_inference_model(model_path, class_names, img_size)` (`Notebooks/cancer_training_pipeline.ipynb:1286–1300`): carrega globalmente o modelo e define `_IMG_SIZE`.
  - `extract_valid_patches(image_path, output_dir, ...)` (`Notebooks/cancer_training_pipeline.ipynb:1303–1323`): extrai patches válidos usando desvio padrão no cinza; retorna lista e metadados.
  - `preprocess_patch(patch_img)` (`Notebooks/cancer_training_pipeline.ipynb:908–923`): BGR→RGB, resize `224x224`, normaliza `[0,1]`, adiciona batch.
  - `process_and_infer(...)` (`Notebooks/cancer_training_pipeline.ipynb:929–1061`): pipeline completo de inferência com tiles.
- `Notebooks/QAT_cancer_training_pipeline.ipynb`
  - `make_generators(...)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:209–219`): igual ao clássico; usa `target_size=(224,224)`.
  - `build_model(input_shape=(224,224,3), ...)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:227–231`): backbone MobileNetV2; cabeça com `GlobalAveragePooling2D`, `Dropout`, `Dense(128)`, `Dense(num_classes)`.
  - `save_history(history, stage_name, checkpoint_path)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:245–264`): persiste métricas e artefatos por etapa.
  - `plot_history(history, title, save_path)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:264–290`): plota e salva curvas.
  - `train_and_save(...)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:290–324`): treina com callbacks e retorna `(history, checkpoint_path)`.
  - `convert_to_lite(model, path)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:324–336`): converte para TFLite INT8, configura `supported_ops=INT8` e `inference_input_type/int8`.
  - `evaluate_model(model, test_gen, title, stage_name)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:864–902`): gera matriz de confusão e relatório de classificação.
  - `load_inference_model()` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3340–3394`): carrega Keras/TFLite, define `_IMG_SIZE`, `_INPUT_DETAILS`, `_OUTPUT_DETAILS`.
  - `preprocess_patch(patch_img, is_tflite)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3396–3446`): BGR→RGB, resize `224x224`, dtype‑aware (uint8/int8/float32).
  - `dequantize_output(output_tensor)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3448–3457`): desquantiza saída.
  - `process_and_infer(...)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3463–3668`): pipeline completo de inferência.
  - Utilitários de teste rápido: `quick_test()` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3725–3775`), `run_full_inference(...)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3789–3864`), `process_multiple_images(...)` (`Notebooks/QAT_cancer_training_pipeline.ipynb:3864–3923`).
- `Notebooks/cervical_cancer_classification.ipynb`
  - `create_mobilenet_model(input_shape=IMG_SIZE+(3,), num_classes)` (`Notebooks/cervical_cancer_classification.ipynb:240–246`): constrói MobileNetV2 com topo simples.
  - `plot_training_history(history, title)` (`Notebooks/cervical_cancer_classification.ipynb:470–?`): plota histórico de treino.

## Etapas de Treinamento (Resumo)
- Etapa 1 — Baseline: congela backbone, treina apenas o topo com `learning_rate=1e‑4` por `EPOCHS_BASELINE=10`; checkpoints e análise de métricas.
  - Referências: `Notebooks/cancer_training_pipeline.ipynb:200–293`, `Notebooks/QAT_cancer_training_pipeline.ipynb:354–854`.
- Etapa 2 — Robusto: habilita augmentation no gerador de treino; faz busca simples de `learning_rate` e salva o melhor (`robust_best.keras`).
  - Referências: `Notebooks/cancer_training_pipeline.ipynb:469–545`.
- Etapa 3 — Fine‑Tuning: descongela parte/todo o backbone, usa `learning_rate` baixo (`1e‑5`) e early stopping; resultado final `final_finetuned_checkpoint.keras`.
  - Referências: `Notebooks/cancer_training_pipeline.ipynb:614–697`, `Notebooks/QAT_cancer_training_pipeline.ipynb:129–139` (hiperparâmetros FT/QAT).
- Etapa 4 — QAT (Quantization‑Aware Training) e Conversão: ajusta com camadas quantizadas, exporta TFLite INT8; valida inferência.
  - Referências: `Notebooks/QAT_cancer_training_pipeline.ipynb:129–137`, `Notebooks/QAT_cancer_training_pipeline.ipynb:324–336`.

## Scripts Utilitários
- `Scripts/create_labels.py`
  - `create_label_file(dataset_path, output_csv)` (`Scripts/create_labels.py:4`): gera CSV de rótulos a partir de diretórios de classe.
- `Scripts/prepare_dataset.py`
  - `create_and_split_dataset(csv_path, train_path, val_path, test_path, ...)` (`Scripts/prepare_dataset.py:5`): divide dataset estratificadamente e normaliza colunas/caminhos.
- `Scripts/augment_directory.py`
  - `augment_directory(directory, target_total, ops, ...)` (`Scripts/augment_directory.py:108`): cria imagens aumentadas até atingir `target_total`.
  - `apply_*` (rotate/hflip/vflip/brightness/contrast/blur/noise/crop): operações de augmentation (`Scripts/augment_directory.py:40–99`).
- `Scripts/count_and_plot.py`
  - `scan_root(root, ...)` (`Scripts/count_and_plot.py:49`): coleta contagens por classe.
  - `plot_counts(df, include_aug, out, show)` (`Scripts/count_and_plot.py:69`): gera gráfico de barras empilhadas.
  - `write_csv(df, path)` (`Scripts/count_and_plot.py:103`): exporta contagens por classe.
- `Scripts/rename.py`
  - `rename_images_in_directory(directory_path, prefix)` (`Scripts/rename.py:3`): renomeia arquivos de imagem com prefixo.

## Observações Importantes
- Normalização em treino: os geradores usam `rescale=1./255` para treino/val/test; augmentation apenas no treino.
- Inferência clássica: normaliza para `[0,1]` e usa `float32`.
- Inferência TFLite: respeita `dtype` do modelo (`uint8`, `int8`, `float32`) e aplica desquantização na saída quando necessário.
- Tamanho dos tiles: extração `768x1024` para robustez; sempre redimensionados para `224x224` antes de inferir.

