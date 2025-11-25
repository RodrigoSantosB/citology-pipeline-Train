import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_and_split_dataset(csv_path, train_path, val_path, test_path, test_size=0.3, val_size=0.5, random_state=42):
    """
    Carrega um arquivo CSV, divide os dados em conjuntos de treino, validação e teste de forma estratificada
    e salva os resultados em novos arquivos CSV.

    Args:
        csv_path (str): Caminho para o arquivo CSV mestre.
        train_path (str): Caminho para salvar o CSV de treino.
        val_path (str): Caminho para salvar o CSV de validação.
        test_path (str): Caminho para salvar o CSV de teste.
        test_size (float): Proporção do dataset a ser alocada para o conjunto de teste e validação combinado.
        val_size (float): Proporção do conjunto de teste/validação a ser alocada para o conjunto de validação.
        random_state (int): Semente para reprodutibilidade.
    """
    try:
        # Carregar o dataset
        print(f"DEBUG: Iniciando o script. Carregando dados de: {csv_path}")
        df = pd.read_csv(csv_path)
        print("DEBUG: Dados carregados com sucesso.")

        # Ajustar os nomes das colunas e os caminhos das imagens
        print("DEBUG: Ajustando o dataframe...")
        if 'Image_path' not in df.columns:
            print("ERRO: Coluna 'Image_path' não encontrada no CSV.")
            return
        if 'Label' not in df.columns:
            print("ERRO: Coluna 'Label' não encontrada no CSV.")
            return

        base_dir = os.path.dirname(csv_path)
        print(f"DEBUG: Diretório base para caminhos relativos: {base_dir}")

        # Garante que os caminhos fiquem relativos e com barras corretas
        df['image_path'] = df['Image_path'].apply(lambda x: os.path.relpath(x, start=base_dir).replace('\\', '/'))
        df = df.rename(columns={'Label': 'lesion_type'})
        df = df.drop(columns=['Image_path'])
        print("DEBUG: Dataframe ajustado. Novas colunas: ", df.columns.tolist())
        print("DEBUG: Exemplo de caminho de imagem:", df['image_path'].iloc[0])

        print(f"Total de amostras: {len(df)}")
        print("Distribuição de classes original:")
        print(df['lesion_type'].value_counts())

        # Dividir em treino (70%) e um conjunto temporário (30%)
        print("DEBUG: Dividindo os dados...")
        train_df, temp_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['lesion_type']
        )

        # Dividir o conjunto temporário em validação (15%) e teste (15%)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_size,
            random_state=random_state,
            stratify=temp_df['lesion_type']
        )
        print("DEBUG: Dados divididos com sucesso.")

        # Salvar os dataframes em arquivos CSV
        print(f"Salvando dados de treino em: {train_path}")
        train_df.to_csv(train_path, index=False)

        print(f"Salvando dados de validação em: {val_path}")
        val_df.to_csv(val_path, index=False)

        print(f"Salvando dados de teste em: {test_path}")
        test_df.to_csv(test_path, index=False)

        print("\nResumo da divisão:")
        print(f"Treino: {len(train_df)} amostras")
        print(f"Validação: {len(val_df)} amostras")
        print(f"Teste: {len(test_df)} amostras")
        print("\nDivisão concluída com sucesso!")

    except FileNotFoundError:
        print(f"Erro: O arquivo {csv_path} não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == '__main__':
    # Definir os caminhos dos arquivos
    base_dir = r"C:\Users\IA\Desktop\citology pipeline Train\Dataset\pre-processado"
    master_csv = os.path.join(base_dir, 'labels.csv')
    train_csv = os.path.join(base_dir, 'train_data.csv')
    val_csv = os.path.join(base_dir, 'val_data.csv')
    test_csv = os.path.join(base_dir, 'test_data.csv')

    # Executar a função
    create_and_split_dataset(master_csv, train_csv, val_csv, test_csv)