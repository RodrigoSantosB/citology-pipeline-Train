import os
import pandas as pd

def create_label_file(dataset_path, output_csv):
    """
    Cria um arquivo CSV de rótulos a partir de um diretório de imagens organizado.

    Args:
        dataset_path (str): O caminho para o diretório do conjunto de dados.
        output_csv (str): O caminho para o arquivo CSV de saída.
    """
    data = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                data.append({"Image_path": image_path, "Label": label})
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Arquivo de rótulos criado em: {output_csv}")

if __name__ == "__main__":
    dataset_directory = "C:\\Users\\IA\\Desktop\\citology pipeline Train\\Dataset\\pre-processado\\3 Classes"
    output_csv_file = "C:\\Users\\IA\\Desktop\\citology pipeline Train\\Dataset\\pre-processado\\labels.csv"
    create_label_file(dataset_directory, output_csv_file)