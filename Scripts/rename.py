# renomear todas as imagens em um diretório adicionando um prefixo "koi_"
import os
def rename_images_in_directory(directory_path, prefix="sup_"):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            old_file_path = os.path.join(directory_path, filename)
            new_file_name = prefix + filename
            new_file_path = os.path.join(directory_path, new_file_name)
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} to {new_file_path}')


# Exemplo de uso
directory = r'C:\Users\IA\Desktop\citology pipeline Train\Dataset\3 Classes\Tile\Normal\im_Superficial-Intermediate\im_Superficial-Intermediate'
rename_images_in_directory(directory)