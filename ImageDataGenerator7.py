import os
import PIL
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from glob import glob

# Chemin vers le dossier contenant les images originales
input_directory = 'C:/Users/Oumayma Aoun/Desktop/ProjetDataMining/DATA/train/happy'

# Chemin vers le dossier où enregistrer les images augmentées
output_directory = 'C:/Users/Oumayma Aoun/Desktop/ProjetDataMining/DATA/happy'

# Initialiser l'ImageDataGenerator avec les transformations souhaitées pour l'augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255  # Assurez-vous de conserver la mise à l'échelle ici si elle est utilisée dans le modèle principal
)

# Charger les chemins des fichiers d'images à augmenter
image_files = glob(os.path.join(input_directory, '*.jpg'))

# Nombre actuel d'images dans le dossier de sortie
current_image_count = len(glob(os.path.join(output_directory, '*.jpg')))

# Nombre d'images restant à créer pour atteindre 8000
remaining_images = 9000 - current_image_count

# Si le nombre d'images actuel est inférieur à 8000, augmentez le nombre d'images
if remaining_images > 0:
    # Boucle sur chaque image et générer des images augmentées
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            img = img.convert('L')  # Convertir l'image en niveaux de gris
            img_array = img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape)  # Ajouter une dimension pour le lot (batch)

            # Générer des images augmentées et les enregistrer dans le répertoire de sortie
            i = 0
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_directory, save_prefix='aug', save_format='jpg'):
                i += 1
                current_image_count += 1  # Mettre à jour le nombre actuel d'images
                if current_image_count >= 9000:  # S'assurer qu'on ne dépasse pas 8000 images
                    break
                if i >= remaining_images:  # Arrêter si on atteint le nombre nécessaire d'images par image originale
                    break
        except (OSError, PIL.UnidentifiedImageError) as e:
            print(f"Erreur lors du traitement de l'image {img_path}: {e}")
            continue