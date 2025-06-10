import os
import csv
from deepface import DeepFace

input_folder = "dataset/real/"
output_csv = "results.csv"

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'gender', 'race'])

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        try:
            objs = DeepFace.analyze(img_path=img_path, actions=['gender', 'race'])
            if isinstance(objs, list):
                obj = objs[0]
            else:
                obj = objs
            writer.writerow([
                img_path,
                obj['dominant_gender'],
                obj['dominant_race']
            ])
        except Exception as e:
            print(f"Errore nell'analisi di {img_path}: {e}")