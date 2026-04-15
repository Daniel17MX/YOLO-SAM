import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
PROJECT_ROOT = r"D:\prueba 2\skin-lesion-detector"
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "HAM10000_metadata.csv")
DIR_PART_1 = os.path.join(PROJECT_ROOT, "data", "raw", "HAM10000_images_part_1")
DIR_PART_2 = os.path.join(PROJECT_ROOT, "data", "raw", "HAM10000_images_part_2")
YOLO_DIR_NO_SAM = os.path.join(PROJECT_ROOT, "data", "yolo_dataset_sin_sam")

CLASS_MAP = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

def main():
    print("=== Generando Dataset SIN SAM (Bounding Box Completo) ===")
    
    if os.path.exists(YOLO_DIR_NO_SAM):
        shutil.rmtree(YOLO_DIR_NO_SAM)
        
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(YOLO_DIR_NO_SAM, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR_NO_SAM, 'labels', split), exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df = df[df['dx'].isin(CLASS_MAP.keys())]
    df_bal = pd.concat([g.sample(min(len(g), 500), random_state=42) for _, g in df.groupby('dx')]).reset_index(drop=True)
    
    train_val, test = train_test_split(df_bal, test_size=0.10, stratify=df_bal['dx'], random_state=42)
    train, val = train_test_split(train_val, test_size=0.22, stratify=train_val['dx'], random_state=42)
    
    train['split'] = 'train'; val['split'] = 'val'; test['split'] = 'test'
    df_final = pd.concat([train, val, test])

    procesadas = 0
    for _, row in df_final.iterrows():
        img_id = row['image_id']
        class_id = CLASS_MAP[row['dx']]
        split = row['split']

        src = os.path.join(DIR_PART_1, f"{img_id}.jpg")
        if not os.path.exists(src):
            src = os.path.join(DIR_PART_2, f"{img_id}.jpg")

        if os.path.exists(src):
            # Copiar imagen
            shutil.copy2(src, os.path.join(YOLO_DIR_NO_SAM, 'images', split, f"{img_id}.jpg"))
            
            # EL SECRETO: Crear caja que cubre el 100% de la imagen (x_centro, y_centro, ancho, alto)
            with open(os.path.join(YOLO_DIR_NO_SAM, 'labels', split, f"{img_id}.txt"), 'w') as f:
                f.write(f"{class_id} 0.500000 0.500000 1.000000 1.000000")
            procesadas += 1

    # Crear el YAML automáticamente (Corrección para Python 3.11)
    ruta_limpia = YOLO_DIR_NO_SAM.replace('\\', '/')
    
    yaml_content = f"""path: {ruta_limpia}
train: images/train
val: images/val
test: images/test
names:
  0: akiec
  1: bcc
  2: bkl
  3: df
  4: mel
  5: nv
  6: vasc"""
    
    with open(os.path.join(YOLO_DIR_NO_SAM, "dataset_sin_sam.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"✓ Listo. {procesadas} imágenes configuradas SIN usar SAM.")

if __name__ == "__main__":
    main()