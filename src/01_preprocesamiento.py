import os
import cv2
import pandas as pd
import torch
import shutil
from sklearn.model_selection import train_test_split
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ─── 1. CONFIGURACIÓN DE RUTAS ───────────────────────────────────────────
PROJECT_ROOT = r"D:\skin-lesion-detector"
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "HAM10000_metadata.csv")
DIR_PART_1 = os.path.join(PROJECT_ROOT, "data", "raw", "HAM10000_images_part_1")
DIR_PART_2 = os.path.join(PROJECT_ROOT, "data", "raw", "HAM10000_images_part_2")

# ⚠️ OBLIGATORIO: Asegúrate de haber descargado el modelo BASE (vit_b)
SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "sam_vit_b_01ec64.pth") 
YOLO_DIR = os.path.join(PROJECT_ROOT, "data", "yolo_dataset")

# ─── 2. CONFIGURACIÓN DE CLASES Y BALANCEO ─────────────────────────────
CLASS_MAP = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
MAX_IMAGENES_POR_CLASE = 500  # Límite estricto para balancear y acelerar el proceso
SEMILLA = 42

def setup_directories():
    """Limpia y crea la estructura de carpetas de YOLO."""
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR) # Limpiamos intentos anteriores
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)

def convert_to_yolo_bbox(bbox, img_width, img_height, class_id):
    x, y, w, h = bbox
    x_center = (x + (w / 2)) / img_width
    y_center = (y + (h / 2)) / img_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w/img_width:.6f} {h/img_height:.6f}"

def get_best_mask(masks, img_area):
    if not masks: return None
    valid_masks = [m for m in masks if m['area'] < (img_area * 0.90)]
    if not valid_masks:
        return min(masks, key=lambda x: x['area'])
    return max(valid_masks, key=lambda x: x['area'])

def balance_and_split_dataset(df):
    """Aplica undersampling y divide el dataset con scikit-learn."""
    print("Balanceando dataset...")
    # Filtrar clases válidas
    df = df[df['dx'].isin(CLASS_MAP.keys())]
    
    # Balancear tomando un máximo de 500 imágenes por cada una de las 7 clases
    df_balanceado = pd.concat([
        grupo.sample(min(len(grupo), MAX_IMAGENES_POR_CLASE), random_state=SEMILLA) 
        for _, grupo in df.groupby('dx')
    ]).reset_index(drop=True)
    
    print(f"Total imágenes a procesar después del balanceo: {len(df_balanceado)}")

    # Split 70% Train, 20% Val, 10% Test conservando proporciones
    train_val, test = train_test_split(df_balanceado, test_size=0.10, stratify=df_balanceado['dx'], random_state=SEMILLA)
    train, val = train_test_split(train_val, test_size=0.22, stratify=train_val['dx'], random_state=SEMILLA)
    
    # Etiquetar cada fila con su carpeta destino para iterar fácilmente
    train = train.copy(); train['split'] = 'train'
    val = val.copy();     val['split'] = 'val'
    test = test.copy();   test['split'] = 'test'
    
    return pd.concat([train, val, test])

def main():
    print("=== Iniciando Pipeline Optimizado ===")
    setup_directories()
    
    df_original = pd.read_csv(CSV_PATH)
    df_final = balance_and_split_dataset(df_original)
    
    print("\nCargando SAM en GPU (Versión Base - Optimizada)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    
   # OPTIMIZACIÓN CRÍTICA PARA RTX 3050 (4GB VRAM)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,       
        points_per_batch=16,      # <--- Esta es la palabra correcta
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92
    )
    procesadas = 0
    errores = 0

    for index, row in df_final.iterrows():
        image_id = row['image_id']
        class_id = CLASS_MAP[row['dx']]
        split = row['split']

        img_path_1 = os.path.join(DIR_PART_1, f"{image_id}.jpg")
        img_path_2 = os.path.join(DIR_PART_2, f"{image_id}.jpg")
        img_path = img_path_1 if os.path.exists(img_path_1) else img_path_2

        if not os.path.exists(img_path):
            errores += 1
            continue

        image = cv2.imread(img_path)
        img_height, img_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MODO INFERENCIA: Ahorra VRAM al no guardar historial de gradientes
        with torch.inference_mode():
            masks = mask_generator.generate(image_rgb)
        
        best_mask = get_best_mask(masks, img_width * img_height)

        if best_mask is None:
            errores += 1
            continue

        yolo_annotation = convert_to_yolo_bbox(best_mask['bbox'], img_width, img_height, class_id)

        # Guardar .jpg y .txt
        shutil.copy2(img_path, os.path.join(YOLO_DIR, 'images', split, f"{image_id}.jpg"))
        with open(os.path.join(YOLO_DIR, 'labels', split, f"{image_id}.txt"), 'w') as f:
            f.write(yolo_annotation)

        procesadas += 1
        
        # Limpieza de basura en GPU y log cada 50 imágenes
        if procesadas % 50 == 0:
            print(f"[+] Procesadas: {procesadas}/{len(df_final)} | Errores: {errores}")
            torch.cuda.empty_cache() # Libera la VRAM asfixiada

    print("\n=== Pipeline Finalizado Exitosamente ===")
    print(f"Listas para entrenar YOLO: {procesadas} imágenes balanceadas.")

if __name__ == "__main__":
    main()