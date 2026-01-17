# Compte Rendu : TP4 Experiment Tracking avec MLflow - YOLO Tiny

Ce document sert de rapport final pour le TP4 sur la détection d'objets (YOLOv8) avec suivi d'expériences (MLflow).

## 1. Contexte et Objectifs
L'objectif de ce TP était de mettre en place un pipeline complet d'entraînement pour la détection de personnes dans un dataset filtré (Tiny COCO), en automatisant le suivi des hyperparamètres et des métriques via **MLflow** et **DVC**.

## 2. Architecture Technique
- **Modèle** : YOLOv8 Nano (Ultralytics).
- **Dataset** : COCO128 filtré pour ne garder que la classe "personne".
- **Infrastructure** : 
  - **MLflow Tracking** pour les logs.
  - **MinIO** comme serveur d'artefacts (S3-compatible).
  - **Docker Compose** pour orchestrer les services.
  - **DVC** pour le versionnement des données.

## 3. Analyse de la Grille d'Expériences
Nous avons testé plusieurs combinaisons :
- **Tailles d'image (imgsz)** : 320, 416.
- **Learning Rates (lr0)** : 0.005, 0.01.
- **Époques** : 3 (pour des tests rapides).

### Meilleurs Résultats (Top 3)
| Run Name | Image Size | LR | mAP50 | Precision | Recall |
|----------|------------|----|-------|-----------|--------|
| `yolov8n_e3_sz416_lr0.005_s42` | 416 | 0.005 | **0.2573** | 0.0070 | 0.6774 |
| `yolov8n_e3_sz320_lr0.01_s42` | 320 | 0.01 | 0.1521 | 0.0073 | 0.6452 |
| `yolov8n_e3_sz320_lr0.005_s42` | 320 | 0.005 | 0.1521 | 0.0073 | 0.6452 |

**Observation** : L'augmentation de la taille d'image à 416 a significativement amélioré le mAP50.

## 4. Visualisations
Voici les graphiques générés durant l'entraînement du meilleur modèle :

### Résultats d'Entraînement
![Results](images/results.png)

### Courbe Précision-Rappel (PR Curve)
![PR Curve](images/BoxPR_curve.png)

### Matrice de Confusion
![Confusion Matrix](images/confusion_matrix.png)

### Prédictions sur le jeu de validation
![Predictions](images/val_batch0_pred.jpg)

## 5. Décision de Promotion
**Candidat promu** : `yolov8n_e3_sz416_lr0.005_s42` (ID: `4f44f0a84b8949c382ced7d9ca916a6c`)

**Justification** :
- Meilleur mAP50 (0.257).
- Bonne stabilité des pertes (loss) malgré le faible nombre d'époques.
- Taille d'image 416 permet de mieux capter les petits objets (Tiny Person).

**Décision** : **OUI** (Promotion en production/Staging dans le Model Registry).

## 6. Conclusion
Le pipeline est fonctionnel. L'intégration de MLflow permet une analyse comparative immédiate des performances. Pour la suite, un entraînement sur 50+ époques avec le dataset complet permettrait d'atteindre des métriques de précision plus robustes.

---
*Auteur : Nadhir*
*Date : 17 Janvier 2026*

# TP5 : Pipelines MLOps avec ZenML et MLflow

Ce document complète le rapport avec l'implémentation de l'orchestration via **ZenML**.

## 1. Objectifs du TP5
- Encapsuler la logique d'entraînement YOLOv8 dans un pipeline orchestré.
- Utiliser des **étapes (Steps)** ZenML pour le chargement des données, l'entraînement et l'évaluation.
- Intégrer MLflow en tant que **Experiment Tracker** au sein de la Stack ZenML.

## 2. Configuration de la Stack
Nous avons configuré une stack locale ZenML nommée `local_mlflow_stack` :
- **Orchestrator** : Default (Local)
- **Artifact Store** : Default (Local)
- **Experiment Tracker** : `local_mlflow_tracker` (MLflow via URI locale `./mlruns`)

## 3. Structure du Pipeline
Le pipeline `yolo_training_pipeline` est composé de 3 étapes :
1. **`data_loader`** : Récupère le chemin du fichier YAML de configuration.
2. **`trainer`** : Entraîne le modèle YOLOv8 et logue les paramètres et le modèle dans MLflow.
3. **`evaluator`** : Valide le modèle sur le jeu de test et renvoie le mAP50.

## 4. Résultats des Exécutions (Grid Search)
Quatre runs ont été exécutés via le script `run_yolo_pipeline_grid.py` en variant `imgsz` (320, 416) et `lr0` (0.005, 0.01).

| Run ZenML | Image Size | LR | mAP50 (Evaluator) | Status |
|-----------|------------|----|-------------------|--------|
| Run #1    | 320        | 0.005 | 0.152 | Success (Cached) |
| Run #2    | 416        | 0.005 | **0.256** | Success |
| Run #3    | 320        | 0.01  | 0.152 | Success |
| Run #4    | 416        | 0.01  | **0.256** | Success |

**Note** : ZenML gère automatiquement le cache, ce qui permet d'éviter de ré-entraîner les combinaisons déjà testées si les entrées n'ont pas changé.

## 5. Conclusion TP5
L'utilisation de ZenML permet de standardiser le workflow MLOps. La séparation en étapes facilite la maintenance et le changement d'infrastructure (par exemple, passer d'un entraînement local à un entraînement sur Kubernetes ou cloud sans changer le code métier).

