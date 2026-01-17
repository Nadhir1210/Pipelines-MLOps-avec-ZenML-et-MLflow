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
