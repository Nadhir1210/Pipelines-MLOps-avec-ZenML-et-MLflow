# TP4 : Experiment Tracking avec MLflow - YOLO Tiny

Ce projet met en œuvre le suivi d'expériences pour la détection d'objets à l'aide de YOLOv8n et MLflow.

## Structure du Projet

- images/ : Captures des résultats d'entraînement (courbes de perte, matrices de confusion).
- data/ : Contient le dataset COCO128 filtré pour la classe "person".
- reports/ : Gabarit de décision pour la promotion du modèle.
- scripts/ : Scripts pour lancer une grille d'expériences.
- src/ : Code source pour l'entraînement et les utilitaires.
- tools/ : Script de génération du mini-dataset.

## Étapes Réalisées

1. Preparation de l'environnement : Creation d'un venv et installation des dependances.
2. Generation du dataset : Transformation de COCO128 en un dataset ultra-leger (1 classe person).
3. Infrastructure de Tracking : Lancement de MLflow et MinIO via Docker Compose.
4. Baseline : Execution d'un premier entrainement YOLOv8n sur 3 epoques.
5. Grille d'experiences : Lancement de plusieurs runs avec variations de la taille d'image (imgsz) et du taux d'apprentissage (lr0).

## Résultats

Les graphiques de performance se trouvent dans le dossier [images/](images/).

Pour visualiser les runs et comparer les métriques (mAP, précision, rappel), accédez à l'interface MLflow : http://localhost:5000.

---
*Auteur : Nadhir*
