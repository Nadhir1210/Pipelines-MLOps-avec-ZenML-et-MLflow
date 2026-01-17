# Décision de promotion — TP MLflow (CV YOLO Tiny)

## Objectifs et contraintes
- **Objectif principal** : Maximiser le mAP@50 pour la détection de personnes.
- **Contraintes** : Le modèle doit rester léger (YOLOv8 Nano) pour une inférence sur mobile/edge.

## Candidat promu
- **Run name** : `yolov8n_e3_sz416_lr0.005_s42`
- **ID** : `4f44f0a84b8949c382ced7d9ca916a6c`
- **Paramètres clés** : 
  - epochs: 3
  - imgsz: 416
  - lr0: 0.005
  - seed: 42
- **Métriques** : 
  - mAP@50: 0.2573
  - precision: 0.0070
  - recall: 0.6774

## Comparaison (résumé)
- **Alternative A (imgsz=320)** : Le mAP chute à 0.152. Moins performant pour détecter les petits objets.
- **Alternative B (lr0=0.01)** : La perte est plus instable sur les premières époques.
- **Observations** : La taille de 416 est le facteur le plus déterminant pour la performance sur ce dataset.

## Risques et mitigations
- **Risque 1 (Faible Précision)** → Mitigation : Entraîner sur plus de 100 époques.
- **Risque 2 (Dataset trop petit)** → Mitigation : Ajouter des images COCO supplémentaires ou faire de l'augmentation de données.

## Décision
- **Promouvoir** : **OUI**
- **Pourquoi** : C'est le modèle le plus équilibré et performant de la grille d'expériences actuelle.
- **Étapes suivantes** : Enregistrement dans le **MLflow Model Registry** et passage au stage "Production".
