# Compte-rendu : Pipelines MLOps avec ZenML (TP5)

## 1. Questions de compréhension
- **À quoi servent les décorateurs `@step` et `@pipeline` ?** : Ils transforment des fonctions Python classiques en composants de pipeline ZenML. `@step` définit une unité de calcul (versionnage, caching, stockage des artefacts), tandis que `@pipeline` définit l'enchaînement de ces étapes.
- **Quels sont les artefacts principaux produits ?** : 
  - `data_loader` : Fichier de configuration YAML du dataset.
  - `trainer` : Le modèle entraîné (fichier `.pt`).
  - `evaluator` : Métriques de performance (mAP50).
- **Qu'est-ce qui est stocké où ?** :
  - **ZenML Server** : Métadonnées du pipeline, versionnage du code, historique des runs.
  - **MinIO (S3)** : Les artefacts ZenML (objets sérialisés, modèles).
  - **MLflow** : Métriques d'entraînement (pertes), paramètres, et courbes de performance.

## 2. Analyse des Runs de Pipeline
### Tableau Comparatif
| Pipeline Run | Image Size | lr0 | mAP50 | Statut |
|--------------|------------|-----|-------|--------|
| Baseline     | 320        | 0.005|       |        |
| Grid Run 1   | 416        | 0.005|       |        |
| Grid Run 2   | 320        | 0.01 |       |        |
| Grid Run 3   | 416        | 0.01 |       |        |

## 3. Discussion
- **Choix du run pour le Staging** : ...
- **Avantages de ZenML vs Script brut (TP4)** : Reproductibilité garantie, caching des étapes inchangées, centralisation des artefacts, et séparation claire entre code et infrastructure.
- **Éléments essentiels pour CI/CD/CT** : Trigger automatique sur changement de données/code, validation automatique des métriques avant déploiement.

## 4. Décision Finale
**Promotion** : [Oui / Non]
**Pourquoi** : ...
