# Eoles

Eoles est un modèle d'optimisation de l'investissement et de l'exploitation du système énergétique en France cherchant à minimiser le coût total tout en satisfaisant la demande en énergie. \
Voici une présentation d'une version antérieure du modèle : _http://www.centre-cired.fr/quel-mix-electrique-optimal-en-france-en-2050/_ \

---

### Lancer le modèle avec Pyomo

---

#### **Installation des dépendances**

Pour pouvoir lancer le modèle vous aurez besoin d'installer certaines dépendances dont ce programme à besoin pour fonctionner :

* **Python** :
Python est un langage de programmation interprété, utilisé avec Pyomo il va permettre de modéliser Eoles. \
Vous pouvez télécharger la dernière version sur le site dédié : *https://www.python.org/downloads/* \
Ensuite il vous suffit de l'installer sur votre ordinateur. \
Si vous comptez installer Conda ou si vous avez installé Condé sur votre ordinateur, Python à des chances d'être déjà installé.
Le modèle nécessite python3. Nous recommendons d'utiliser une version utlérieure à python3.10 (les versions antérieures à 3.9 n'ont pas été testées)

* **Conda** ou **Pip** selon votre préférence :
Conda et Pip sont des gestionnaires de paquets pour Python.
    * **Conda** \
    Vous pouvez retrouver toutes les informations nécéssaires pour télécharger et installer Conda ici: \
    _https://docs.conda.io/projects/conda/en/latest/user-guide/install/_ \
    __Attention à bien choisir la version de Conda en accord avec la version de Python !__ \
    Vous pouvez installer Miniconda qui est une version minimale de Conda,\
    cela vous permettra de ne pas installer tous les paquets compris dans Conda, \
    mais seulement ceux qui sont nécéssaires.
    * **Pip** \
    Vous pouvez retrouver toutes les informations nécéssaires pour télécharger et installer Pip  ici : \
    _https://pip.pypa.io/en/stable/installing/_ \
    Pip est également installé si vous avez installé Conda.

* Installer les dépendances avec **Conda**:
Déplacez-vous jusqu'au dossier de votre choix
Créer l'environnement et installer les dépendances: ```conda env create -f environment.yml```
Activer l'environnement : ```conda activate envEOLES```
Si vous souhaitez utiliser Jupyter Notebook :
Utiliser ```conda install -c anaconda ipykernel``` and ```python -m ipykernel install --user --name=envEOLES```
L'environnement sera alors disponible dans la liste des kernel.

* Installer les dépendances avec **Pip**:
Créer un environnement virtuel: ```python -m venv envEOLES```
Si vous tuilisez un autre nom pour l'environnement et avez prévu d'envoyer vos modifications vers le github, souvenez-vous d'exclure le dossier de l'environnement des commit.
Activer l'environnement :
Windows : ```envEOLES\Scripts\activate```
macOS/Linux: ```source envEOLES/bin/activate```
Installer les dépendances : ```pip install -r requirements.txt```

* **Solveur** :
Le solveur que ce modèle utilise est le solveur Gurobi, bien plus rapide que le solveur Cbc. \
Des licenses gratuites sont mises à dispositions pour les chercheurs et étudiants.
Pour utiliser Gurobi :
    * Se créer un compte et télécharger Gurobi Optimizer ici : _https://www.gurobi.com/downloads/_
    * Demander une license académique gratuite : _https://www.gurobi.com/downloads/end-user-license-agreement-academic/_
    * Utiliser la commande ```grbgetkey``` pour importer sa license, comme indiquer sur celle-ci.

#### **Récupération du code :**

Si vous n'avez pas installé Git sur votre ordinateur, vous pouvez téléchargez le dossier de Github ou Zenodo, dans le format que vous souhaitez.
Sinon, vous pouvez récupérer les fichiers de ce GitLab à travers la commande :\
```git clone https://github.com/CIRED/EOLES.git```\
Un dossier sera créé dans le répertoire courant avec tout les fichiers contenus dans ce GitLab. \

#### **Utilisation du modèle :**

Le modèle Eoles est écrit sous forme de classe contenue dans le package ```modelEoles.py```.\
Plusieurs fonctions utilitaires (pour l'initialisation ou pour générer des graphiques) sont incluses dans le package ```utils.py```.\
Pour utiliser le modèle, il suffit d'importer la classe ModelEOLES, d'en créer un instance à partir du fichier de configuration voulu, et d'utiliser les différentes méthodes pour construire et résoudre le modèle et extraire les résultats. Un fichier ```.py``` d'exemple est fourni.


---

### Données d'entrées

---

Des examples de données d'entrée sont fournies dans le dossier **inputs**.\
Le chemin d'accès à chaque fichier de donnée peut être modifié dans les fichier de configuration.\
Des profils de demande et de production supplémentaires peuvent être trouvés dans [ce dépôt Zenodo](https://doi.org/10.5281/zenodo.13124746)

Le format attendu pour les données d'entrée est clarifié pour chaque type (constante ou profil) dans la fonction utilitaire associée dans ```utils.py```.



La version originale de ce README a été écrite par Quentin Bustarret
