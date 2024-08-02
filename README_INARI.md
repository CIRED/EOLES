# Lancer le modèle sur INARI

## Générer une clé SSH

Sur Windows 8 et antérieurs, utilisez PuTTY. \
Attention à bien générer une clé selon le nouveau chiffrement ed25519. \

Sur macOS/Windows10+, la commande est : ```ssh-keygen -t ed25519``` \
Avec PuTTY, la génération se fait grâce à l'utilitaire 'PuTTYgen'. \

## Connexion à INARI

Envoyez votre clé publique à Florian Leblanc avec le formulaire signé pour être ajouté sur Inari et obtenir un nom d'utilisateur. \
Connexion sous macOS/Windows10+ : ```ssh USER_NAME@inari.centre-cired.fr``` \
Connexion avec PuTTY : utilisez l'utilitaire 'PuTTY'. \

## Gurobi

Le modèle utilise le solver Gurobi, accessible gratuitement pour les chercheurs et bien plus performant que les solvers open-source. Les licenses gratuites étant nominatives, il vous faut cependant générer votre propre license et l'importer sur INARI. \
Si vous ne souhaitez pas utiliser Gurobi, vous pouvez utiliser le solver libre 'cbc'. Il vous faudra en revanche l'installer vous-même. Les performances sont sensiblement moins bonnes que Gurobi (de l'ordre d'un facteur 10). \

### Générer une license académique Gurobi

Vous pouvez générer une license académique sur le site de Gurobi : \
https://www.gurobi.com/downloads/end-user-license-agreement-academic/ \
Il faudra vous créer un compte.
Chaque license est valable un an.
Il n'y a pas de limite au nombre de licenses académiques que vous pouvez générer. \
Chaque licence est associée à une machine et un utilisateur : il n'est donc pas possible d'utiliser la même licence que sur votre ordinateur personnel ou qu'un autre utilisateur sur Inari. \

### Importer la license sur INARI

L'importation de la license sur INARI se fait avec la commande suivante ```grbgetkey``` depuis INARI \
Le script vous demandera où vous souhaitez stocker la license. \
La license étant nomminative, il est recommandé de la placer dans votre home repostitory (```/home/USER_NAME```).
La commande va importer un fichier ```gurobi.lic```. \
Ensuite, ajoutez ces lignes dans votre .bashrc :
  ```
    # gurobi path and licence
  export GUROBI_HOME=/data/software/gurobi1002/linux64
  export PATH=$GUROBI_HOME/bin:$PATH
  export GRB_LICENSE_FILE=/home/user/gurobi.lic
  ```
En remplacant ```user``` avec votre username, ou le chemin d'accès complet si vous en utilisez un autre. \

## Lancer le modèle

Importez le code depuis github avec ```git clone https://github.com/CIRED/EOLES.git``` ou copiez votre propre code depuis votre ordinateur personnel (et pas depuis INARI) avec ```scp -r LOCAL_PATH USER_NAME@inari.centre-cired.fr:/home/USER_NAME/Eoles```, en remplaçant LOCAL_PATH et USER_NAME.
Déplacez-vous dans le dossier où se trouve Eoles (par exemple, ```cd Eoles/```)

Puis lancez le modèle avec : \
```nohup nice -n 10 /data/software/anaconda3/envs/envEOLES/bin/python example.py > outputs/example.log 2>&1 &``` \
L'utilisation du mot clé ```nice -n 10``` est importante pour ne pas surcharger le serveur de calcul. \
Vous pouvez également modifier le solver, mais seul gurobi est pour le moment disponible sur le serveur. \
La résolution prend une centaine de secondes environ pour une année. \
Attention, la version 19 ans prend environ 90 minutes et utilise une part importante des ressources du serveur, à utiliser ponctuellement seulement. \

Si une erreur de license s'affiche (nom d'utilisateur incorrect par exemple), deux raisons possibles :
- Vous avez oublié d'indiquer le chemin vers votre license. Reprenez cette section depuis le début.
- Votre licence est expirée. Il faut en créer une autre. Reprenez toutes les étapes depuis la section 'Générer une license académique Gurobi'. \

## Quelques commandes linux utiles

La commande ```cd``` permet de se déplacer dans l'arborescence des fichiers. ```cd FOLDER``` permet d'ouvrir le dossier FOLDER, ```cd ..``` permet de revenir au dossier parent. \
La commande ```ls``` permet d'afficher la liste des fichiers présents dans le répertoire actuel. ```ls FOLDER``` permet d'afficher la liste des fichiers présents dans le répertoire ```FOLDER```. \
La commande ```cat FILE``` permet d'afficher dans la console le contenu d'un fichier simple ```FILE``` (.txt, .csv,...). \

## Gestion des fichiers

L'interface en ligne de commande n'étant pas très pratique pour exploiter les résultats, il peut être préférable de copier les résultats en local sur votre ordinateur. \

Sous macOS/Windows10+, utilisez la commande ```scp -r USER_NAME@inari.centre-cired.fr:/home/USER_NAME/Eoles/outputs LOCAL_PATH``` dans le terminal de votre ordinateur (et pas sur INARI) (le chemin pour atteindre le dossier Eoles sur Inari peut être différent en fonction de vos choix). \
Vous pouvez spécifier le lieu où stocker les données en remplaçant ```LOCAL_PATH``` par le chemin souhaité. \
L'argument ```-r``` permet de copier récursivement tous les sous-fichiers et sous-dossiers. Il n'est pas nécessaire dans le cas d'un fichier seul. \

Si à l'inverse vous souhaitez mettre à jour le code ou les données d'entrée, vous pouvez copier vos fichiers en inversant les deux arguments (toujours sur le terminal de votre machine) : \
```scp FILE_NAME USER_NAME@inari.centre-cired.fr:/...``` pour un fichier unique ; \
```scp -r FOLDER_NAME USER_NAME@inari.centre-cired.fr:/...``` pour un dossier. \

Sous Windows8 et antérieurs, il est conseillé d'utiliser WinSCP pour la gestion des fichiers. \
L'outil récupère automatiquement les infos paramétrées sur PuTTY pour fournir une interface graphique permettant de copier les fichiers dans les deux sens, pour mettre à jour le code, les données d'entrée ou pour récupérer les résultats. \
Si dans ce cas vous tentez d'écraser un fichier présent sur le serveur avec un nouveau et qu'une erreur de permission d'affiche : cliquez sur le bouton 'ignorer' et le fichier sera importé.
