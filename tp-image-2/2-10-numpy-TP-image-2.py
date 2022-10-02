# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
#     cell_metadata_json: true
#     notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version,
#       -jupytext.text_representation.format_version, -language_info.version, -language_info.codemirror_mode.version,
#       -language_info.codemirror_mode, -language_info.file_extension, -language_info.mimetype,
#       -toc
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#   nbhosting:
#     title: suite du TP simple avec des images
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %%
from IPython.display import HTML
HTML(url="https://raw.githubusercontent.com/ue12-p22/python-numerique/main/notebooks/_static/style.html")



# %% [markdown]
# # suite du TP simple avec des images
#
# merci à Wikipedia et à stackoverflow
#
# **le but de ce TP n'est pas d'apprendre le traitement d'image  
# on se sert d'images pour égayer des exercices avec `numpy`  
# (et parce que quand on se trompe ça se voit)**

# %%
import numpy as np
from matplotlib import pyplot as plt

# %% [markdown] {"tags": ["framed_cell"]}
# **notions intervenant dans ce TP**
#
# sur les tableaux `numpy.ndarray`
#
# * `reshape()`, tests, masques booléens, *ufunc*, agrégation, opérations linéaires sur les `numpy.ndarray`
# * les autres notions utilisées sont rappelées (très succinctement)
#
# pour la lecture, l'écriture et l'affichage d'images
#
# * utilisez `plt.imread`, `plt.imshow`
# * utilisez `plt.show()` entre deux `plt.imshow()` dans la même cellule
#
# **note**
#
# * nous utilisons les fonctions de base sur les images de `pyplot` par souci de simplicité
# * nous ne signifions pas là du tout que ce sont les meilleures  
# par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
# alors que la fonction `save` de `PIL` le permet
#
# * vous êtes libres d'utiliser une autre librairie comme `opencv`  
#   si vous la connaissez assez pour vous débrouiller (et l'installer), les images ne sont qu'un prétexte
#
# **n'oubliez pas d'utiliser le help en cas de problème.**

# %% [markdown]
# ## Création d'un patchwork

# %% [markdown]
# 1. Le fichier `rgb-codes.txt` contient une table de couleurs:
# ```
# AliceBlue 240 248 255
# AntiqueWhite 250 235 215
# Aqua 0 255 255
# .../...
# YellowGreen 154 205 50
# ```
# Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
# Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.
# <br>
#
# 1. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
# `'Red'`, `'Lime'`, `'Blue'`
# <br>
#
# 1. Faites une fonction `patchwork` qui  
#
#    * prend une liste de couleurs et la structure donnant le code des couleurs RGB
#    * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
#    * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
#    si besoin de compléter l'image mettez du blanc  
#    (`numpy.indices` peut être utilisé)
# <br>
# <br>   
# 1. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.
# <br>
#
# 1. Sélectionnez toutes les couleurs à base de blanc et affichez leur patchwork  
# même chose pour des jaunes  
# <br>
#
# 1. Appliquez la fonction à toutes les couleurs du fichier  
# et sauver ce patchwork dans le fichier `patchwork.jpg` avec `plt.imsave`
# <br>
#
# 1. Relisez et affichez votre fichier  
#    attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels
#    
# vous devriez obtenir quelque chose comme ceci
# <img src="patchwork-all.jpg" width="200px">

# %%
# 1

# Je choisis d'implémenter la table de couleurs par un dictionnaire dont les clés sont les noms des couleurs.

dico = {}
colours = [] # Future liste des noms des couleurs

fichier = open("rgb-codes.txt", "r")
for line in fichier:
    l = line.split()     # Liste dont le 1er élément est le nom de la couleur, et les 3 suivants sont les valeurs RGB
    dico[l[0]] = l[1:]   # Clé : nom de la couleur, valeur : code RGB
    colours.append(l[0]) # Nom de la couleur
fichier.close()

# %%
# 2

for color in "Red", "Lime", "Blue":
    print("Le code RGB de la couleur {}".format(color), "est", dico[color])


# %%
# 3

def patchwork(liste, dico):
    """Entrée : liste de couleurs, dictionnaire associant ces noms (les clés) à leur codage RGB
       Sortie : tableau numpy qui représente un patchwork de ces couleurs"""
    n = len(liste)
    p = 10*int( np.ceil( np.sqrt(n) ) )                    # 10p est une taille de côté adaptée pour un cadre carré (calculs...)
    tab = np.full((p, p, 3), 255 , dtype = np.uint8) # Le cadre est initialisé avec la couleur blanche 
    i, j = 0, 0                                            # Je parcours les carrés en faisant varier deux indices
    for colour in liste:
        tab[10*i:10*(i+1),10*j:10*(j+1)] = dico[colour]    # On remplit le carré
        if j >= p//10:                                     # Si l'on est en bout de ligne...
            j = 0                                         
            i += 1                                         # On revient au début de la suivante
        else:
            j += 1                                         # Sinon, on passe à la colonne suivante
    return tab


# %%
# 4

# Les couleurs sélectionnées ne sont pas nécessairement 2 à 2 distinctes.

n = np.random.randint(10, len(colours))      # Nombre de couleurs à sélectionner
liste = np.random.choice(colours, size=n)    # Liste des noms de couleurs sélectionnés aléatoirement

# On affiche le patchwork créé avec les lignes suivantes :

plt.imshow(patchwork(liste,dico)) ;

# %%
# 5

# On sélectionne toutes les couleurs dont le nom contient "White", puis on affiche le patchwork associé.

w = np.flatnonzero(np.core.defchararray.find(colours,'White')!=-1)  # Liste d'indices
W = np.array([colours[k] for k in w])                               # Liste de couleurs
plt.imshow(patchwork(W,dico))
plt.show()
print("Il y a", len(w), "couleurs sur le patchwork ci-dessus")

# Même chose pour "Yellow".

w = np.flatnonzero(np.core.defchararray.find(colours,'Yellow')!=-1)  # Liste d'indices
W = np.array([colours[k] for k in w])                                # Liste de couleurs
plt.imshow(patchwork(W,dico))
plt.show()
print("Il y a", len(w), "couleurs sur le patchwork ci-dessus")

# %%
# 6

# On applique patchwork à toutes les couleurs du fichier.

t = patchwork(colours,dico)
plt.imshow(t)
plt.show()
plt.imsave('patchwork.jpg', t)

# %% [markdown]
# ## Somme des valeurs RGB d'une image

# %% [markdown]
# 0. Lisez l'image `les-mines.jpg`
#
# 1. Créez un nouveau tableau `numpy.ndarray` en sommant **avec l'opérateur `+`** les valeurs RGB des pixels de votre image  
#
# 2. Affichez l'image (pas terrible), son maximum et son type
#
# 3. Créez un nouveau tableau `numpy.ndarray` en sommant **avec la fonction d'agrégation `np.sum`** les valeurs RGB des pixels de votre image
#
# 4. Affichez l'image, son maximum et son type
#
# 5. Pourquoi cette différence ? Utilisez le help `np.sum?`
#
# 6. Passez l'image en niveaux de gris de type entiers non-signés 8 bits  
# (de la manière que vous préférez)
#
# 7. Remplacez dans l'image en niveaux de gris,   
# les valeurs >= à 127 par 255 et celles inférieures par 0  
# Affichez l'image avec une carte des couleurs des niveaux de gris  
# vous pouvez utilisez la fonction `numpy.where`
#
# 8. avec la fonction `numpy.unique`  
# regardez les valeurs différentes que vous avez dans votre image en noir et blanc

# %%
# 0
im = plt.imread('les-mines.jpg')
print(im[0,0,0])

# %%
# 1 & 2

tab = im[:,:,0] + im[:,:,1] + im[:,:,2]
plt.imshow(tab) ;
print(np.max(tab), tab.dtype)

# %% [markdown]
# La nouvelle image a le même type que l'originale. Il n'y a pas eu d'adaptation en cas de dépassement de la valeur 255.

# %%
# 3 & 4

tab = im.sum(axis=2)
plt.imshow(tab) ;
print(np.max(tab), tab.dtype)

# %% [markdown]
# Cette fois-ci, la nouvelle image s'est automatiquement convertie afin de prendre en compte les valeurs qui dépassent 255.

# %%
# 5

help(np.sum)

# %% [markdown]
# La documentation de la fonction sum indique bien "the type of the output
#         values will be cast if necessary."

# %%
im.shape

# %%
# 6 - On passe l'image en gris en prenant la moyenne des valeurs RGB de ses pixels par exemple.

tab = (im/3).sum(axis=2, dtype=np.uint8)  # La fonction mean ne fonctionne pas ici car en ajoutant les pixels on peut dépasser 255
tab.dtype

# %%
# 7

a = np.where(tab>=127)        # Indices des pixels de valeur supérieure à 127
tab[:,:] = 0                  # On réinitialise le tableau à zéro...
tab[a] = 255                  # Et on colorie en noir les pixels sélectionnés plus tôt.
plt.imshow(tab, cmap='Greys') ;

# %%
# 8

np.unique(tab)


# %% [markdown]
# On n'a bien que deux valeurs dans l'image précédente.

# %% [markdown]
# ## Image en sépia

# %% [markdown]
# Pour passer en sépia les valeurs R, G et B d'un pixel  
# (encodées ici sur un entier non-signé 8 bits)  
#
# 1. on transforme les valeurs $R$, $G$ et $B$ par la transformation  
# $0.393\, R + 0.769\, G + 0.189\, B$  
# $0.349\, R + 0.686\, G + 0.168\, B$  
# $0.272\, R + 0.534\, G + 0.131\, B$  
# (attention les calculs doivent se faire en flottants pas en uint8  
# pour ne pas avoir, par exemple, 256 devenant 0)  
# 1. puis on seuille les valeurs qui sont plus grandes que `255` à `255`
# 1. naturellement l'image doit être ensuite remise dans un format correct  
# (uint8 ou float entre 0 et 1)

# %% [markdown]
# **Exercice**
#
# 1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
# la fonction `numpy.dot` doit être utilisée (si besoin, voir l'exemple ci-dessous) 
#
# 1. Passez votre patchwork de couleurs en sépia  
# Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso
# 2. Passez l'image `les-mines.jpg` en sépia   

# %%
# 1

def sepia(image):
    tab = np.array([[0.393, 0.349, 0.272],   # Rouge
                    [0.769, 0.686, 0.534],   # Vert
                    [0.189, 0.168, 0.131]])  # BLeu
    im_float = np.array(image,dtype = np.float64)
    image_sepia = np.dot(im_float,tab)
    indices = np.where(image_sepia > 255)
    image_sepia[indices] = 255
    return np.array(image_sepia, dtype=np.uint8)


# %%
# 2

plt.imshow(sepia(patchwork(colours,dico))) ;

# %%
# 3

plt.imshow(sepia(im))

# %% {"scrolled": true}
# INDICE:

# exemple de produit de matrices avec `numpy.dot`
# le help(np.dot) dit: dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])

i, j, k, m, n = 2, 3, 4, 5, 6
A = np.arange(i*j*k).reshape(i, j, k)
B = np.arange(m*k*n).reshape(m, k, n)

C = A.dot(B)
# or C = np.dot(A, B)

A.shape, B.shape, C.shape

# %% [markdown]
# ## Exemple de qualité de compression

# %% [markdown]
# 1. Importez la librairie `Image`de `PIL` (pillow)   
# (vous devez peut être installer PIL dans votre environnement)
# 1. Quelle est la taille du fichier 'les-mines.jpg' sur disque ?
# 1. Lisez le fichier 'les-mines.jpg' avec `Image.open` et avec `plt.imread`  
#
# 3. Vérifiez que les valeurs contenues dans les deux objets sont proches
#
# 4. Sauvez (toujours avec de nouveaux noms de fichiers)  
# l'image lue par `imread` avec `plt.imsave`  
# l'image lue par `Image.open` avec `save` et une `quality=100`  
# (`save` s'applique à l'objet créé par `Image.open`)
#
# 5. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
# Que constatez-vous ?
#
# 6. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence  

# %%
# 1

import PIL.Image as Image

# %%
# 2

import os
print(os.stat('les-mines.jpg').st_size, "octets")

# %%
# 3

image = Image.open('les-mines.jpg')
file = plt.imread('les-mines.jpg')

# %%
# 4

np.allclose(file,im)

# %%
# 5

plt.imsave('les-mines-plt.jpg', file)
image.save('les-mines-Image.jpg', quality=100)

# %%
# 6

print("le fichier créé par imsave pèse", os.stat('les-mines-plt.jpg').st_size, "octets")
print("le fichier créé par file.save pèse", os.stat('les-mines-Image.jpg').st_size, "octets")

# %% [markdown]
# On constate que les deux images ont été compressées par rapport à l'originale.  
# En particulier, 'les-mines-plt.jpg' a été bien plus compressée que 'les-mines-Image.jpg'.

# %%
# 7

image2 = Image.open('les-mines-Image.jpg')
file2 = plt.imread('les-mines-plt.jpg')
tab = abs( image2 - file2)
plt.imshow(tab) ;
