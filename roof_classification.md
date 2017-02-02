
Cet article présente comment utiliser un réseau de neurones pré-entrainé pour résoudre un problème de classification d’images (tâche différente que celle pour laquelle le réseau a été préalablement entrainé). Dans notre cas, l’objectif est de classer des images satellites de toits en 4 catégories : orientation EST/OUEST, orientation NORD/SUD, toit plat, et catégorie “Autre”. Il est important de préciser une nouvelle fois que cette tâche de classification est différente de celle pour laquelle le réseau de neuronnes que nous allons utiliser a été entrainé.

La librairie de réseau de neurones que nous utiliserons se nomme MxNET. Elle est disponible à la fois sous Linux et Windows, et possède (entre autre) un wrapper en Python et en R. Dans cet article le code a été exécuté sous Ubuntu avec le wrapper Python.

# Installation de la librairie sous Ubuntu
Commencer par exécuter les commandes suivantes dans un terminal afin d'installer les dépendances :

{% highlight javascript %}
sudo apt-get update
sudo apt-get install -y build-essential git libatlas-base-dev libopencv-dev
{% endhighlight %}

Ensuite, clonez le dépot Github de MxNet grâce à la commande ci-après :
{% highlight javascript %}
git clone --recursive https://github.com/dmlc/mxnet
{% endhighlight %}

Puis pour compiler la librairie : 
{% highlight javascript %}
cd mxnet; make -j$(nproc)
{% endhighlight %}

Si vous disposez d'un GPU assez récent et que vous avez déjà installé les librairies CUDA suffisantes, vous pouvez compiler la librairie avec la commande ci-dessous. Cela permettra à MxNet de tirer partie de votre GPU.
{% highlight javascript %}
cd mxnet;make -j4 USE_CUDA=1
{% endhighlight %}

Enfin, pour installer le package Python : 
{% highlight javascript %}
cd python; sudo python setup.py install
{% endhighlight %}

# Extraction des features

La première étape consiste à utiliser un réseau de neurones déjà entrainé afin d'extraire des features de nos images. Pour cela nous utilisons un réseau de neurones nommés Inception-BN. BN vient du fait qu'il a été entrainé en utilisant une technique ce nomment batch normalization.
Il a été entrainé sur le dataset ILSVRC2012 sur lequel il atteint une accuracy top-1 de 72.5% et top-5 de 90.2%.


```python
# imports relatifs a l'extraction de features via le CNN
import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
from skimage.util import img_as_float
from skimage.transform import resize
import pandas as pd
```


```python
def preprocess_image(path, show_img=False, mean_img=None):
    # Loads the image
    img = io.imread(path)
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # Resizes image to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    if show_img:
        io.imshow(resized_img)
    sample = np.asarray(resized_img) * 256
    # Swaps axis of the image to transform it from (224, 224, 4) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # We substract the mean
    normed_img = sample - mean_img.asnumpy()
    normed_img.resize(1, 3, 224, 224)
    return normed_img
```


```python
df = pd.read_csv("./data/id_train.csv")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
#On charge le reseau pre entraine
prefix = "./model/Inception_BN"
num_round = 39
model = mx.model.FeedForward.load(prefix , num_round, ctx=mx.cpu(), numpy_batch_size=1)

#on charge l'image moyenne
mean_img = mx.nd.load("./model/mean_224.nd")["mean_img"]
f = open("./features_train.csv", 'w')
internals = model.symbol.get_internals()
fea_symbol = internals["global_pool_output"]
feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=1,
                             arg_params=model.arg_params, aux_params=model.aux_params,
                             allow_extra_params=True)

# column0 = image id, column 1= label, other columns = image features
for index, row in df.iterrows():
    imagePath = "./data/roof_images/"+str(row["Id"])+".jpg"
    batch = preprocess_image(imagePath, False, mean_img)
    # on extrait les features de l'image
    global_pooling_feature = feature_extractor.predict(batch)
    # on sauvegarde les features dans un fichier
    f.write(str(row["Id"])+","+str(row["label"]))
    f.write(",")
    for i in range(0, len(global_pooling_feature[0])-1):
        f.write(str(global_pooling_feature[0][i][0][0]))
        f.write(",")
    f.write(str(global_pooling_feature[0][len(global_pooling_feature[0])-1][0][0]))
    f.write("\n")

f.close()
```

# Classifieur supervisé

L'objectif est d'entraîner un classifieur supervisé, ici une SVM, grâce aux features extraites précédemment afin de classifier nos images dans l'une des 4 catégories : nord/sud, est/ouest, toit plat, autre.


```python
import numpy as np
import skimage.io as io
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
```

    /home/avastel/.local/lib/python3.5/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
df = pd.read_csv("./features_train.csv")
train, test = train_test_split(df, train_size = 0.8, test_size = 0.2)
X_train = train.ix[:,2:]
y_train = train.ix[:,1]
X_test = test.ix[:, 2:]
y_test = test.ix[:, 1]
```


```python
# Hyperparameters tuning
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
                       scoring='accuracy', n_jobs=4)
clf.fit(X_train, y_train)
print("Best parameters set found on development set:\n")
print(clf.best_params_)
```

    Best parameters set found on development set:
    
    {'kernel': 'rbf', 'gamma': 0.001, 'C': 10}



```python
print("Grid scores on development set:\n")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:\n")
print("The model is trained on the full development set.")
print("The score is computed on the full evaluation set.\n")
y_true, y_pred = y_test, clf.predict(X_test)
print("Accuracy : %f " % accuracy_score(y_test, y_pred))
```

    Grid scores on development set:
    
    0.747 (+/-0.022) for {'kernel': 'rbf', 'gamma': 0.001, 'C': 1}
    0.679 (+/-0.032) for {'kernel': 'rbf', 'gamma': 0.0001, 'C': 1}
    0.762 (+/-0.014) for {'kernel': 'rbf', 'gamma': 0.001, 'C': 10}
    0.746 (+/-0.016) for {'kernel': 'rbf', 'gamma': 0.0001, 'C': 10}
    0.746 (+/-0.007) for {'kernel': 'rbf', 'gamma': 0.001, 'C': 100}
    0.750 (+/-0.015) for {'kernel': 'rbf', 'gamma': 0.0001, 'C': 100}
    0.745 (+/-0.008) for {'kernel': 'rbf', 'gamma': 0.001, 'C': 1000}
    0.724 (+/-0.016) for {'kernel': 'rbf', 'gamma': 0.0001, 'C': 1000}
    0.693 (+/-0.015) for {'kernel': 'linear', 'C': 1}
    0.670 (+/-0.023) for {'kernel': 'linear', 'C': 10}
    0.667 (+/-0.018) for {'kernel': 'linear', 'C': 100}
    0.667 (+/-0.018) for {'kernel': 'linear', 'C': 1000}
    
    Detailed classification report:
    
    The model is trained on the full development set.
    The scores are computed on the full evaluation set.
    
    Accuracy : 0.766875 



```python
# train the svm with best hyperparameters)
model = svm.SVC(kernel="rbf", C=10.0, gamma=0.001)
model.fit(X_train, y_train)
```




    SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
# evaluate the accuracy of our model
predictions = model.predict(X_test)
print("Accuracy : %f " % accuracy_score(y_test, predictions))
```

    Accuracy : 0.770625 

