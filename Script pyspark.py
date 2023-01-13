from pyspark.sql import SparkSession

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

import pandas as pd
from PIL import Image
import numpy as np
import io

from pyspark.sql.functions import col, pandas_udf, PandasUDFType

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import Row


spark = SparkSession.builder.getOrCreate()

model = ResNet50(include_top=False)

sc = spark.sparkContext
bc_model_weights = sc.broadcast(model.get_weights())


def model_fn():
    """
      Retourne un modèle ResNet50 sans les couches de classification et avec les poids pré-entrainés ajoutés.
    
    """
    model = ResNet50(weights=None, include_top=False)
    model.set_weights(bc_model_weights.value)
    return model


def preprocess(content):
    """
      Preprocess les images d'entrées pour le modèle ResNet50.
      
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)


def featurize(model, content_series):
    """
    Retourne une pd.Series de features d'images obtenues grâce au ResNet50.
    
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    output = [p.flatten() for p in preds]
    return pd.Series(output)


@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    Retourne une Spark DataFrame colonne de type ArrayType(FloatType) contenant les features obtenues grâce au ResNet50.
    'content_series_iter' permet d'itérer sur les batchs de données contenant les images d'entrées.
    
    '''
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize(model, content_series)
        
        

def main():
    # On charge les images à partir de S3.
    images = spark.read.format("binaryFile") \
                        .option("pathGlobFilter", "*.jpg") \
                        .option("recursiveFileLookup", "true") \
                        .load("s3a://oc-p8-compartiment/sample fruits 360/")
                        
    # On crée un Spark DataFrame contenant 2 colonnes : une pour le 'path' des images et une autre pour les features
    # obtenues grâce au ResNet50.
    features_df = images.select(col("path"), featurize_udf("content").alias("features"))
    
    # On collecte les features dans une liste.
    list_features = features_df.select('features').collect()

    list_features_vector = []

    # On convertit chaque ensemble de features en 'dense vectors' pour pouvoir les paralléliser et les mettre dans
    # une RowMatrix pour la PCA.
    for i in range(len(list_features)):
        list_features_vector.append(Vectors.dense(list_features[i][0]))
        
    rows = sc.parallelize(list_features_vector)

    rm = RowMatrix(rows)

    # On réalise une PCA à deux composantes principales.
    pca = rm.computePrincipalComponents(2)

    # On projète les features de chaque image dans le nouvel espace à deux composantes principales.
    dim_reduc = rm.multiply(pca).rows.collect()
    
    list_path = features_df.select('path').collect()

    path_dim_reduc_list =[]
    
    for i, j in enumerate(dim_reduc):
        path_dim_reduc_list.append((list_path[i][0], j))
    
    # On crée un Spark DataFrame contenant 2 colonnes : une contenant le 'path' vers les images et une contenant 
    # la réduction de dimension des features.    
    dim_reduc_df = spark.createDataFrame(data=path_dim_reduc_list, schema = ["path", "features after pca 2 components"])

    # On stocke les résultats sur S3 au format JSON.
    dim_reduc_df.write.mode('overwrite').json('s3a://oc-p8-compartiment/output p8/')
    
    
    
if __name__ == '__main__':
    main()