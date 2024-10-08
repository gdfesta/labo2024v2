# Arbol elemental con libreria  rpart
# Debe tener instaladas las librerias  data.table ,  rpart  y  rpart.plot
# Correr en Google Cloud con RStudio

# cargo las librerias que necesito
require("data.table")
require("rpart")
require("rpart.plot")

# cargo el dataset pequeno vivencial del disco local
dataset <- fread("/Users/gaston.festa/Documents/studies/master-data-science/labo1/datasets/clase_3_5.csv")

# genero el modelo,  aqui se construye el arbol
# quiero predecir clase_ternaria a partir de el resto de las variables
modelo <- rpart(
    formula = "ganancia_promedio ~ .",
    data = dataset, # los datos donde voy a entrenar
    xval = 0,
    cp = -0.3, # esto significa no limitar la complejidad de los splits
    minsplit = 0, # minima cantidad de registros para que se haga el split
    minbucket = 1, # tamaño minimo de una hoja
    maxdepth = 3  # profundidad maxima del arbol
)


# grafico el arbol
prp(modelo,
    extra = 101, digits = -5,
    branch = 1, type = 4, varlen = 0, faclen = 0
)

