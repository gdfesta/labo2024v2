#!/usr/bin/env Rscript
cat( "ETAPA  z2202_HT_lightgbm_AUCROC.r  INIT\n")

# Hyperparameter Tuning  lightgbm

# inputs
#  * dataset  con particion  train, valitate, test
#  * hiperparametros fijos y variables (que van a la Bayesian Optimization )
# output  
#   archivo  BO_log.txt  resultado de la Bayesian Optimization

# limpio la memoria
rm(list = ls(all.names = TRUE)) # remove all objects
gc(full = TRUE, verbose= FALSE) # garbage collection

require("data.table", quietly=TRUE)
require("rlist", quietly=TRUE)
require("yaml", quietly=TRUE)

require("lightgbm", quietly=TRUE)
require("ROCR", quietly=TRUE)

# paquetes necesarios para la Bayesian Optimization
require("DiceKriging", quietly=TRUE)
require("mlrMBO", quietly=TRUE)


#cargo la libreria
# args <- c( "~/labo2024v2" )
args <- commandArgs(trailingOnly=TRUE)
source( paste0( args[1] , "/src/lib/mlog.r" ) )
source( paste0( args[1] , "/src/lib/action_lib.r" ) )

#------------------------------------------------------------------------------

GLOBAL_arbol <- 0L
GLOBAL_gan_max <- -Inf
vcant_optima <- c()

#------------------------------------------------------------------------------

EstimarGanancia_AUC_lightgbm <- function(x) {

  cat( "Inicio EstimarGanancia_AUC_lightgbm()\n")
  gc(verbose= FALSE)
  GLOBAL_iteracion <<- GLOBAL_iteracion + 1L
  envg$OUTPUT$BO$iteracion_actual <<- GLOBAL_iteracion
  GrabarOutput()

  # para que una siguiente corrida pueda retomar
  if( file.exists("bayesiana.RDATA") ) {

    cat(format(Sys.time(), "%Y%m%d %H%M%S"), "\n",
      file = "z-Rcanresume.txt",
      append = TRUE
     )
  }


  # hago la union de los parametros basicos y los moviles que vienen en x
  param_completo <- c(envg$PARAM$lgb_basicos, x)

  if( "extra_trees"  %in%  names(param_completo) )
   param_completo$extra_trees <- as.logical( param_completo$extra_trees )

  if( "is_unbalance"  %in%  names(param_completo) )
   param_completo$is_unbalance <- as.logical( param_completo$is_unbalance )


  param_completo$early_stopping_rounds <-
    as.integer(400 + 4 / param_completo$learning_rate)

  GLOBAL_arbol <<- 0L
  GLOBAL_gan_max <<- -Inf
  vcant_optima <<- c()
  set.seed(envg$PARAM$lgb_semilla, kind = "L'Ecuyer-CMRG")
  modelo_train <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalidate),
    eval = "auc", 
    param = param_completo,
    verbose = -100
  )

  cat("\n")

  cant_corte <- vcant_optima[modelo_train$best_iter]

  # aplico el modelo a testing y calculo la ganancia
  prediccion <- predict(
    modelo_train,
    data.matrix(dataset_test[, campos_buenos, with = FALSE])
  )

  tbl <- copy(dataset_test[, list("gan" = 
    ifelse(get(envg$PARAM$dataset_metadata$clase) %in%  envg$PARAM$train$positivos, 
       envg$PARAM$train$gan1, 
       envg$PARAM$train$gan0))])

  tbl[, prob := prediccion]
  setorder(tbl, -prob)
  tbl[, gan_acum := cumsum(gan)]
  tbl[, gan_suavizada := frollmean(
    x = gan_acum, n = envg$PARAM$train$meseta,
    align = "center", na.rm = TRUE, hasNA = TRUE
  )]


  ganancia_test <- tbl[, max(gan_suavizada, na.rm = TRUE)]

  cantidad_test_normalizada <- which.max(tbl[, gan_suavizada])

  # calculo la AUCROC
  curva_roc <-  ROCR::prediction(prediccion,
                   dataset_test$clase01,
                   label.ordering = c(0, 1))

  perf <-  ROCR::performance(curva_roc, "auc")
  AUC_test <- unlist(perf@y.values)


  rm(tbl)
  gc(verbose= FALSE)

  ganancia_test_normalizada <- ganancia_test

  # logueo final
  ds <- list("cols" = ncol(dtrain), "rows" = nrow(dtrain))
  xx <- c(ds, copy(param_completo))

  xx$early_stopping_rounds <- NULL
  xx$num_iterations <- modelo_train$best_iter
  xx$estimulos <- cantidad_test_normalizada
  xx$qsemillas <- 1L
  xx$AUC <- AUC_test
  xx$ganancia <- ganancia_test_normalizada
  xx$metrica <- AUC_test
  xx$iteracion_bayesiana <- GLOBAL_iteracion


  superacion <- FALSE
  # voy grabando las mejores column importance
  if (ganancia_test_normalizada > GLOBAL_AUC) {
    GLOBAL_AUC <<- AUC_test
    tb_importancia <- as.data.table(lgb.importance(modelo_train))

    fwrite(tb_importancia,
      file = paste0("impo_", sprintf("%03d", GLOBAL_iteracion), ".txt"),
      sep = "\t"
    )

    rm(tb_importancia)
    envg$OUTPUT$BO$mejor$iteracion <<- GLOBAL_iteracion
    envg$OUTPUT$BO$mejor$ganancia <<- ganancia_test_normalizada
    envg$OUTPUT$BO$mejor$metrica <<- GLOBAL_AUC
    envg$OUTPUT$BO$mejor$arboles <<- modelo_train$best_iter
    GrabarOutput()
    mlog_log(xx, arch = "BO_log_mejor.txt")

    t <- format(Sys.time(), "%Y%m%d %H%M%S")
    cat( t, "\n",
      file = "z-Rcanbypass.txt",
      append = TRUE
    )

    superacion <- TRUE
  }

  mlog_log(xx, arch = "BO_log.txt", parentreplicate= superacion)

  cat( "Fin EstimarGanancia_AUC_lightgbm()\n")
  set.seed(envg$PARAM$lgb_semilla, kind = "L'Ecuyer-CMRG")
  return(AUC_test)  # guio la optimizacion por AUC_test, FUNDAMENTAL
}
#------------------------------------------------------------------------------

parametrizar  <- function( lparam )
{
  param_fijos  <- copy( lparam )
  hs  <- list()

  for( param  in  names( lparam ) )
  {
    if( length( lparam[[ param ]] ) > 1 )
    {
      desde  <- as.numeric( lparam[[ param ]][[1]]  )
      hasta  <- as.numeric( lparam[[ param ]][[2]]  )

      if( length( lparam[[ param ]] ) == 2 )
      {
         hs  <- append( hs,  
                        list( makeNumericParam( param, lower= desde, upper= hasta)  ) )
      } else {
         hs  <- append( hs, 
                        list( makeIntegerParam( param, lower= desde, upper= hasta) ) )
      }

      param_fijos[[ param ]] <- NULL  #lo quito 
    }
  }

  return( list( "param_fijos" =  param_fijos,
                "paramSet"    =  hs ) )
}

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
cat( "ETAPA  z2201_HT_lightgbm_AUCROC.r  START\n")
action_inicializar() 

# cargo las semillas
envg$PARAM$lgb_semilla <- envg$PARAM$semilla


# apertura de los parametros de LightGBM
#  en los que van fijos directo al LightGBM
#  y los que pasan a formar parte de la Bayesian Optimization
apertura  <- parametrizar( envg$PARAM$lgb_param )
envg$PARAM$lgb_basicos <- apertura$param_fijos
envg$PARAM$bo_lgb <- makeParamSet( params= apertura$paramSet )

envg$PARAM$lgb_basicos$seed <- envg$PARAM$semilla

# cargo el dataset donde voy a entrenar
# esta en la carpeta del exp_input y siempre se llama  dataset.csv.gz
# cargo el dataset
envg$PARAM$dataset <- paste0( "./", envg$PARAM$input, "/dataset_training.csv.gz" )
envg$PARAM$dataset_metadata <- read_yaml( paste0( "./", envg$PARAM$input, "/dataset_metadata.yml" ) )

cat( "lectura del dataset\n")
action_verificar_archivo( envg$PARAM$dataset )
cat( "Iniciando lectura del dataset\n" )
dataset <- fread(envg$PARAM$dataset)
cat( "Finalizada lectura del dataset\n" )

if( "azar" %in% colnames(dataset) )
  dataset[, azar := NULL]

# Verificaciones
if (!("fold_train" %in% colnames(dataset))) {
  stop("Error, el dataset no tiene el campo fold_train \n")
}

if (!("fold_validate" %in% colnames(dataset))) {
  stop("Error, el dataset no tiene el campo fold_validate \n")
}

if (!("fold_test" %in% colnames(dataset))) {
  stop("Error, el dataset no tiene el campo fold_test  \n")
}

if (dataset[fold_train == 1, .N] == 0) {
  stop("Error, en el dataset no hay registros con fold_train==1 \n")
}


GrabarOutput()

cat(envg$PARAM$exp_input,
  file = "TrainingStrategy.txt",
  append = FALSE
)

# defino la clase binaria clase01
cat( "creacion clase01\n")
dataset[, clase01 :=  0L ]
dataset[ get(envg$PARAM$dataset_metadata$clase) %in% envg$PARAM$train$clase01_valor1, clase01 := 1L ]


# los campos que se pueden utilizar para la prediccion
cat( "creacion campos_buenos\n")
campos_buenos <- setdiff(
  copy(colnames(dataset)),
  c("clase01", envg$PARAM$dataset_metadata$clase, "fold_train", "fold_validate", "fold_test")
)

# la particion de train siempre va
cat( "creacion dtrain\n")
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[fold_train == 1, campos_buenos, with = FALSE]),
  label = dataset[fold_train == 1, clase01],
  weight = dataset[
    fold_train == 1,
    ifelse(get(envg$PARAM$dataset_metadata$clase) %in%  envg$PARAM$train$positivos, 1.0000001, 1.0 )
  ],
  free_raw_data = FALSE
)

envg$OUTPUT$train$ncol <- ncol(dtrain)
envg$OUTPUT$train$nrow <- nrow(dtrain)
envg$OUTPUT$train$periodos <- dataset[fold_train == 1, length(unique(get(envg$PARAM$dataset_metadata$periodo)))]

kvalidate <- FALSE
ktest <- FALSE
kcrossvalidation <- TRUE

# Si hay que hacer validacion
cat( "creacion dvalidate\n")
if (dataset[fold_train == 0 & fold_test == 0 & fold_validate == 1, .N] > 0) {
  kcrossvalidation <- FALSE
  kvalidate <- TRUE
  dvalidate <- lgb.Dataset(
    data = data.matrix(dataset[fold_validate == 1, campos_buenos, with = FALSE]),
    label = dataset[fold_validate == 1, clase01],
    weight = dataset[
      fold_validate == 1,
      ifelse(get(envg$PARAM$dataset_metadata$clase) %in%  envg$PARAM$train$positivos, 1.0000001, 1.0 )
    ],
    free_raw_data = FALSE
  )

  envg$OUTPUT$validate$ncol <- ncol(dvalidate)
  envg$OUTPUT$validate$nrow <- nrow(dvalidate)

  envg$OUTPUT$validate$periodos <- dataset[
    fold_validate == 1,
    length(unique(get(envg$PARAM$dataset_metadata$periodo)))
  ]
}


# Si hay que hacer testing
if (dataset[fold_train == 0 & fold_validate == 0 & fold_test == 1, .N] > 0) {
  cat( "creacion testing\n")
  ktest <- TRUE
  campos_buenos_test <- setdiff(
    copy(colnames(dataset)),
    c("fold_train", "fold_validate", "fold_test")
  )

  dataset_test <- dataset[fold_test == 1, campos_buenos_test, with = FALSE]

  envg$OUTPUT$test$ncol <- ncol(dataset_test)
  envg$OUTPUT$test$nrow <- nrow(dataset_test)
  envg$OUTPUT$test$periodos <- dataset_test[, length(unique(get(envg$PARAM$dataset_metadata$periodo)))]

}

# Inicializo MLFlow
mlog_init()

if (kcrossvalidation) {
  datahash <- mlog_table_hash( dataset[, envg$PARAM$dataset_metadata$primarykey, with=FALSE ] )
  mlflow_exp_det <- paste0( "/xval-", datahash )
} else {

  datahash <- mlog_table_hash( dataset_test[, envg$PARAM$dataset_metadata$primarykey, with=FALSE ] )
  mlflow_exp_det <- paste0( "/test-", datahash )
}


mlog_addfile("BO_log.txt",
             mlflow_exp= mlflow_exp_det,
             mlflow_run= envg$PARAM$experimento,
             cols_fijas= list(expw=envg$PARAM$experimento_largo) )

rm(dataset)
gc(verbose= FALSE)


# si ya existe el archivo log, traigo hasta donde procese
if (file.exists("BO_log.txt")) {
  tabla_log <- fread("BO_log.txt")
  GLOBAL_iteracion <- nrow(tabla_log)
  GLOBAL_AUC <- tabla_log[, max(AUC)]
  rm(tabla_log)
} else {
  GLOBAL_iteracion <- 0L
  GLOBAL_AUC <- -Inf
}


# Aqui comienza la configuracion de mlrMBO


envg$OUTPUT$crossvalidation <- kcrossvalidation
GrabarOutput()

# deobo hacer cross validation o  Train/Validate/Test
if (kcrossvalidation) {
  funcion_optimizar <- EstimarGanancia_AUC_lightgbmCV
} else {
  funcion_optimizar <- EstimarGanancia_AUC_lightgbm
}


configureMlr(show.learner.output = FALSE)

# configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
# por favor, no desesperarse por lo complejo
obj.fun <- makeSingleObjectiveFunction(
  fn = funcion_optimizar, # la funcion que voy a maximizar
  minimize = FALSE, # estoy Maximizando la ganancia
  noisy = TRUE,
  par.set = envg$PARAM$bo_lgb, # definido al comienzo del programa
  has.simple.signature = FALSE # paso los parametros en una lista
)

# archivo donde se graba y cada cuantos segundos
ctrl <- makeMBOControl(
  save.on.disk.at.time = 60,
  save.file.path = "bayesiana.RDATA"
)

ctrl <- setMBOControlTermination(ctrl,
  iters = envg$PARAM$bo_iteraciones
) # cantidad de iteraciones

ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

# establezco la funcion que busca el maximo
surr.km <- makeLearner("regr.km",
  predict.type = "se",
  covtype = "matern3_2",
  control = list(trace = TRUE)
)

surr.km <- makeLearner("regr.km",
  predict.type = "se",
  covtype = "matern3_2",
  optim.method = "BFGS",
  nugget.estim = TRUE,
  jitter = TRUE,
  control = list(trace = TRUE)
)


# Aqui inicio la optimizacion bayesiana
set.seed(envg$PARAM$lgb_semilla, kind = "L'Ecuyer-CMRG")
if (!file.exists("bayesiana.RDATA")) {
  run <- mbo(obj.fun, learner = surr.km, control = ctrl)
} else {
  # si ya existe el archivo RDATA,
  # debo continuar desde el punto hasta donde llegue
  #  usado para cuando se corta la virtual machine
  run <- mboContinue("bayesiana.RDATA") # retomo en caso que ya exista
}

#------------------------------------------------------------------------------
BO_log <- fread("BO_log.txt")
envg$OUTPUT$ganancia_max <- BO_log[, max(ganancia, na.rm = TRUE)]

envg$OUTPUT$time$end <- format(Sys.time(), "%Y%m%d %H%M%S")
GrabarOutput()
#------------------------------------------------------------------------------

# ya no tiene sentido retomar, se termino el trabajo
file.remove("z-Rcanresume.txt")
#------------------------------------------------------------------------------

# finalizo la corrida
#  archivos tiene a los files que debo verificar existen para no abortar

action_finalizar( archivos = c("BO_log.txt")) 
cat( "ETAPA  z2202_HT_lightgbm_AUCROC.r  END\n")