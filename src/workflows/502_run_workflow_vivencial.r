require("rlang")

# workflow que voy a correr
PARAM <- "src/workflows/527_workflow_base_f202107_viv.r"

envg <- env()

envg$EXPENV <- list()
envg$EXPENV$repo_dir <- "~/labo2024v2/"

#------------------------------------------------------------------------------

correr_workflow <- function( wf_scriptname )
{
  dir.create( "~/tmp", showWarnings = FALSE)
  setwd("~/tmp" )

  # creo el script que corre el experimento
  comando <- paste0( 
      "#!/bin/bash\n", 
      "source /home/$USER/.venv/bin/activate\n",
      "nice -n 15 Rscript --vanilla ",
      envg$EXPENV$repo_dir,
      wf_scriptname,
      "   ",
      wf_scriptname,
     "\n",
     "deactivate\n"
    )
  cat( comando, file="run.sh" )

  Sys.chmod( "run.sh", mode = "744", use_umask = TRUE)

  system( "./run.sh" )
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# aqui efectivamente llamo al workflow
correr_workflow( PARAM )