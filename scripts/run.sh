APP_PATH=$PWD"/../"
INPUT_PATH=$PWD"/"input
OUT_NAME=$PWD"/"
SLURM_FILE=
NTHREADS=
APP_ARGS=

if [[ "$ARCH" == "HASWELL" ]]; then
  APP_PATH=$APP_PATH"MaskedSpGEMM-haswell/build"
  RUN_ARGS=$RUN_ARGS" --constraint haswell"
  NTHREADS=32
elif [[ "$ARCH" == "KNL" ]]; then
  APP_PATH=$APP_PATH"MaskedSpGEMM-knl/build"
  RUN_ARGS=$RUN_ARGS" --constraint knl"
  NTHREADS=68
else
  echo Unknown arch: $ARCH
  exit 1
fi

if [[ "$APP" == "TRICNT" ]]; then
  APP_PATH=$APP_PATH"/tricnt-all-grb"
  SLURM_FILE="tricnt.slurm"
  INPUT_PATH=$INPUT_PATH"-tricnt"
  OUT_NAME=$OUT_NAME"tricnt"
elif [[ "$APP" == "KTRUSS" ]]; then
  APP_PATH=$APP_PATH"/ktruss-all-grb"
  SLURM_FILE="ktruss.slurm"
  INPUT_PATH=$INPUT_PATH"-ktruss"
  OUT_NAME=$OUT_NAME"ktruss"
  APP_ARGS=$K
elif [[ "$APP" == "BC" ]]; then
  APP_PATH=$APP_PATH"/bc-all-grb"
  SLURM_FILE="bc.slurm"
  INPUT_PATH=$INPUT_PATH"-bc"
  OUT_NAME=$OUT_NAME"bc"
else
  echo Unknown app: $APP
  exit 1
fi

if [[ "$TYPE" == "RMAT" ]]; then
  INPUT_PATH=$INPUT_PATH"-rmat"
  OUT_NAME=$OUT_NAME"-rmat"
elif [[ "$TYPE" == "KNL26" ]]; then
  INPUT_PATH=$INPUT_PATH"-knl26"
  OUT_NAME=$OUT_NAME"-knl26"
else
  echo Unknown input type: $TYPE
  exit 1
fi

if [[ "$ARCH" == "HASWELL" ]]; then
  OUT_NAME=$OUT_NAME"-haswell"
elif [[ "$ARCH" == "KNL" ]]; then
  OUT_NAME=$OUT_NAME"-knl"
fi

OUT_NAME=$OUT_NAME"-"$(date +%s)".out"

RUN_ARGS=$RUN_ARGS" -o "$OUT_NAME

APP_PATH=$APP_PATH INPUT_PATH=$INPUT_PATH APP_ARGS=$APP_ARGS NTHREADS=$NTHREADS sbatch $RUN_ARGS $SLURM_FILE
