#!/bin/bash
# Run setup instructions as a single script
if [ -z "$DATA_ROOT" ]; then
    echo "I do not know where the data is. Please show me the data. Set DATA_ROOT environment variable such that $DATA_ROOT/data/radish/map_pgm contains the bag files."
fi

pip install -r requirements.txt
cd $CODE_DIR/data && ln -s $DATA_ROOT/data/radish
mkdir -p $DATA_ROOT/logs && cd $CODE_DIR && ln -s $DATA_ROOT/logs
