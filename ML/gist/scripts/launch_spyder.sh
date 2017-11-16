#!/user/bin/env bash

unset PYTHONPATH  #
python -c "from spyder.app import start; start.main()" &
printf "Spyder launched.\n"
