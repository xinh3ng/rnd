#!/user/bin/env bash

printf "Ensure Python environment is already activated.\n"
# unset PYTHONPATH  # Might need this if spyder cannot be launched

python -c "from spyder.app import start; start.main()" &
printf "Spyder launched.\n"
