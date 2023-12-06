# setup script
echo "creating virtual environment"
python3 -m venv .venv
source .venv/bin/activate
echo "installing local utilities."
python -m pip install -e .
echo "making directories."
mkdir -p figures data results tables
