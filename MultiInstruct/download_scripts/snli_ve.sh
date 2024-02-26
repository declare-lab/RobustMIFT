cd raw_datasets

mkdir SNLI-VE
cd SNLI-VE

echo "Downloading SNLI..."

git clone https://github.com/necla-ml/SNLI-VE.git

cd SNLI-VE
pip install jsonlines
python ./vet/tools/snli_ve_generator.py

cd data
./download # y to all if necessary
