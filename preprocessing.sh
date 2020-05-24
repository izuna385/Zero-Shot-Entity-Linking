mkdir ./data/
mkdir ./data/worlds/

FILE_ID=1ZcKZ1is0VEkY9kNfPxIG19qEIqHE5LIO
FILE_NAME=zeshel.tar.bz2
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}

mv zeshel.tar.bz2 ./data/
cd data
mkdir mentions_split_by_world
tar xf zeshel.tar.bz2
cd zeshel
mv * ./../
cd ./../
rm -r zeshel
cd ..
python3 ./src/preprocess.py