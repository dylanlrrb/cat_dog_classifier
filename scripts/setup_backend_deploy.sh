# copy client into backend directory
cp -r client/ backend/static
# copy model into /model directory from s3 specified in spec
# install dependencies for flask app and ML framework
pip install -r backend/requirements.txt
