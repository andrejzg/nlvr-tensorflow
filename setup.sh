git clone https://github.com/clic-lab/nlvr.git /tmp/nlvr
mkdir data
mv /tmp/nlvr/test data/
mv /tmp/nlvr/dev data/
mv /tmp/nlvr/train data/
mv /tmp/nlvr/metrics_images.py .
rm -rf /tmp/nlvr
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
pip install requirements.txt
